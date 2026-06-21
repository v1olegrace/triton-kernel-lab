"""Fused RMSNorm forward and two-stage backward with autograd integration.

The forward computes RMS statistics in FP32 and stores them for
backpropagation. The backward computes ``dx`` per row and accumulates partial
``dweight`` values into lock-protected buffers before a second kernel reduces
those buffers across lock groups. No bias, no mean centering.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tklab.registry import (
    BenchmarkCall,
    BenchmarkScalar,
    KernelSpec,
    TensorArgs,
    register,
)

_ROWS = 4096
_EPS = 1e-5
_MAX_FEATURE_BYTES = 65_536
_STAGE2_BLOCK_M = 32
_STAGE2_BLOCK_N = 128
_MAX_INT32_OFFSET = 2**31 - 1


@triton.jit  # type: ignore[untyped-decorator]
def _rms_norm_forward_kernel(
    x_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    output_ptr: tl.tensor,
    rstd_ptr: tl.tensor,
    x_row_stride: int,
    output_row_stride: int,
    n_cols: int,
    eps: float,
    STORE_STATS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Normalize one row by its RMS and optionally stash the reciprocal RMS."""
    row = tl.program_id(axis=0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < n_cols

    values = tl.load(
        x_ptr + row * x_row_stride + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mean_sq = tl.sum(values * values, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    output = values * rstd * weight
    tl.store(
        output_ptr + row * output_row_stride + columns,
        output,
        mask=mask,
    )

    if STORE_STATS:
        tl.store(rstd_ptr + row, rstd)


@triton.jit  # type: ignore[untyped-decorator]
def _rms_norm_backward_stage1_kernel(
    dx_ptr: tl.tensor,
    dy_ptr: tl.tensor,
    incoming_dx_ptr: tl.tensor,
    partial_dw_ptr: tl.tensor,
    x_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    rstd_ptr: tl.tensor,
    lock_ptr: tl.tensor,
    x_row_stride: int,
    dy_row_stride: int,
    incoming_dx_row_stride: int,
    dx_row_stride: int,
    n_cols: int,
    GROUP_SIZE_M: tl.constexpr,
    ADD_INCOMING_DX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Compute one row's ``dx`` and lock-reduced weight-gradient partials."""
    row = tl.program_id(axis=0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < n_cols

    values = tl.load(
        x_ptr + row * x_row_stride + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    dy = tl.load(
        dy_ptr + row * dy_row_stride + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)

    # x_hat = x * rstd (no centering — RMSNorm has no mean subtraction)
    x_hat = tl.where(mask, values * rstd, 0.0)
    g = tl.where(mask, weight * dy, 0.0)
    # dx_j = rstd * (g_j - x_hat_j * (1/N) * Σ_i g_i * x_hat_i)
    mean_g_xhat = tl.sum(g * x_hat, axis=0) / n_cols
    dx = rstd * (g - x_hat * mean_g_xhat)
    if ADD_INCOMING_DX:
        incoming_dx = tl.load(
            incoming_dx_ptr + row * incoming_dx_row_stride + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dx += incoming_dx
    tl.store(dx_ptr + row * dx_row_stride + columns, dx, mask=mask)

    partial_dw = dy * x_hat
    lock_id = row % GROUP_SIZE_M
    lock = lock_ptr + lock_id
    count = lock_ptr + GROUP_SIZE_M + lock_id
    partial_offsets = lock_id * n_cols + columns
    group_dw = partial_dw_ptr + partial_offsets

    while tl.atomic_cas(lock, 0, 1, sem="acq_rel", scope="gpu") == 1:
        pass
    initialized = tl.load(count)
    if initialized == 0:
        tl.atomic_xchg(count, 1, sem="acq_rel", scope="gpu")
    else:
        partial_dw += tl.load(group_dw, mask=mask, other=0.0)
    tl.store(group_dw, partial_dw, mask=mask)
    # Join every lane's vector stores before the device-scoped unlock publishes them.
    tl.debug_barrier()
    tl.atomic_xchg(lock, 0, sem="acq_rel", scope="gpu")


@triton.jit  # type: ignore[untyped-decorator]
def _rms_norm_backward_stage2_kernel(
    partial_dw_ptr: tl.tensor,
    dw_ptr: tl.tensor,
    group_count: int,
    n_cols: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Reduce lock-group partials into the final weight gradient."""
    column_block = tl.program_id(axis=0)
    columns = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
    dw = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for row_start in range(0, group_count, BLOCK_M):
        rows = row_start + tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < group_count) & (columns[None, :] < n_cols)
        offsets = rows[:, None] * n_cols + columns[None, :]
        dw += tl.load(partial_dw_ptr + offsets, mask=mask, other=0.0)

    tl.store(dw_ptr + columns, tl.sum(dw, axis=0), mask=columns < n_cols)


def _validate(x: torch.Tensor, weight: torch.Tensor) -> None:
    """Validate the supported RMSNorm tensor contract."""
    if x.ndim != 2:
        raise ValueError("rms_norm expects a 2D tensor shaped (rows, columns)")
    rows, columns = x.shape
    if rows == 0 or columns == 0:
        raise ValueError("rms_norm does not support empty dimensions")
    if x.stride(1) != 1:
        raise ValueError("rms_norm requires a contiguous final dimension")
    if weight.ndim != 1:
        raise ValueError("weight must be one-dimensional")
    if weight.shape != (columns,):
        raise ValueError("weight must match the final input dimension")
    if weight.stride(0) != 1:
        raise ValueError("weight must be contiguous")
    if x.device != weight.device:
        raise ValueError("input and weight must be on the same device")
    if x.device.type != "cuda":
        raise ValueError("rms_norm requires CUDA tensors")
    if x.dtype != weight.dtype:
        raise ValueError("input and weight must have the same dtype")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("rms_norm supports float16 and float32 tensors")
    if columns * x.element_size() > _MAX_FEATURE_BYTES:
        raise ValueError(
            f"feature row uses {columns * x.element_size()} bytes; "
            f"the fused limit is {_MAX_FEATURE_BYTES}"
        )
    max_relative_offset = (rows - 1) * x.stride(0) + columns - 1
    if max_relative_offset > _MAX_INT32_OFFSET:
        raise ValueError("input strides exceed the kernel's signed int32 offset range")


def _block_size_and_warps(columns: int, element_size: int) -> tuple[int, int]:
    """Return a power-of-two row block and its warp count."""
    max_elements = _MAX_FEATURE_BYTES // element_size
    block_size = min(max_elements, triton.next_power_of_2(columns))
    if columns > block_size:
        raise ValueError("feature dimension exceeds the fused RMSNorm limit")
    num_warps = min(max(block_size // 256, 1), 8)
    return block_size, num_warps


def _group_size_m(columns: int) -> int:
    """Select the number of independent weight-gradient lock groups."""
    if columns <= 1024:
        return 256
    if columns <= 4096:
        return 128
    if columns <= 8192:
        return 96
    return 64


@dataclass(frozen=True, slots=True)
class _BackwardBuffers:
    """Preallocated outputs and partial-reduction state for backward."""

    dx: torch.Tensor
    partial_dw: torch.Tensor
    dw: torch.Tensor
    locks: torch.Tensor
    group_size: int
    group_count: int
    block_size: int
    num_warps: int


def _launch_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    rstd: torch.Tensor,
    *,
    eps: float,
    store_stats: bool,
) -> None:
    """Launch RMSNorm forward into preallocated output and statistics."""
    _validate(x, weight)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive")
    if output.shape != x.shape or output.device != x.device or output.dtype != x.dtype:
        raise ValueError("output metadata must match the input")
    if output.stride(1) != 1:
        raise ValueError("rms_norm output requires a contiguous final dimension")
    rows, columns = x.shape
    if store_stats:
        if rstd.shape != (rows,):
            raise ValueError("rstd must contain one float32 value per row")
        if rstd.dtype != torch.float32:
            raise ValueError("rstd must use float32")
        if rstd.device != x.device:
            raise ValueError("rstd must be on the input device")

    block_size, num_warps = _block_size_and_warps(columns, x.element_size())
    _rms_norm_forward_kernel[(rows,)](
        x,
        weight,
        output,
        rstd,
        x.stride(0),
        output.stride(0),
        columns,
        eps,
        STORE_STATS=store_stats,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def _make_backward_buffers(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    group_size: int | None = None,
) -> _BackwardBuffers:
    """Allocate backward outputs and lock-group partial buffers."""
    rows, columns = x.shape
    block_size, num_warps = _block_size_and_warps(columns, x.element_size())
    selected_group_size = _group_size_m(columns) if group_size is None else group_size
    if selected_group_size <= 0:
        raise ValueError("group_size must be positive")
    group_count = min(selected_group_size, rows)
    return _BackwardBuffers(
        dx=torch.empty(x.shape, dtype=x.dtype, device=x.device),
        partial_dw=torch.empty(
            (selected_group_size, columns),
            dtype=torch.float32,
            device=x.device,
        ),
        dw=torch.empty_like(weight),
        locks=torch.zeros(2 * selected_group_size, dtype=torch.int32, device=x.device),
        group_size=selected_group_size,
        group_count=group_count,
        block_size=block_size,
        num_warps=num_warps,
    )


def _launch_backward_stage1(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    buffers: _BackwardBuffers,
    *,
    incoming_dx: torch.Tensor | None = None,
) -> None:
    """Launch per-row ``dx`` and lock-protected weight-gradient partials.

    ``incoming_dx`` is the optional direct gradient of a fused residual-sum
    output. When present, it is added to the RMSNorm input gradient inside the
    same program before the final store.
    """
    rows, columns = x.shape
    incoming = dy if incoming_dx is None else incoming_dx
    _rms_norm_backward_stage1_kernel[(rows,)](
        buffers.dx,
        dy,
        incoming,
        buffers.partial_dw,
        x,
        weight,
        rstd,
        buffers.locks,
        x.stride(0),
        dy.stride(0),
        incoming.stride(0),
        buffers.dx.stride(0),
        columns,
        GROUP_SIZE_M=buffers.group_size,
        ADD_INCOMING_DX=incoming_dx is not None,
        BLOCK_SIZE=buffers.block_size,
        num_warps=buffers.num_warps,
    )


def _launch_backward_stage2(
    buffers: _BackwardBuffers,
    columns: int,
) -> None:
    """Launch final reduction of weight-gradient partials."""
    _rms_norm_backward_stage2_kernel[(triton.cdiv(columns, _STAGE2_BLOCK_N),)](
        buffers.partial_dw,
        buffers.dw,
        buffers.group_count,
        columns,
        BLOCK_M=_STAGE2_BLOCK_M,
        BLOCK_N=_STAGE2_BLOCK_N,
        num_warps=4,
    )


def _launch_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    *,
    incoming_dx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch both RMSNorm backward stages and return all gradients."""
    if dy.shape != x.shape or dy.device != x.device or dy.dtype != x.dtype:
        raise ValueError("upstream gradient metadata must match the input")
    if dy.stride(1) != 1:
        dy = dy.contiguous()
    rows, columns = dy.shape
    max_relative_offset = (rows - 1) * dy.stride(0) + columns - 1
    if max_relative_offset > _MAX_INT32_OFFSET:
        raise ValueError("upstream gradient strides exceed signed int32 offsets")
    if incoming_dx is not None:
        if (
            incoming_dx.shape != x.shape
            or incoming_dx.device != x.device
            or incoming_dx.dtype != x.dtype
        ):
            raise ValueError("incoming input gradient metadata must match the input")
        if incoming_dx.stride(1) != 1:
            incoming_dx = incoming_dx.contiguous()
        max_incoming_offset = (rows - 1) * incoming_dx.stride(0) + columns - 1
        if max_incoming_offset > _MAX_INT32_OFFSET:
            raise ValueError("incoming input gradient strides exceed signed int32 offsets")
    if torch.are_deterministic_algorithms_enabled():
        message = (
            "Triton RMSNorm backward uses an order-dependent lock reduction "
            "and is not deterministic"
        )
        if torch.is_deterministic_algorithms_warn_only_enabled():
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        else:
            raise RuntimeError(message)

    buffers = _make_backward_buffers(x, weight)
    _launch_backward_stage1(
        dy,
        x,
        weight,
        rstd,
        buffers,
        incoming_dx=incoming_dx,
    )
    _launch_backward_stage2(buffers, x.shape[1])
    return buffers.dx, buffers.dw


class _RMSNormFunction(torch.autograd.Function):
    """Autograd bridge for the Triton RMSNorm kernels."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Run forward and save FP32 RMS statistics for backward."""
        _validate(x, weight)
        output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        rstd = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
        _launch_forward(x, weight, output, rstd, eps=eps, store_stats=True)
        ctx.save_for_backward(x, weight, rstd)
        return output

    @staticmethod
    def backward(
        ctx: Any,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Compute input and weight gradients."""
        x, weight, rstd = ctx.saved_tensors
        dx, dw = _launch_backward(dy, x, weight, rstd)
        return dx, dw, None


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = _EPS,
) -> torch.Tensor:
    """Apply RMSNorm over the final dimension with Triton autograd.

    FP16 is intended for inference-oriented throughput experiments. FP32 is
    supported for numerically strict gradient validation. Backward uses an
    order-dependent reduction and rejects PyTorch's deterministic mode.
    """
    result = _RMSNormFunction.apply(x, weight, eps)  # type: ignore[no-untyped-call]
    return cast(torch.Tensor, result)


def _torch_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Return PyTorch RMSNorm with the lab's default epsilon."""
    return F.rms_norm(x, (x.shape[-1],), weight, _EPS)


def _naive_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Compose RMSNorm from ordinary PyTorch pointwise and reduction ops."""
    mean_square = torch.mean(x.float() * x.float(), dim=-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(mean_square + _EPS)
    return (normalized * weight.float()).to(x.dtype)


def _make_output(args: TensorArgs) -> torch.Tensor:
    """Allocate a contiguous forward output."""
    x = args[0]
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


def _launch(args: TensorArgs, output: torch.Tensor) -> None:
    """Launch output-only forward for correctness and fallback benchmarking."""
    x, weight = args
    _launch_forward(x, weight, output, output, eps=_EPS, store_stats=False)


def _make_benchmark_call(args: TensorArgs, output: torch.Tensor) -> BenchmarkCall:
    """Prepare saved-statistic buffers outside the timed region."""
    x, weight = args
    rstd = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    return partial(
        _launch_forward,
        x,
        weight,
        output,
        rstd,
        eps=_EPS,
        store_stats=True,
    )


def _make_reference_call(args: TensorArgs, output: torch.Tensor) -> BenchmarkCall:
    """Prepare PyTorch's native RMSNorm baseline.

    PyTorch 2.12 exposes no ``aten.rms_norm.out`` overload, so the native
    baseline necessarily returns a newly allocated output.
    """
    x, weight = args

    def launch() -> object:
        """Call PyTorch RMSNorm for the benchmark baseline."""
        return F.rms_norm(x, (x.shape[-1],), weight, _EPS)

    return launch


def _make_inputs(
    columns: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TensorArgs:
    """Create a fixed-row RMSNorm benchmark problem."""
    x = torch.randn(_ROWS, columns, device=device, dtype=dtype)
    weight = torch.randn(columns, device=device, dtype=dtype)
    return x, weight


def _make_adversarial(device: torch.device) -> TensorArgs:
    """Create row-strided FP32 input with a non-power-of-two width."""
    rows, columns = 37, 1000
    storage = torch.randn(rows * 2, columns, device=device, dtype=torch.float32)
    x = storage[::2]
    weight = torch.randn(columns, device=device, dtype=torch.float32)
    return x, weight


def _bytes_moved(columns: int, dtype: torch.dtype) -> int:
    """Return effective input-read plus output-write traffic.

    Weight and reciprocal-RMS traffic is amortized across 4096 rows and
    intentionally excluded from the roofline numerator.
    """
    element_size = torch.empty((), dtype=dtype).element_size()
    return 2 * _ROWS * columns * element_size


def _benchmark_metadata(
    columns: int,
    dtype: torch.dtype,
) -> dict[str, BenchmarkScalar]:
    """Describe the two RMSNorm benchmark baselines."""
    del columns, dtype
    return {
        "reference_baseline": "torch.nn.functional.rms_norm",
        "naive_baseline": "manual PyTorch pointwise and reduction composition",
        "reference_allocates_output": True,
    }


RMS_NORM = register(
    KernelSpec(
        name="rms_norm_forward",
        description="Fused row-wise RMSNorm with FP32 statistics and autograd backward.",
        triton_fn=rms_norm,
        launch_fn=_launch,
        ref_fn=_torch_rms_norm,
        make_inputs=_make_inputs,
        make_output=_make_output,
        make_adversarial=_make_adversarial,
        sizes=(128, 256, 512, 1024, 2048, 4096, 8192, 16384),
        correctness_sizes=(128, 1024, 4096),
        bound="memory",
        bytes_moved=_bytes_moved,
        dtypes=(torch.float16,),
        naive_fn=_naive_rms_norm,
        benchmark_metadata=_benchmark_metadata,
        benchmark_call_factory=_make_benchmark_call,
        reference_call_factory=_make_reference_call,
        supports_interpreter=False,
    )
)
