"""Fused LayerNorm forward and two-stage backward with autograd integration.

The forward computes row statistics in FP32 and optionally stores them for
backpropagation. The backward computes ``dx`` per row and accumulates partial
``dweight``/``dbias`` values into lock-protected buffers before a second
kernel reduces those buffers across lock groups.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from tklab.harness.addressing import assert_int32_addressable
from tklab.kernels._norm_common import (
    EPS,
    MAX_FEATURE_BYTES,
    STAGE2_BLOCK_M,
    STAGE2_BLOCK_N,
    block_size_and_warps,
    group_size_m,
    validate_epsilon,
)
from tklab.registry import BenchmarkCall, KernelSpec, TensorArgs, register

_ROWS = 4096


@triton.jit  # type: ignore[untyped-decorator]
def _layer_norm_forward_kernel(
    x_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    output_ptr: tl.tensor,
    mean_ptr: tl.tensor,
    rstd_ptr: tl.tensor,
    x_row_stride: int,
    output_row_stride: int,
    n_cols: int,
    eps: float,
    STORE_STATS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Normalize one row and optionally stash its mean and reciprocal std."""
    row = tl.program_id(axis=0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < n_cols

    values = tl.load(
        x_ptr + row * x_row_stride + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(values, axis=0) / n_cols
    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(variance + eps)

    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    output = centered * rstd * weight + bias
    tl.store(
        output_ptr + row * output_row_stride + columns,
        output,
        mask=mask,
    )

    if STORE_STATS:
        tl.store(mean_ptr + row, mean)
        tl.store(rstd_ptr + row, rstd)


@triton.jit  # type: ignore[untyped-decorator]
def _layer_norm_backward_stage1_kernel(
    dx_ptr: tl.tensor,
    dy_ptr: tl.tensor,
    partial_dw_ptr: tl.tensor,
    partial_db_ptr: tl.tensor,
    x_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    mean_ptr: tl.tensor,
    rstd_ptr: tl.tensor,
    lock_ptr: tl.tensor,
    x_row_stride: int,
    dy_row_stride: int,
    dx_row_stride: int,
    n_cols: int,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Compute one row's ``dx`` and lock-reduced parameter partials."""
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
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)

    x_hat = tl.where(mask, (values - mean) * rstd, 0.0)
    weighted_dy = tl.where(mask, weight * dy, 0.0)
    mean_weighted_dy = tl.sum(weighted_dy, axis=0) / n_cols
    mean_weighted_dy_xhat = tl.sum(weighted_dy * x_hat, axis=0) / n_cols
    dx = rstd * (weighted_dy - mean_weighted_dy - x_hat * mean_weighted_dy_xhat)
    tl.store(dx_ptr + row * dx_row_stride + columns, dx, mask=mask)

    partial_dw = dy * x_hat
    partial_db = dy
    lock_id = row % GROUP_SIZE_M
    lock = lock_ptr + lock_id
    count = lock_ptr + GROUP_SIZE_M + lock_id
    partial_offsets = lock_id * n_cols + columns
    group_dw = partial_dw_ptr + partial_offsets
    group_db = partial_db_ptr + partial_offsets

    while tl.atomic_cas(lock, 0, 1, sem="acq_rel", scope="gpu") == 1:
        pass
    initialized = tl.load(count)
    if initialized == 0:
        tl.atomic_xchg(count, 1, sem="acq_rel", scope="gpu")
    else:
        partial_dw += tl.load(group_dw, mask=mask, other=0.0)
        partial_db += tl.load(group_db, mask=mask, other=0.0)
    tl.store(group_dw, partial_dw, mask=mask)
    tl.store(group_db, partial_db, mask=mask)
    # Join every lane's vector stores before the device-scoped unlock publishes them.
    tl.debug_barrier()
    tl.atomic_xchg(lock, 0, sem="acq_rel", scope="gpu")


@triton.jit  # type: ignore[untyped-decorator]
def _layer_norm_backward_stage2_kernel(
    partial_dw_ptr: tl.tensor,
    partial_db_ptr: tl.tensor,
    dw_ptr: tl.tensor,
    db_ptr: tl.tensor,
    group_count: int,
    n_cols: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Reduce lock-group partials into final parameter gradients."""
    column_block = tl.program_id(axis=0)
    columns = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
    dw = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for row_start in range(0, group_count, BLOCK_M):
        rows = row_start + tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < group_count) & (columns[None, :] < n_cols)
        offsets = rows[:, None] * n_cols + columns[None, :]
        dw += tl.load(partial_dw_ptr + offsets, mask=mask, other=0.0)
        db += tl.load(partial_db_ptr + offsets, mask=mask, other=0.0)

    tl.store(dw_ptr + columns, tl.sum(dw, axis=0), mask=columns < n_cols)
    tl.store(db_ptr + columns, tl.sum(db, axis=0), mask=columns < n_cols)


def _validate(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> None:
    """Validate the supported LayerNorm tensor contract."""
    if x.ndim != 2:
        raise ValueError("layer_norm expects a 2D tensor shaped (rows, columns)")
    rows, columns = x.shape
    if rows == 0 or columns == 0:
        raise ValueError("layer_norm does not support empty dimensions")
    if x.stride(1) != 1:
        raise ValueError("layer_norm requires a contiguous final dimension")
    if weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("weight and bias must be one-dimensional")
    if weight.shape != (columns,) or bias.shape != (columns,):
        raise ValueError("weight and bias must match the final input dimension")
    if weight.stride(0) != 1 or bias.stride(0) != 1:
        raise ValueError("weight and bias must be contiguous")
    if x.device != weight.device or x.device != bias.device:
        raise ValueError("input, weight, and bias must be on the same device")
    if x.device.type != "cuda":
        raise ValueError("layer_norm requires CUDA tensors")
    if x.dtype != weight.dtype or x.dtype != bias.dtype:
        raise ValueError("input, weight, and bias must have the same dtype")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("layer_norm supports float16 and float32 tensors")
    if columns * x.element_size() > MAX_FEATURE_BYTES:
        raise ValueError(
            f"feature row uses {columns * x.element_size()} bytes; "
            f"the fused limit is {MAX_FEATURE_BYTES}"
        )
    assert_int32_addressable(x, name="input")


@dataclass(frozen=True, slots=True)
class _BackwardBuffers:
    """Preallocated outputs and partial-reduction state for backward."""

    dx: torch.Tensor
    partial_dw: torch.Tensor
    partial_db: torch.Tensor
    dw: torch.Tensor
    db: torch.Tensor
    locks: torch.Tensor
    group_size: int
    group_count: int
    block_size: int
    num_warps: int


def _launch_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    *,
    eps: float,
    store_stats: bool,
) -> None:
    """Launch LayerNorm forward into preallocated output and statistics."""
    _validate(x, weight, bias)
    validate_epsilon(eps)
    if output.shape != x.shape or output.device != x.device or output.dtype != x.dtype:
        raise ValueError("output metadata must match the input")
    if output.stride(1) != 1:
        raise ValueError("layer_norm output requires a contiguous final dimension")
    assert_int32_addressable(output, name="output")
    rows, columns = x.shape
    if store_stats:
        expected = (rows,)
        if mean.shape != expected or rstd.shape != expected:
            raise ValueError("mean and rstd must contain one float32 value per row")
        if mean.dtype != torch.float32 or rstd.dtype != torch.float32:
            raise ValueError("mean and rstd must use float32")
        if mean.device != x.device or rstd.device != x.device:
            raise ValueError("mean and rstd must be on the input device")

    block_size, num_warps = block_size_and_warps(columns, x.element_size())
    _layer_norm_forward_kernel[(rows,)](
        x,
        weight,
        bias,
        output,
        mean,
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
    block_size, num_warps = block_size_and_warps(columns, x.element_size())
    selected_group_size = group_size_m(columns) if group_size is None else group_size
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
        partial_db=torch.empty(
            (selected_group_size, columns),
            dtype=torch.float32,
            device=x.device,
        ),
        dw=torch.empty_like(weight),
        db=torch.empty_like(weight),
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
    mean: torch.Tensor,
    rstd: torch.Tensor,
    buffers: _BackwardBuffers,
) -> None:
    """Launch per-row ``dx`` and lock-protected parameter partials."""
    rows, columns = x.shape
    _layer_norm_backward_stage1_kernel[(rows,)](
        buffers.dx,
        dy,
        buffers.partial_dw,
        buffers.partial_db,
        x,
        weight,
        mean,
        rstd,
        buffers.locks,
        x.stride(0),
        dy.stride(0),
        buffers.dx.stride(0),
        columns,
        GROUP_SIZE_M=buffers.group_size,
        BLOCK_SIZE=buffers.block_size,
        num_warps=buffers.num_warps,
    )


def _launch_backward_stage2(
    buffers: _BackwardBuffers,
    columns: int,
) -> None:
    """Launch final reduction of parameter-gradient partials."""
    _layer_norm_backward_stage2_kernel[(triton.cdiv(columns, STAGE2_BLOCK_N),)](
        buffers.partial_dw,
        buffers.partial_db,
        buffers.dw,
        buffers.db,
        buffers.group_count,
        columns,
        BLOCK_M=STAGE2_BLOCK_M,
        BLOCK_N=STAGE2_BLOCK_N,
        num_warps=4,
    )


def _launch_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch both LayerNorm backward stages and return all gradients."""
    if dy.shape != x.shape or dy.device != x.device or dy.dtype != x.dtype:
        raise ValueError("upstream gradient metadata must match the input")
    if dy.stride(1) != 1:
        dy = dy.contiguous()
    assert_int32_addressable(dy, name="upstream gradient")
    if torch.are_deterministic_algorithms_enabled():
        message = (
            "Triton LayerNorm backward uses an order-dependent lock reduction "
            "and is not deterministic"
        )
        if torch.is_deterministic_algorithms_warn_only_enabled():
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        else:
            raise RuntimeError(message)

    buffers = _make_backward_buffers(x, weight)
    _launch_backward_stage1(dy, x, weight, mean, rstd, buffers)
    _launch_backward_stage2(buffers, x.shape[1])
    return buffers.dx, buffers.dw, buffers.db


class _LayerNormFunction(torch.autograd.Function):
    """Autograd bridge for the Triton LayerNorm kernels."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Run forward and save FP32 row statistics for backward."""
        _validate(x, weight, bias)
        output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
        rstd = torch.empty_like(mean)
        _launch_forward(
            x,
            weight,
            bias,
            output,
            mean,
            rstd,
            eps=eps,
            store_stats=True,
        )
        ctx.save_for_backward(x, weight, mean, rstd)
        return output

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Compute input, weight, and bias gradients."""
        x, weight, mean, rstd = ctx.saved_tensors
        dx, dw, db = _launch_backward(dy, x, weight, mean, rstd)
        return dx, dw, db, None


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Apply LayerNorm over the final dimension with Triton autograd.

    FP16 is intended for inference-oriented throughput experiments. FP32 is
    supported for numerically strict gradient validation. Backward uses an
    order-dependent reduction and rejects PyTorch's deterministic mode.
    """
    validate_epsilon(eps)
    _validate(x, weight, bias)
    needs_grad = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (x, weight, bias)
    )
    if not needs_grad:
        output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        _launch_forward(
            x,
            weight,
            bias,
            output,
            output,
            output,
            eps=eps,
            store_stats=False,
        )
        return output
    result = _LayerNormFunction.apply(x, weight, bias, eps)  # type: ignore[no-untyped-call]
    return cast(torch.Tensor, result)


def _torch_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Return PyTorch LayerNorm with the lab's default epsilon."""
    return F.layer_norm(x, (x.shape[-1],), weight, bias, EPS)


def _make_output(args: TensorArgs) -> torch.Tensor:
    """Allocate a contiguous forward output."""
    x = args[0]
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


def _launch(args: TensorArgs, output: torch.Tensor) -> None:
    """Launch output-only forward for correctness and fallback benchmarking."""
    x, weight, bias = args
    _launch_forward(
        x,
        weight,
        bias,
        output,
        output,
        output,
        eps=EPS,
        store_stats=False,
    )


def _make_benchmark_call(args: TensorArgs, output: torch.Tensor) -> BenchmarkCall:
    """Prepare saved-statistic buffers outside the timed region."""
    x, weight, bias = args
    mean = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    rstd = torch.empty_like(mean)
    return partial(
        _launch_forward,
        x,
        weight,
        bias,
        output,
        mean,
        rstd,
        eps=EPS,
        store_stats=True,
    )


def _make_reference_call(args: TensorArgs, output: torch.Tensor) -> BenchmarkCall:
    """Prepare allocation-free native LayerNorm outputs for the baseline."""
    x, weight, bias = args
    mean = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    rstd = torch.empty_like(mean)

    def launch() -> object:
        """Call the native CUDA LayerNorm with preallocated outputs."""
        return torch.ops.aten.native_layer_norm.out(
            x,
            [x.shape[-1]],
            weight,
            bias,
            EPS,
            out0=output,
            out1=mean,
            out2=rstd,
        )

    return launch


def _make_inputs(
    columns: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TensorArgs:
    """Create a fixed-row LayerNorm benchmark problem."""
    x = torch.randn(_ROWS, columns, device=device, dtype=dtype)
    weight = torch.randn(columns, device=device, dtype=dtype)
    bias = torch.randn(columns, device=device, dtype=dtype)
    return x, weight, bias


def _make_adversarial(device: torch.device) -> TensorArgs:
    """Create row-strided FP32 input with a non-power-of-two width."""
    rows, columns = 37, 1000
    storage = torch.randn(rows * 2, columns, device=device, dtype=torch.float32)
    x = storage[::2]
    weight = torch.randn(columns, device=device, dtype=torch.float32)
    bias = torch.randn(columns, device=device, dtype=torch.float32)
    return x, weight, bias


def _bytes_moved(columns: int, dtype: torch.dtype) -> int:
    """Return effective input-read plus output-write traffic.

    Weight, bias, mean, and reciprocal-standard-deviation traffic is
    amortized across 4096 rows and intentionally excluded from the roofline
    numerator.
    """
    element_size = torch.empty((), dtype=dtype).element_size()
    return 2 * _ROWS * columns * element_size


LAYER_NORM = register(
    KernelSpec(
        name="layer_norm_forward",
        description="Fused row-wise LayerNorm with FP32 statistics and autograd backward.",
        triton_fn=layer_norm,
        launch_fn=_launch,
        ref_fn=_torch_layer_norm,
        make_inputs=_make_inputs,
        make_output=_make_output,
        make_adversarial=_make_adversarial,
        sizes=(128, 256, 512, 1024, 2048, 4096, 8192, 16384),
        correctness_sizes=(128, 1024, 4096),
        bound="memory",
        bytes_moved=_bytes_moved,
        dtypes=(torch.float16,),
        benchmark_call_factory=_make_benchmark_call,
        reference_call_factory=_make_reference_call,
        supports_interpreter=False,
    )
)
