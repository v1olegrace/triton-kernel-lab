"""Fused residual addition and RMSNorm with two-output autograd."""

from __future__ import annotations

import math
from functools import partial
from typing import Any, cast

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tklab.kernels.rms_norm import (
    _EPS,
    _MAX_INT32_OFFSET,
    _block_size_and_warps,
    _launch_backward,
    _validate,
)
from tklab.registry import (
    BenchmarkCall,
    BenchmarkScalar,
    KernelSpec,
    TensorArgs,
    register,
)

_ROWS = 4096


@triton.jit  # type: ignore[untyped-decorator]
def _residual_rms_norm_forward_kernel(
    x_ptr: tl.tensor,
    residual_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    residual_sum_ptr: tl.tensor,
    output_ptr: tl.tensor,
    rstd_ptr: tl.tensor,
    x_row_stride: int,
    residual_row_stride: int,
    residual_sum_row_stride: int,
    output_row_stride: int,
    n_cols: int,
    eps: float,
    INPUT_FP16: tl.constexpr,
    STORE_RESIDUAL: tl.constexpr,
    STORE_STATS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Add one residual row, normalize it, and optionally save both auxiliaries."""
    row = tl.program_id(axis=0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < n_cols

    values = tl.load(
        x_ptr + row * x_row_stride + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + row * residual_row_stride + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    residual_sum = values + residual
    if INPUT_FP16:
        residual_sum = residual_sum.to(tl.float16).to(tl.float32)
    mean_square = tl.sum(residual_sum * residual_sum, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(mean_square + eps)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    output = residual_sum * rstd * weight

    if STORE_RESIDUAL:
        tl.store(
            residual_sum_ptr + row * residual_sum_row_stride + columns,
            residual_sum,
            mask=mask,
        )
    tl.store(
        output_ptr + row * output_row_stride + columns,
        output,
        mask=mask,
    )
    if STORE_STATS:
        tl.store(rstd_ptr + row, rstd)


def _validate_inputs(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    """Validate the fused residual RMSNorm tensor contract."""
    if residual.shape != x.shape:
        raise ValueError("residual must match the input shape")
    if residual.device != x.device:
        raise ValueError("input and residual must be on the same device")
    if residual.dtype != x.dtype:
        raise ValueError("input and residual must have the same dtype")
    if residual.ndim != 2:
        raise ValueError("residual_rms_norm expects 2D input tensors")
    if residual.stride(1) != 1:
        raise ValueError("residual requires a contiguous final dimension")
    _validate(x, weight)
    rows, columns = residual.shape
    max_relative_offset = (rows - 1) * residual.stride(0) + columns - 1
    if max_relative_offset > _MAX_INT32_OFFSET:
        raise ValueError("residual strides exceed the kernel's signed int32 offset range")


def _validate_output(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    name: str,
) -> None:
    """Validate one preallocated forward output."""
    if (
        output.shape != reference.shape
        or output.device != reference.device
        or output.dtype != reference.dtype
    ):
        raise ValueError(f"{name} metadata must match the input")
    if output.stride(1) != 1:
        raise ValueError(f"{name} requires a contiguous final dimension")


def _launch_forward(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    residual_sum: torch.Tensor,
    output: torch.Tensor,
    rstd: torch.Tensor,
    *,
    eps: float,
    store_residual: bool,
    store_stats: bool,
) -> None:
    """Launch fused residual addition and RMSNorm into supplied buffers."""
    _validate_inputs(x, residual, weight)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive")
    _validate_output(output, x, name="output")
    if store_residual:
        _validate_output(residual_sum, x, name="residual_sum")
    rows, columns = x.shape
    if store_stats:
        if rstd.shape != (rows,) or rstd.device != x.device:
            raise ValueError("rstd must contain one value per input row")
        if rstd.dtype != torch.float32:
            raise ValueError("rstd must use float32")

    block_size, num_warps = _block_size_and_warps(columns, x.element_size())
    _residual_rms_norm_forward_kernel[(rows,)](
        x,
        residual,
        weight,
        residual_sum,
        output,
        rstd,
        x.stride(0),
        residual.stride(0),
        residual_sum.stride(0),
        output.stride(0),
        columns,
        eps,
        INPUT_FP16=x.dtype == torch.float16,
        STORE_RESIDUAL=store_residual,
        STORE_STATS=store_stats,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


class _ResidualRMSNormFunction(torch.autograd.Function):
    """Autograd bridge for fused residual addition and RMSNorm."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the residual sum followed by its normalized representation."""
        _validate_inputs(x, residual, weight)
        ctx.set_materialize_grads(False)
        residual_sum = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        output = torch.empty_like(residual_sum)
        rstd = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
        _launch_forward(
            x,
            residual,
            weight,
            residual_sum,
            output,
            rstd,
            eps=eps,
            store_residual=True,
            store_stats=True,
        )
        ctx.save_for_backward(residual_sum, weight, rstd)
        return residual_sum, output

    @staticmethod
    def backward(
        ctx: Any,
        incoming_residual_sum: torch.Tensor | None,
        incoming_output: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Accumulate both output paths into equal input and residual gradients."""
        residual_sum, weight, rstd = ctx.saved_tensors
        if incoming_output is None:
            total = (
                torch.zeros_like(residual_sum)
                if incoming_residual_sum is None
                else incoming_residual_sum
            )
            dweight = torch.zeros_like(weight)
        else:
            total, dweight = _launch_backward(
                incoming_output,
                residual_sum,
                weight,
                rstd,
                incoming_dx=incoming_residual_sum,
            )
        return total, total, dweight, None


def residual_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = _EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x + residual, RMSNorm(x + residual))``.

    The two input gradients are identical because both inputs enter through
    the same residual sum. Backward reuses RMSNorm's audited lock reduction
    and adds the direct residual-sum gradient inside stage 1.
    """
    result = _ResidualRMSNormFunction.apply(  # type: ignore[no-untyped-call]
        x,
        residual,
        weight,
        eps,
    )
    return cast(tuple[torch.Tensor, torch.Tensor], result)


def _registered_residual_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Return only the normalized output for the single-output registry."""
    _, output = residual_rms_norm(x, residual, weight)
    return output


def _torch_residual_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Return the native composed reference's normalized output."""
    residual_sum = x + residual
    return F.rms_norm(residual_sum, (x.shape[-1],), weight, _EPS)


def _make_inputs(
    columns: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TensorArgs:
    """Create a fixed-row residual RMSNorm benchmark problem."""
    x = torch.randn(_ROWS, columns, device=device, dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(columns, device=device, dtype=dtype)
    return x, residual, weight


def _make_adversarial(device: torch.device) -> TensorArgs:
    """Create independently row-strided FP32 inputs with a masked tail."""
    rows, columns = 37, 1000
    x_storage = torch.randn(rows * 2, columns, device=device, dtype=torch.float32)
    residual_storage = torch.randn(
        rows * 3,
        columns,
        device=device,
        dtype=torch.float32,
    )
    x = x_storage[::2]
    residual = residual_storage[::3]
    weight = torch.randn(columns, device=device, dtype=torch.float32)
    return x, residual, weight


def _make_output(args: TensorArgs) -> torch.Tensor:
    """Allocate the normalized registry output."""
    x = args[0]
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


def _launch(args: TensorArgs, output: torch.Tensor) -> None:
    """Launch normalized output only for correctness fallback."""
    x, residual, weight = args
    _launch_forward(
        x,
        residual,
        weight,
        output,
        output,
        output,
        eps=_EPS,
        store_residual=False,
        store_stats=False,
    )


def _make_benchmark_call(args: TensorArgs, output: torch.Tensor) -> BenchmarkCall:
    """Prepare residual-sum and statistic buffers outside the timed region."""
    x, residual, weight = args
    residual_sum = torch.empty_like(output)
    rstd = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    return partial(
        _launch_forward,
        x,
        residual,
        weight,
        residual_sum,
        output,
        rstd,
        eps=_EPS,
        store_residual=True,
        store_stats=True,
    )


def _make_reference_call(args: TensorArgs, output: torch.Tensor) -> BenchmarkCall:
    """Prepare the native add plus ``F.rms_norm`` baseline."""
    x, residual, weight = args

    def launch() -> object:
        """Run the semantically equivalent native PyTorch composition."""
        residual_sum = x + residual
        return residual_sum, F.rms_norm(
            residual_sum,
            (x.shape[-1],),
            weight,
            _EPS,
        )

    return launch


def _make_naive_call(args: TensorArgs, output: torch.Tensor) -> BenchmarkCall:
    """Prepare a multi-output decomposition from ordinary PyTorch operations."""
    x, residual, weight = args

    def launch() -> object:
        """Run separate add, reduction, reciprocal-root, and scaling ops."""
        residual_sum = x + residual
        residual_sum_float = residual_sum.float()
        mean_square = torch.mean(
            residual_sum_float * residual_sum_float,
            dim=-1,
            keepdim=True,
        )
        normalized = residual_sum_float * torch.rsqrt(mean_square + _EPS)
        output = (normalized * weight.float()).to(x.dtype)
        return residual_sum, output

    return launch


def _bytes_moved(columns: int, dtype: torch.dtype) -> int:
    """Return effective traffic for two reads and two materialized outputs."""
    element_size = torch.empty((), dtype=dtype).element_size()
    return 4 * _ROWS * columns * element_size


def _benchmark_metadata(
    columns: int,
    dtype: torch.dtype,
) -> dict[str, BenchmarkScalar]:
    """Describe the native and deliberately decomposed baselines."""
    del columns, dtype
    return {
        "reference_baseline": "torch.add plus torch.nn.functional.rms_norm",
        "naive_baseline": "manual add, pointwise, and reduction composition",
        "reference_allocates_output": True,
    }


RESIDUAL_RMS_NORM = register(
    KernelSpec(
        name="residual_rms_norm_forward",
        description="Fused residual addition and RMSNorm with two-output autograd.",
        triton_fn=_registered_residual_rms_norm,
        launch_fn=_launch,
        ref_fn=_torch_residual_rms_norm,
        make_inputs=_make_inputs,
        make_output=_make_output,
        make_adversarial=_make_adversarial,
        sizes=(128, 256, 512, 1024, 2048, 4096, 8192, 16384),
        correctness_sizes=(128, 1024, 4096),
        bound="memory",
        bytes_moved=_bytes_moved,
        dtypes=(torch.float16,),
        benchmark_metadata=_benchmark_metadata,
        benchmark_call_factory=_make_benchmark_call,
        reference_call_factory=_make_reference_call,
        naive_call_factory=_make_naive_call,
        supports_interpreter=False,
    )
)
