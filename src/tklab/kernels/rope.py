"""Rotate-half rotary positional embedding with autograd."""

from __future__ import annotations

from functools import partial
from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from tklab.harness.addressing import assert_int32_addressable
from tklab.registry import (
    BenchmarkCall,
    BenchmarkScalar,
    KernelSpec,
    TensorArgs,
    register,
)

_ROWS = 16_384
_BLOCK_SIZE = 256


@triton.jit  # type: ignore[untyped-decorator]
def _rope_forward_kernel(
    x_ptr: tl.tensor,
    cos_ptr: tl.tensor,
    sin_ptr: tl.tensor,
    output_ptr: tl.tensor,
    x_row_stride: int,
    cos_row_stride: int,
    sin_row_stride: int,
    output_row_stride: int,
    half_dim: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Rotate one masked block using shared half-dimension angle tables."""
    row = tl.program_id(axis=0)
    block = tl.program_id(axis=1)
    half_columns = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = half_columns < half_dim
    x_first = tl.load(
        x_ptr + row * x_row_stride + half_columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x_second = tl.load(
        x_ptr + row * x_row_stride + half_dim + half_columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    cosine = tl.load(
        cos_ptr + row * cos_row_stride + half_columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sine = tl.load(
        sin_ptr + row * sin_row_stride + half_columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output_ptr + row * output_row_stride + half_columns,
        x_first * cosine - x_second * sine,
        mask=mask,
    )
    tl.store(
        output_ptr + row * output_row_stride + half_dim + half_columns,
        x_second * cosine + x_first * sine,
        mask=mask,
    )


@triton.jit  # type: ignore[untyped-decorator]
def _rope_backward_kernel(
    dy_ptr: tl.tensor,
    x_ptr: tl.tensor,
    cos_ptr: tl.tensor,
    sin_ptr: tl.tensor,
    dx_ptr: tl.tensor,
    dcos_ptr: tl.tensor,
    dsin_ptr: tl.tensor,
    dy_row_stride: int,
    x_row_stride: int,
    cos_row_stride: int,
    sin_row_stride: int,
    dx_row_stride: int,
    dcos_row_stride: int,
    dsin_row_stride: int,
    half_dim: int,
    COMPUTE_DCOS: tl.constexpr,
    COMPUTE_DSIN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Apply the inverse rotation and differentiate both angle tables."""
    row = tl.program_id(axis=0)
    block = tl.program_id(axis=1)
    half_columns = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = half_columns < half_dim
    dy_first = tl.load(
        dy_ptr + row * dy_row_stride + half_columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    dy_second = tl.load(
        dy_ptr + row * dy_row_stride + half_dim + half_columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    cosine = tl.load(
        cos_ptr + row * cos_row_stride + half_columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sine = tl.load(
        sin_ptr + row * sin_row_stride + half_columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # Transpose of the forward rotation: same cosine, negated sine.
    dx_first = dy_first * cosine + dy_second * sine
    dx_second = dy_second * cosine - dy_first * sine
    tl.store(
        dx_ptr + row * dx_row_stride + half_columns,
        dx_first,
        mask=mask,
    )
    tl.store(
        dx_ptr + row * dx_row_stride + half_dim + half_columns,
        dx_second,
        mask=mask,
    )
    if COMPUTE_DCOS or COMPUTE_DSIN:
        x_first = tl.load(
            x_ptr + row * x_row_stride + half_columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x_second = tl.load(
            x_ptr + row * x_row_stride + half_dim + half_columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if COMPUTE_DCOS:
            dcos = dy_first * x_first + dy_second * x_second
            tl.store(
                dcos_ptr + row * dcos_row_stride + half_columns,
                dcos,
                mask=mask,
            )
        if COMPUTE_DSIN:
            dsin = -dy_first * x_second + dy_second * x_first
            tl.store(
                dsin_ptr + row * dsin_row_stride + half_columns,
                dsin,
                mask=mask,
            )


def _validate_inputs(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> None:
    """Validate rotate-half shapes, layouts, dtypes, and address widths."""
    if x.ndim != 2 or cos.ndim != 2 or sin.ndim != 2:
        raise ValueError("rope expects 2D input and angle-table tensors")
    rows, columns = x.shape
    if rows == 0 or columns == 0:
        raise ValueError("rope does not support empty dimensions")
    if columns % 2 != 0:
        raise ValueError("rope requires an even feature dimension")
    expected_table_shape = (rows, columns // 2)
    if cos.shape != expected_table_shape or sin.shape != expected_table_shape:
        raise ValueError("cos and sin must have shape (rows, columns // 2)")
    if x.device != cos.device or x.device != sin.device:
        raise ValueError("x, cos, and sin must be on the same device")
    if x.dtype != cos.dtype or x.dtype != sin.dtype:
        raise ValueError("x, cos, and sin must have the same dtype")
    if x.device.type != "cuda":
        raise ValueError("rope requires CUDA tensors")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("rope supports float16 and float32 tensors")
    if x.stride(1) != 1 or cos.stride(1) != 1 or sin.stride(1) != 1:
        raise ValueError("rope requires contiguous final dimensions")
    for name, tensor in (("x", x), ("cos", cos), ("sin", sin)):
        assert_int32_addressable(tensor, name=name)


def _validate_output(output: torch.Tensor, x: torch.Tensor, *, name: str) -> None:
    """Validate a preallocated full-width RoPE tensor."""
    if output.shape != x.shape or output.device != x.device or output.dtype != x.dtype:
        raise ValueError(f"{name} metadata must match x")
    if output.stride(1) != 1:
        raise ValueError(f"{name} requires a contiguous final dimension")
    assert_int32_addressable(output, name=name)


def _launch_forward(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Launch rotate-half RoPE into a supplied output."""
    _validate_inputs(x, cos, sin)
    _validate_output(output, x, name="output")
    rows, columns = x.shape
    half_dim = columns // 2
    _rope_forward_kernel[(rows, triton.cdiv(half_dim, _BLOCK_SIZE))](
        x,
        cos,
        sin,
        output,
        x.stride(0),
        cos.stride(0),
        sin.stride(0),
        output.stride(0),
        half_dim,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )


def _launch_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    need_dcos: bool = True,
    need_dsin: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Launch deterministic RoPE backward."""
    _validate_inputs(x, cos, sin)
    if dy.shape != x.shape or dy.device != x.device or dy.dtype != x.dtype:
        raise ValueError("upstream gradient metadata must match x")
    if dy.stride(1) != 1:
        dy = dy.contiguous()
    assert_int32_addressable(dy, name="upstream gradient")
    rows, columns = x.shape
    dx = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    dcos = torch.empty(cos.shape, dtype=cos.dtype, device=cos.device) if need_dcos else None
    dsin = torch.empty(sin.shape, dtype=sin.dtype, device=sin.device) if need_dsin else None
    dcos_buffer = dx if dcos is None else dcos
    dsin_buffer = dx if dsin is None else dsin
    half_dim = columns // 2
    _rope_backward_kernel[(rows, triton.cdiv(half_dim, _BLOCK_SIZE))](
        dy,
        x,
        cos,
        sin,
        dx,
        dcos_buffer,
        dsin_buffer,
        dy.stride(0),
        x.stride(0),
        cos.stride(0),
        sin.stride(0),
        dx.stride(0),
        dcos_buffer.stride(0),
        dsin_buffer.stride(0),
        half_dim,
        COMPUTE_DCOS=need_dcos,
        COMPUTE_DSIN=need_dsin,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )
    return dx, dcos, dsin


class _RoPEFunction(torch.autograd.Function):
    """Autograd bridge for rotate-half RoPE."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward and save inputs for the inverse rotation."""
        _validate_inputs(x, cos, sin)
        output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        _launch_forward(x, cos, sin, output)
        ctx.save_for_backward(x, cos, sin)
        return output

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return gradients for x and both half-dimension angle tables."""
        x, cos, sin = ctx.saved_tensors
        return _launch_backward(
            dy,
            x,
            cos,
            sin,
            need_dcos=ctx.needs_input_grad[1],
            need_dsin=ctx.needs_input_grad[2],
        )


def rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotate-half RoPE using ``(rows, feature_dim // 2)`` tables."""
    result = _RoPEFunction.apply(x, cos, sin)  # type: ignore[no-untyped-call]
    return cast(torch.Tensor, result)


def _torch_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Return the explicit rotate-half PyTorch reference."""
    x_first, x_second = x.chunk(2, dim=-1)
    return torch.cat(
        (
            x_first * cos - x_second * sin,
            x_second * cos + x_first * sin,
        ),
        dim=-1,
    )


def _make_inputs(
    columns: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TensorArgs:
    """Create a fixed-row RoPE benchmark problem with valid angle tables."""
    x = torch.randn(_ROWS, columns, device=device, dtype=dtype)
    angles = torch.randn(_ROWS, columns // 2, device=device, dtype=torch.float32)
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    return x, cos, sin


def _make_adversarial(device: torch.device) -> TensorArgs:
    """Create row-strided FP32 input and tables with a masked half-width tail."""
    rows, columns = 37, 1000
    x_storage = torch.randn(rows * 2, columns, device=device, dtype=torch.float32)
    angles_storage = torch.randn(
        rows * 3,
        columns // 2,
        device=device,
        dtype=torch.float32,
    )
    angles = angles_storage[::3]
    return x_storage[::2], torch.cos(angles), torch.sin(angles)


def _make_output(args: TensorArgs) -> torch.Tensor:
    """Allocate a contiguous RoPE output."""
    x = args[0]
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


def _launch(args: TensorArgs, output: torch.Tensor) -> None:
    """Launch allocation-free RoPE forward for the benchmark harness."""
    x, cos, sin = args
    _launch_forward(x, cos, sin, output)


def _make_benchmark_call(args: TensorArgs, output: torch.Tensor) -> BenchmarkCall:
    """Prepare the allocation-free Triton benchmark call."""
    x, cos, sin = args
    return partial(_launch_forward, x, cos, sin, output)


def _benchmark_metadata(
    columns: int,
    dtype: torch.dtype,
) -> dict[str, BenchmarkScalar]:
    """Describe the rotate-half reference convention."""
    del columns, dtype
    return {
        "reference_baseline": "PyTorch rotate-half composition with torch.cat",
        "naive_baseline": "not applicable",
        "reference_allocates_output": True,
    }


def _bytes_moved(columns: int, dtype: torch.dtype) -> int:
    """Return full-width input, angle-table, and output traffic."""
    element_size = torch.empty((), dtype=dtype).element_size()
    return 3 * _ROWS * columns * element_size


ROPE = register(
    KernelSpec(
        name="rope_forward",
        description="Rotate-half rotary embedding with half-dimension angle tables.",
        triton_fn=rope,
        launch_fn=_launch,
        ref_fn=_torch_rope,
        make_inputs=_make_inputs,
        make_output=_make_output,
        make_adversarial=_make_adversarial,
        sizes=(32, 64, 128, 256, 512, 1024, 2048),
        correctness_sizes=(64, 128, 256),
        bound="memory",
        bytes_moved=_bytes_moved,
        dtypes=(torch.float16,),
        benchmark_metadata=_benchmark_metadata,
        benchmark_call_factory=_make_benchmark_call,
        supports_interpreter=False,
    )
)
