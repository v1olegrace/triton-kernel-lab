"""Fused SwiGLU forward and deterministic elementwise backward."""

from __future__ import annotations

from functools import partial
from typing import Any, cast

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from tklab.harness.addressing import assert_int32_addressable
from tklab.kernels._elementwise_math import stable_sigmoid
from tklab.registry import (
    BenchmarkCall,
    BenchmarkScalar,
    KernelSpec,
    TensorArgs,
    register,
)

_ROWS = 4096
_BLOCK_SIZE = 256


@triton.jit  # type: ignore[untyped-decorator]
def _swiglu_forward_kernel(
    value_ptr: tl.tensor,
    gate_ptr: tl.tensor,
    output_ptr: tl.tensor,
    value_row_stride: int,
    gate_row_stride: int,
    output_row_stride: int,
    n_cols: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Compute one masked block of ``value * silu(gate)``."""
    row = tl.program_id(axis=0)
    block = tl.program_id(axis=1)
    columns = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = columns < n_cols
    value = tl.load(
        value_ptr + row * value_row_stride + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr + row * gate_row_stride + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sigmoid = stable_sigmoid(gate)
    tl.store(
        output_ptr + row * output_row_stride + columns,
        value * gate * sigmoid,
        mask=mask,
    )


@triton.jit  # type: ignore[untyped-decorator]
def _swiglu_backward_kernel(
    dy_ptr: tl.tensor,
    value_ptr: tl.tensor,
    gate_ptr: tl.tensor,
    dvalue_ptr: tl.tensor,
    dgate_ptr: tl.tensor,
    dy_row_stride: int,
    value_row_stride: int,
    gate_row_stride: int,
    dvalue_row_stride: int,
    dgate_row_stride: int,
    n_cols: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Differentiate one masked SwiGLU block."""
    row = tl.program_id(axis=0)
    block = tl.program_id(axis=1)
    columns = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = columns < n_cols
    dy = tl.load(
        dy_ptr + row * dy_row_stride + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    value = tl.load(
        value_ptr + row * value_row_stride + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr + row * gate_row_stride + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sigmoid = stable_sigmoid(gate)
    silu = gate * sigmoid
    silu_derivative = sigmoid * (1.0 + gate * (1.0 - sigmoid))
    tl.store(
        dvalue_ptr + row * dvalue_row_stride + columns,
        dy * silu,
        mask=mask,
    )
    tl.store(
        dgate_ptr + row * dgate_row_stride + columns,
        dy * value * silu_derivative,
        mask=mask,
    )


def _validate_inputs(value: torch.Tensor, gate: torch.Tensor) -> None:
    """Validate SwiGLU's shape, layout, dtype, and address-width contract."""
    if value.ndim != 2 or gate.ndim != 2:
        raise ValueError("swiglu expects 2D tensors shaped (rows, columns)")
    rows, columns = value.shape
    if rows == 0 or columns == 0:
        raise ValueError("swiglu does not support empty dimensions")
    if gate.shape != value.shape:
        raise ValueError("value and gate must have identical shapes")
    if gate.device != value.device:
        raise ValueError("value and gate must be on the same device")
    if gate.dtype != value.dtype:
        raise ValueError("value and gate must have the same dtype")
    if value.device.type != "cuda":
        raise ValueError("swiglu requires CUDA tensors")
    if value.dtype not in (torch.float16, torch.float32):
        raise ValueError("swiglu supports float16 and float32 tensors")
    if value.stride(1) != 1 or gate.stride(1) != 1:
        raise ValueError("swiglu requires contiguous final dimensions")
    for name, tensor in (("value", value), ("gate", gate)):
        assert_int32_addressable(tensor, name=name)


def _validate_output(output: torch.Tensor, reference: torch.Tensor, *, name: str) -> None:
    """Validate one preallocated SwiGLU output or gradient."""
    if (
        output.shape != reference.shape
        or output.device != reference.device
        or output.dtype != reference.dtype
    ):
        raise ValueError(f"{name} metadata must match the input")
    if output.stride(1) != 1:
        raise ValueError(f"{name} requires a contiguous final dimension")
    assert_int32_addressable(output, name=name)


def _launch_forward(
    value: torch.Tensor,
    gate: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Launch SwiGLU forward into a supplied output."""
    _validate_inputs(value, gate)
    _validate_output(output, value, name="output")
    rows, columns = value.shape
    _swiglu_forward_kernel[(rows, triton.cdiv(columns, _BLOCK_SIZE))](
        value,
        gate,
        output,
        value.stride(0),
        gate.stride(0),
        output.stride(0),
        columns,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )


def _launch_backward(
    dy: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch deterministic SwiGLU backward."""
    _validate_inputs(value, gate)
    if dy.shape != value.shape or dy.device != value.device or dy.dtype != value.dtype:
        raise ValueError("upstream gradient metadata must match the inputs")
    if dy.stride(1) != 1:
        dy = dy.contiguous()
    assert_int32_addressable(dy, name="upstream gradient")
    rows, columns = value.shape
    dvalue = torch.empty(value.shape, dtype=value.dtype, device=value.device)
    dgate = torch.empty_like(dvalue)
    _swiglu_backward_kernel[(rows, triton.cdiv(columns, _BLOCK_SIZE))](
        dy,
        value,
        gate,
        dvalue,
        dgate,
        dy.stride(0),
        value.stride(0),
        gate.stride(0),
        dvalue.stride(0),
        dgate.stride(0),
        columns,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )
    return dvalue, dgate


class _SwiGLUFunction(torch.autograd.Function):
    """Autograd bridge for fused SwiGLU."""

    @staticmethod
    def forward(
        ctx: Any,
        value: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward and save both inputs for deterministic backward."""
        _validate_inputs(value, gate)
        output = torch.empty(value.shape, dtype=value.dtype, device=value.device)
        _launch_forward(value, gate, output)
        ctx.save_for_backward(value, gate)
        return output

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return gradients for value and gate."""
        value, gate = ctx.saved_tensors
        return _launch_backward(dy, value, gate)


def swiglu(value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Return ``value * SiLU(gate)`` with FP32 nonlinear intermediates."""
    result = _SwiGLUFunction.apply(value, gate)  # type: ignore[no-untyped-call]
    return cast(torch.Tensor, result)


def _torch_swiglu(value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Return the native PyTorch SwiGLU composition."""
    return value * F.silu(gate)


def _naive_swiglu(value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compose SwiGLU with the same stable FP32 sigmoid policy."""
    gate_float = gate.float()
    exp_negative_abs = torch.exp(-torch.abs(gate_float))
    sigmoid = torch.where(
        gate_float >= 0,
        1.0 / (1.0 + exp_negative_abs),
        exp_negative_abs / (1.0 + exp_negative_abs),
    )
    return (value.float() * gate_float * sigmoid).to(value.dtype)


def _make_inputs(
    columns: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TensorArgs:
    """Create a fixed-row SwiGLU benchmark problem."""
    value = torch.randn(_ROWS, columns, device=device, dtype=dtype)
    gate = torch.randn_like(value)
    return value, gate


def _make_adversarial(device: torch.device) -> TensorArgs:
    """Create independently row-strided FP32 inputs and a masked tail."""
    rows, columns = 37, 1000
    value_storage = torch.randn(rows * 2, columns, device=device, dtype=torch.float32)
    gate_storage = torch.randn(rows * 3, columns, device=device, dtype=torch.float32)
    return value_storage[::2], gate_storage[::3]


def _make_output(args: TensorArgs) -> torch.Tensor:
    """Allocate a contiguous SwiGLU output."""
    value = args[0]
    return torch.empty(value.shape, dtype=value.dtype, device=value.device)


def _launch(args: TensorArgs, output: torch.Tensor) -> None:
    """Launch allocation-free SwiGLU forward for the benchmark harness."""
    value, gate = args
    _launch_forward(value, gate, output)


def _make_benchmark_call(args: TensorArgs, output: torch.Tensor) -> BenchmarkCall:
    """Prepare the allocation-free Triton benchmark call."""
    value, gate = args
    return partial(_launch_forward, value, gate, output)


def _benchmark_metadata(
    columns: int,
    dtype: torch.dtype,
) -> dict[str, BenchmarkScalar]:
    """Describe SwiGLU's native and stable-decomposition baselines."""
    del columns, dtype
    return {
        "reference_baseline": "value * torch.nn.functional.silu(gate)",
        "naive_baseline": "stable FP32 sigmoid and explicit pointwise composition",
        "reference_allocates_output": True,
    }


def _bytes_moved(columns: int, dtype: torch.dtype) -> int:
    """Return two reads plus one write."""
    element_size = torch.empty((), dtype=dtype).element_size()
    return 3 * _ROWS * columns * element_size


SWIGLU = register(
    KernelSpec(
        name="swiglu_forward",
        description="Fused value-times-SiLU gate with stable FP32 nonlinear math.",
        triton_fn=swiglu,
        launch_fn=_launch,
        ref_fn=_torch_swiglu,
        make_inputs=_make_inputs,
        make_output=_make_output,
        make_adversarial=_make_adversarial,
        sizes=(128, 256, 512, 1024, 2048, 4096, 8192, 16384),
        correctness_sizes=(128, 1024, 4096),
        bound="memory",
        bytes_moved=_bytes_moved,
        dtypes=(torch.float16,),
        naive_fn=_naive_swiglu,
        benchmark_metadata=_benchmark_metadata,
        benchmark_call_factory=_make_benchmark_call,
        supports_interpreter=False,
    )
)
