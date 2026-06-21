"""Fused elementwise activations with deterministic autograd.

This module ships the pointwise activations that round out the lab's elementwise
coverage alongside :mod:`tklab.kernels.vector_add` and the gated
:mod:`tklab.kernels.swiglu`. Each activation is a single masked forward kernel
plus a deterministic, elementwise backward kernel. Nonlinear intermediates are
computed in FP32 regardless of the storage dtype, and a compile-time activation
selector keeps one auditable kernel per direction instead of duplicating the
launcher for every function.

Implemented activations:

- ``relu``: ``max(x, 0)``.
- ``gelu``: the exact error-function form ``0.5 x (1 + erf(x / sqrt(2)))``,
  matching ``torch.nn.functional.gelu`` with its default ``approximate="none"``.
- ``silu``: ``x * sigmoid(x)`` (also known as swish).
- ``tanh``: the hyperbolic tangent, evaluated as ``2 * sigmoid(2x) - 1`` because
  this Triton build does not expose ``tl.math.tanh``.
"""

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
from tklab.registry import BenchmarkCall, KernelSpec, TensorArgs, TensorFn, register

_ROWS = 4096
_BLOCK_SIZE = 256

# Compile-time activation selectors. They key the Triton cache, so each value
# specializes its own forward and backward kernel. They are wrapped as
# ``tl.constexpr`` so the @jit device helpers may read them as module globals.
_RELU = tl.constexpr(0)
_GELU = tl.constexpr(1)
_SILU = tl.constexpr(2)
_TANH = tl.constexpr(3)

# Wrapped as ``tl.constexpr`` so the @jit GELU helpers may read them as globals.
_INV_SQRT2 = tl.constexpr(0.7071067811865476)  # 1 / sqrt(2)
_INV_SQRT_2PI = tl.constexpr(0.3989422804014327)  # 1 / sqrt(2 * pi)


@triton.jit  # type: ignore[untyped-decorator]
def _activation_value(x: tl.tensor, ACTIVATION: tl.constexpr) -> tl.tensor:
    """Return one activation evaluated in FP32."""
    if ACTIVATION == _RELU:
        return tl.maximum(x, 0.0)
    if ACTIVATION == _GELU:
        return 0.5 * x * (1.0 + tl.math.erf(x * _INV_SQRT2))
    if ACTIVATION == _SILU:
        return x * stable_sigmoid(x)
    # _TANH, expressed through the shared overflow-safe sigmoid.
    return 2.0 * stable_sigmoid(2.0 * x) - 1.0


@triton.jit  # type: ignore[untyped-decorator]
def _activation_grad(x: tl.tensor, ACTIVATION: tl.constexpr) -> tl.tensor:
    """Return the activation's derivative with respect to ``x`` in FP32."""
    if ACTIVATION == _RELU:
        return tl.where(x > 0.0, 1.0, 0.0)
    if ACTIVATION == _GELU:
        cdf = 0.5 * (1.0 + tl.math.erf(x * _INV_SQRT2))
        pdf = _INV_SQRT_2PI * tl.exp(-0.5 * x * x)
        return cdf + x * pdf
    if ACTIVATION == _SILU:
        sigmoid = stable_sigmoid(x)
        return sigmoid * (1.0 + x * (1.0 - sigmoid))
    # _TANH derivative: 1 - tanh(x)^2.
    tanh = 2.0 * stable_sigmoid(2.0 * x) - 1.0
    return 1.0 - tanh * tanh


@triton.jit  # type: ignore[untyped-decorator]
def _activation_forward_kernel(
    x_ptr: tl.tensor,
    output_ptr: tl.tensor,
    x_row_stride: int,
    output_row_stride: int,
    n_cols: int,
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Apply one masked activation block."""
    row = tl.program_id(axis=0)
    block = tl.program_id(axis=1)
    columns = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = columns < n_cols
    x = tl.load(x_ptr + row * x_row_stride + columns, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output_ptr + row * output_row_stride + columns,
        _activation_value(x, ACTIVATION),
        mask=mask,
    )


@triton.jit  # type: ignore[untyped-decorator]
def _activation_backward_kernel(
    dy_ptr: tl.tensor,
    x_ptr: tl.tensor,
    dx_ptr: tl.tensor,
    dy_row_stride: int,
    x_row_stride: int,
    dx_row_stride: int,
    n_cols: int,
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Differentiate one masked activation block."""
    row = tl.program_id(axis=0)
    block = tl.program_id(axis=1)
    columns = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = columns < n_cols
    dy = tl.load(dy_ptr + row * dy_row_stride + columns, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * x_row_stride + columns, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        dx_ptr + row * dx_row_stride + columns,
        dy * _activation_grad(x, ACTIVATION),
        mask=mask,
    )


def _validate_input(x: torch.Tensor) -> None:
    """Validate the shared elementwise-activation tensor contract."""
    if x.ndim != 2:
        raise ValueError("activation expects a 2D tensor shaped (rows, columns)")
    rows, columns = x.shape
    if rows == 0 or columns == 0:
        raise ValueError("activation does not support empty dimensions")
    if x.device.type != "cuda":
        raise ValueError("activation requires CUDA tensors")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("activation supports float16 and float32 tensors")
    if x.stride(1) != 1:
        raise ValueError("activation requires a contiguous final dimension")
    assert_int32_addressable(x, name="input")


def _validate_output(output: torch.Tensor, reference: torch.Tensor, *, name: str) -> None:
    """Validate one preallocated activation output or gradient."""
    if (
        output.shape != reference.shape
        or output.device != reference.device
        or output.dtype != reference.dtype
    ):
        raise ValueError(f"{name} metadata must match the input")
    if output.stride(1) != 1:
        raise ValueError(f"{name} requires a contiguous final dimension")
    assert_int32_addressable(output, name=name)


def _launch_forward(x: torch.Tensor, output: torch.Tensor, *, activation: int) -> None:
    """Launch one activation forward into a supplied output."""
    _validate_input(x)
    _validate_output(output, x, name="output")
    rows, columns = x.shape
    _activation_forward_kernel[(rows, triton.cdiv(columns, _BLOCK_SIZE))](
        x,
        output,
        x.stride(0),
        output.stride(0),
        columns,
        ACTIVATION=activation,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )


def _launch_backward(dy: torch.Tensor, x: torch.Tensor, *, activation: int) -> torch.Tensor:
    """Launch deterministic activation backward and return ``dx``."""
    _validate_input(x)
    if dy.shape != x.shape or dy.device != x.device or dy.dtype != x.dtype:
        raise ValueError("upstream gradient metadata must match the input")
    if dy.stride(1) != 1:
        dy = dy.contiguous()
    assert_int32_addressable(dy, name="upstream gradient")
    rows, columns = x.shape
    dx = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _activation_backward_kernel[(rows, triton.cdiv(columns, _BLOCK_SIZE))](
        dy,
        x,
        dx,
        dy.stride(0),
        x.stride(0),
        dx.stride(0),
        columns,
        ACTIVATION=activation,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )
    return dx


class _ActivationFunction(torch.autograd.Function):
    """Autograd bridge shared by every elementwise activation."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, activation: int) -> torch.Tensor:
        """Run forward and save the input for the deterministic backward."""
        _validate_input(x)
        output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        _launch_forward(x, output, activation=activation)
        ctx.save_for_backward(x)
        ctx.activation = activation
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, dy: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Return the input gradient; the activation selector has no gradient."""
        (x,) = ctx.saved_tensors
        return _launch_backward(dy, x, activation=ctx.activation), None


def _apply(x: torch.Tensor, activation: int) -> torch.Tensor:
    """Apply one activation through the shared autograd function."""
    result = _ActivationFunction.apply(x, activation)  # type: ignore[no-untyped-call]
    return cast(torch.Tensor, result)


def relu(x: torch.Tensor) -> torch.Tensor:
    """Return ``max(x, 0)`` with deterministic autograd."""
    return _apply(x, _RELU)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Return the exact error-function GELU with deterministic autograd."""
    return _apply(x, _GELU)


def silu(x: torch.Tensor) -> torch.Tensor:
    """Return ``x * sigmoid(x)`` (swish) with deterministic autograd."""
    return _apply(x, _SILU)


def tanh(x: torch.Tensor) -> torch.Tensor:
    """Return ``tanh(x)`` with deterministic autograd."""
    return _apply(x, _TANH)


def _make_inputs(
    columns: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TensorArgs:
    """Create a fixed-row activation benchmark problem."""
    return (torch.randn(_ROWS, columns, device=device, dtype=dtype),)


def _make_adversarial(device: torch.device) -> TensorArgs:
    """Create row-strided FP32 input with a non-power-of-two masked tail."""
    rows, columns = 37, 1000
    storage = torch.randn(rows * 2, columns, device=device, dtype=torch.float32)
    return (storage[::2],)


def _make_output(args: TensorArgs) -> torch.Tensor:
    """Allocate a contiguous activation output."""
    x = args[0]
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


def _bytes_moved(columns: int, dtype: torch.dtype) -> int:
    """Return one read plus one write."""
    element_size = torch.empty((), dtype=dtype).element_size()
    return 2 * _ROWS * columns * element_size


def _register_activation(
    *,
    name: str,
    activation: int,
    triton_fn: TensorFn,
    ref_fn: TensorFn,
    description: str,
) -> KernelSpec:
    """Register one activation specification with the shared harness wiring."""

    def launch(args: TensorArgs, output: torch.Tensor) -> None:
        """Launch allocation-free forward for the benchmark harness."""
        _launch_forward(args[0], output, activation=activation)

    def benchmark_call(args: TensorArgs, output: torch.Tensor) -> BenchmarkCall:
        """Prepare the allocation-free Triton benchmark call."""
        return partial(_launch_forward, args[0], output, activation=activation)

    return register(
        KernelSpec(
            name=name,
            description=description,
            triton_fn=triton_fn,
            launch_fn=launch,
            ref_fn=ref_fn,
            make_inputs=_make_inputs,
            make_output=_make_output,
            make_adversarial=_make_adversarial,
            sizes=(128, 256, 512, 1024, 2048, 4096, 8192, 16384),
            correctness_sizes=(128, 1024, 4096),
            bound="memory",
            bytes_moved=_bytes_moved,
            dtypes=(torch.float16,),
            benchmark_call_factory=benchmark_call,
            supports_interpreter=False,
        )
    )


RELU = _register_activation(
    name="relu_forward",
    activation=_RELU,
    triton_fn=relu,
    ref_fn=torch.relu,
    description="Elementwise ReLU with deterministic autograd.",
)

GELU = _register_activation(
    name="gelu_forward",
    activation=_GELU,
    triton_fn=gelu,
    ref_fn=F.gelu,
    description="Elementwise exact (erf) GELU with deterministic autograd.",
)

SILU = _register_activation(
    name="silu_forward",
    activation=_SILU,
    triton_fn=silu,
    ref_fn=F.silu,
    description="Elementwise SiLU/swish with deterministic autograd.",
)

TANH = _register_activation(
    name="tanh_forward",
    activation=_TANH,
    triton_fn=tanh,
    ref_fn=torch.tanh,
    description="Elementwise tanh with deterministic autograd.",
)
