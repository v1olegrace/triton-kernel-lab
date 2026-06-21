"""Strided vector addition implemented with one-dimensional Triton programs."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from tklab.harness.addressing import assert_int32_addressable
from tklab.registry import KernelSpec, TensorArgs, register

_BLOCK_SIZE = 1024


@triton.jit  # type: ignore[untyped-decorator]
def _vector_add_kernel(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    output_ptr: tl.tensor,
    n_elements: int,
    x_stride: int,
    y_stride: int,
    output_stride: int,
    block_size: tl.constexpr,
) -> None:
    """Add one block of elements with a masked tail."""
    block_start = tl.program_id(axis=0) * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets * x_stride, mask=mask)
    y = tl.load(y_ptr + offsets * y_stride, mask=mask)
    tl.store(output_ptr + offsets * output_stride, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the elementwise sum of two one-dimensional CUDA tensors.

    Args:
        x: First input vector.
        y: Second input vector with matching shape, dtype, and device.

    Returns:
        Contiguous tensor with the same shape, dtype, and device as ``x``.

    Raises:
        ValueError: If inputs are empty, incompatible, or not 1D.
    """
    _validate_inputs(x, y)
    output = _make_output((x, y))
    _launch((x, y), output)
    return output


def _validate_inputs(x: torch.Tensor, y: torch.Tensor) -> None:
    """Validate vector-add inputs before launching Triton."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("vector_add expects one-dimensional tensors")
    if x.numel() == 0:
        raise ValueError("vector_add does not support empty tensors")
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} != {y.shape}")
    if x.device != y.device:
        raise ValueError(f"device mismatch: {x.device} != {y.device}")
    if x.dtype != y.dtype:
        raise ValueError(f"dtype mismatch: {x.dtype} != {y.dtype}")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("vector_add supports float16 and float32 tensors")
    interpreter_enabled = os.environ.get("TRITON_INTERPRET") == "1"
    if x.device.type != "cuda" and not (interpreter_enabled and x.device.type == "cpu"):
        raise ValueError("vector_add requires CUDA unless TRITON_INTERPRET=1")
    assert_int32_addressable(x, name="left input")
    assert_int32_addressable(y, name="right input")


def _make_output(args: TensorArgs) -> torch.Tensor:
    """Allocate a contiguous output.

    Inputs may be expanded with a zero stride. Preserving such a stride for a
    writable output would alias every logical element and create racing stores.
    """
    x = args[0]
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


def _launch(args: TensorArgs, output: torch.Tensor) -> None:
    """Launch vector addition into a preallocated output tensor."""
    x, y = args
    _validate_inputs(x, y)
    if output.shape != x.shape or output.device != x.device or output.dtype != x.dtype:
        raise ValueError("output metadata must match the inputs")
    if not output.is_contiguous():
        raise ValueError("vector_add output must be contiguous")
    assert_int32_addressable(output, name="output")

    grid = (triton.cdiv(x.numel(), _BLOCK_SIZE),)
    _vector_add_kernel[grid](
        x,
        y,
        output,
        x.numel(),
        x.stride(0),
        y.stride(0),
        output.stride(0),
        _BLOCK_SIZE,
        num_warps=4,
    )


def _launch_reference(args: TensorArgs, output: torch.Tensor) -> None:
    """Launch allocation-free PyTorch vector addition."""
    x, y = args
    torch.add(x, y, out=output)


def _make_inputs(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TensorArgs:
    """Create random vector-add benchmark inputs."""
    return (
        torch.randn(size, device=device, dtype=dtype),
        torch.randn(size, device=device, dtype=dtype),
    )


def _make_adversarial(device: torch.device) -> TensorArgs:
    """Create non-contiguous inputs with a non-block-aligned length."""
    size = 1009
    left = torch.randn(size * 2, device=device, dtype=torch.float32)[::2]
    right = torch.randn(size * 3, device=device, dtype=torch.float32)[::3]
    return left, right


def _bytes_moved(size: int, dtype: torch.dtype) -> int:
    """Return two reads plus one write for ``size`` elements."""
    return 3 * size * torch.empty((), dtype=dtype).element_size()


VECTOR_ADD = register(
    KernelSpec(
        name="vector_add",
        description="Strided elementwise vector addition with masked tails.",
        triton_fn=vector_add,
        launch_fn=_launch,
        ref_fn=torch.add,
        reference_launch_fn=_launch_reference,
        make_inputs=_make_inputs,
        make_output=_make_output,
        sizes=tuple(2**exponent for exponent in range(12, 28)),
        bound="memory",
        bytes_moved=_bytes_moved,
        dtypes=(torch.float16, torch.float32),
        make_adversarial=_make_adversarial,
    )
)
