"""Numerically stable, row-wise fused softmax."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from tklab.harness.addressing import assert_int32_addressable
from tklab.registry import KernelSpec, TensorArgs, register

_ROWS = 4096
_MAX_FUSED_COLUMNS = 1 << 16


@triton.jit  # type: ignore[untyped-decorator]
def _softmax_kernel(
    x_ptr: tl.tensor,
    output_ptr: tl.tensor,
    x_row_stride: int,
    output_row_stride: int,
    n_cols: int,
    block_size: tl.constexpr,
) -> None:
    """Compute one softmax row entirely in on-chip storage."""
    row = tl.program_id(axis=0)
    columns = tl.arange(0, block_size)
    mask = columns < n_cols

    values = tl.load(
        x_ptr + row * x_row_stride + columns,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    shifted = values - tl.max(values, axis=0)
    numerator = tl.exp(shifted)
    output = numerator / tl.sum(numerator, axis=0)

    tl.store(
        output_ptr + row * output_row_stride + columns,
        output,
        mask=mask,
    )


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute softmax over the contiguous final dimension of a 2D tensor.

    Args:
        x: Float16 or float32 tensor shaped ``(rows, columns)``. Row stride may
            be arbitrary, but the final dimension must be contiguous.

    Returns:
        Softmax probabilities with matching shape, dtype, and device.

    Raises:
        ValueError: If the input shape, dtype, layout, or row width is
            unsupported.
    """
    _validate(x)
    output = _make_output((x,))
    _launch((x,), output)
    return output


def _validate(x: torch.Tensor) -> None:
    """Validate the single-pass softmax contract."""
    if x.ndim != 2:
        raise ValueError("softmax expects a 2D tensor shaped (rows, columns)")
    rows, columns = x.shape
    if rows == 0 or columns == 0:
        raise ValueError("softmax does not support empty dimensions")
    if columns > _MAX_FUSED_COLUMNS:
        raise ValueError(
            f"n_cols={columns} exceeds the {_MAX_FUSED_COLUMNS}-column "
            "single-pass limit; use an online softmax"
        )
    if x.stride(1) != 1:
        raise ValueError("softmax requires a contiguous final dimension")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("softmax supports float16 and float32 tensors")
    interpreter_enabled = os.environ.get("TRITON_INTERPRET") == "1"
    if x.device.type != "cuda" and not (interpreter_enabled and x.device.type == "cpu"):
        raise ValueError("softmax requires CUDA unless TRITON_INTERPRET=1")
    assert_int32_addressable(x, name="input")


def _num_warps_for(block_size: int) -> int:
    """Select a warp count for a power-of-two row block."""
    if block_size >= 8192:
        return 16
    if block_size >= 2048:
        return 8
    return 4


def _make_output(args: TensorArgs) -> torch.Tensor:
    """Allocate a softmax output with the input's suggested memory format."""
    return torch.empty_like(args[0])


def _launch(args: TensorArgs, output: torch.Tensor) -> None:
    """Launch fused softmax into a preallocated output."""
    (x,) = args
    _validate(x)
    if output.shape != x.shape or output.device != x.device or output.dtype != x.dtype:
        raise ValueError("output metadata must match the input")
    if output.stride(1) != 1:
        raise ValueError("softmax output requires a contiguous final dimension")
    assert_int32_addressable(output, name="output")

    rows, columns = x.shape
    block_size = triton.next_power_of_2(columns)
    _softmax_kernel[(rows,)](
        x,
        output,
        x.stride(0),
        output.stride(0),
        columns,
        block_size=block_size,
        num_warps=_num_warps_for(block_size),
    )


def _torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """Return PyTorch's fused row-wise softmax reference."""
    return torch.softmax(x, dim=-1)


def _launch_reference(args: TensorArgs, output: torch.Tensor) -> None:
    """Launch allocation-free PyTorch softmax."""
    (x,) = args
    torch.softmax(x, dim=-1, out=output)


def _make_inputs(
    n_cols: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TensorArgs:
    """Create a fixed-row-count softmax benchmark input."""
    return (torch.randn(_ROWS, n_cols, device=device, dtype=dtype),)


def _bytes_moved(n_cols: int, dtype: torch.dtype) -> int:
    """Return one global-memory read plus one write."""
    element_size = torch.empty((), dtype=dtype).element_size()
    return 2 * _ROWS * n_cols * element_size


def _make_adversarial(device: torch.device) -> TensorArgs:
    """Create a row-strided, non-power-of-two softmax input."""
    x = torch.randn(64, 1000, device=device, dtype=torch.float32)
    return (x[::2],)


def _naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute a pedagogical multi-pass softmax baseline."""
    shifted = x - x.max(dim=-1, keepdim=True).values
    numerator = shifted.exp()
    return numerator / numerator.sum(dim=-1, keepdim=True)


SOFTMAX = register(
    KernelSpec(
        name="fused_softmax",
        description="Single-pass row-wise softmax with FP32 reductions.",
        triton_fn=softmax,
        launch_fn=_launch,
        ref_fn=_torch_softmax,
        reference_launch_fn=_launch_reference,
        make_inputs=_make_inputs,
        make_output=_make_output,
        make_adversarial=_make_adversarial,
        sizes=(128, 256, 512, 1024, 2048, 4096, 8192, 16384),
        bound="memory",
        bytes_moved=_bytes_moved,
        dtypes=(torch.float16, torch.float32),
        naive_fn=_naive_softmax,
    )
)
