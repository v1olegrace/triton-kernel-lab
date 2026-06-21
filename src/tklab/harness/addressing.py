"""Shared address-width guards for Triton kernel launchers.

Several kernels index global memory with signed int32 offset arithmetic. A
tensor whose maximum relative element offset exceeds ``2**31 - 1`` would wrap
that arithmetic and silently corrupt loads or stores, so every affected
launcher rejects such layouts before reaching Triton. This module centralizes
that check so the bound and its error message stay identical across kernels.
"""

from __future__ import annotations

import torch

MAX_INT32_OFFSET = 2**31 - 1


def max_relative_offset(tensor: torch.Tensor) -> int:
    """Return the largest absolute element displacement within ``tensor``.

    Args:
        tensor: Tensor whose strided extent is measured.

    Returns:
        ``sum((dim - 1) * abs(stride))`` across every dimension. Using the
        absolute stride keeps the guard correct for layouts whose storage is
        traversed in reverse as well as ordinary positive-stride tensors.
    """
    if tensor.numel() == 0:
        return 0
    return sum(
        (dimension - 1) * abs(stride)
        for dimension, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )


def assert_int32_addressable(tensor: torch.Tensor, *, name: str) -> None:
    """Reject layouts that overflow signed int32 offset arithmetic.

    Args:
        tensor: Tensor about to be indexed with int32 offsets.
        name: Human-readable tensor name used in the error message.

    Raises:
        ValueError: If the tensor's maximum relative offset exceeds
            ``MAX_INT32_OFFSET``.
    """
    if max_relative_offset(tensor) > MAX_INT32_OFFSET:
        raise ValueError(f"{name} strides exceed the signed int32 offset range")
