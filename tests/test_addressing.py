"""Unit tests for shared Triton address-width guards."""

from __future__ import annotations

import pytest
import torch

from tklab.harness.addressing import assert_int32_addressable, max_relative_offset


def test_max_relative_offset_handles_dense_and_broadcast_layouts() -> None:
    """Measure the actual logical extent without treating zero stride as unsafe."""
    dense = torch.empty_strided((3, 5), (5, 1), device="meta")
    broadcast = torch.empty_strided((3, 5), (0, 1), device="meta")
    assert max_relative_offset(dense) == 14
    assert max_relative_offset(broadcast) == 4


def test_max_relative_offset_handles_empty_tensor() -> None:
    """An empty layout performs no pointer arithmetic regardless of its strides."""
    empty = torch.empty_strided((0, 2), (2**31, 1), device="meta")
    assert max_relative_offset(empty) == 0


def test_int32_guard_rejects_oversized_strided_extent() -> None:
    """Reject a relative displacement that cannot fit signed int32 arithmetic."""
    tensor = torch.empty_strided((2, 2), (2**31, 1), device="meta")
    with pytest.raises(ValueError, match="int32 offset"):
        assert_int32_addressable(tensor, name="test tensor")
