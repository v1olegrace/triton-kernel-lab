"""Real-GPU edge-case tests for strided vector addition."""

from __future__ import annotations

import pytest
import torch

from tklab.kernels.vector_add import vector_add

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required"),
]


def test_vector_add_accepts_broadcast_inputs_without_aliasing_output() -> None:
    """Read zero-stride inputs but always return distinct writable elements."""
    left = torch.tensor([2.0], device="cuda").expand(1009)
    right = torch.arange(1009, device="cuda", dtype=torch.float32)
    output = vector_add(left, right)
    torch.testing.assert_close(output, left + right)
    assert output.is_contiguous()
    assert output.untyped_storage().nbytes() >= output.numel() * output.element_size()
