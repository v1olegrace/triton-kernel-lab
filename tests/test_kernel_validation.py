"""CPU-only validation tests for public kernel wrappers."""

from __future__ import annotations

import pytest
import torch

from tklab.kernels.fused_softmax import softmax
from tklab.kernels.matmul import matmul_fp32acc
from tklab.kernels.vector_add import vector_add


def test_vector_add_rejects_shape_mismatch() -> None:
    """Reject incompatible vectors before invoking Triton."""
    with pytest.raises(ValueError, match="shape mismatch"):
        vector_add(torch.zeros(2), torch.zeros(3))


def test_softmax_rejects_non_contiguous_final_dimension() -> None:
    """Reject column-strided layouts explicitly."""
    x = torch.zeros(4, 8)[:, ::2]
    with pytest.raises(ValueError, match="contiguous final dimension"):
        softmax(x)


def test_matmul_rejects_incompatible_inner_dimensions() -> None:
    """Reject invalid GEMM shapes before invoking Triton."""
    with pytest.raises(ValueError, match="incompatible shapes"):
        matmul_fp32acc(
            torch.zeros(2, 3, dtype=torch.float16),
            torch.zeros(4, 2, dtype=torch.float16),
        )
