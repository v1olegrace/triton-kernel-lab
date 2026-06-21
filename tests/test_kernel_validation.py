"""CPU-only validation tests for public kernel wrappers."""

from __future__ import annotations

import pytest
import torch

from tklab.kernels.flash_attention import attention_noncausal
from tklab.kernels.fused_softmax import softmax
from tklab.kernels.layer_norm import layer_norm
from tklab.kernels.matmul import matmul_fp32acc
from tklab.kernels.residual_rms_norm import residual_rms_norm
from tklab.kernels.rms_norm import rms_norm
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


def test_layer_norm_rejects_mismatched_parameters() -> None:
    """Reject affine parameters that do not match the feature dimension."""
    with pytest.raises(ValueError, match="match the final input dimension"):
        layer_norm(
            torch.zeros(2, 8, dtype=torch.float32),
            torch.zeros(7, dtype=torch.float32),
            torch.zeros(8, dtype=torch.float32),
        )


def test_layer_norm_rejects_strided_parameters() -> None:
    """Reject affine vectors whose unit-stride assumption does not hold."""
    weight = torch.zeros(16, dtype=torch.float32)[::2]
    bias = torch.zeros(16, dtype=torch.float32)[::2]
    with pytest.raises(ValueError, match="must be contiguous"):
        layer_norm(torch.zeros(2, 8), weight, bias)


def test_layer_norm_requires_cuda() -> None:
    """Fail clearly instead of reaching a Triton launch with CPU tensors."""
    with pytest.raises(ValueError, match="requires CUDA"):
        layer_norm(
            torch.zeros(2, 8),
            torch.ones(8),
            torch.zeros(8),
        )


def test_rms_norm_rejects_mismatched_weight() -> None:
    """Reject a weight vector that does not match the feature dimension."""
    with pytest.raises(ValueError, match="match the final input dimension"):
        rms_norm(
            torch.zeros(2, 8),
            torch.ones(7),
        )


def test_residual_rms_norm_rejects_shape_mismatch() -> None:
    """Reject residual metadata before attempting a CUDA launch."""
    with pytest.raises(ValueError, match="residual must match"):
        residual_rms_norm(
            torch.zeros(2, 8),
            torch.zeros(3, 8),
            torch.ones(8),
        )


def test_matmul_rejects_offsets_outside_int32_range() -> None:
    """Reject layouts that overflow the contiguous fast-path address width."""
    left = torch.empty_strided(
        (2, 2),
        (2**31, 1),
        dtype=torch.float16,
        device="meta",
    )
    right = torch.empty((2, 2), dtype=torch.float16, device="meta")
    with pytest.raises(ValueError, match="int32 offset"):
        matmul_fp32acc(left, right)


def test_attention_rejects_shape_mismatch() -> None:
    """Reject mismatched Q/K/V metadata before requiring CUDA."""
    query = torch.zeros(1, 2, 8, 64, dtype=torch.float16)
    key = torch.zeros(1, 2, 7, 64, dtype=torch.float16)
    value = torch.zeros_like(query)
    with pytest.raises(ValueError, match="identical shapes"):
        attention_noncausal(query, key, value)
