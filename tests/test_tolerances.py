"""Unit tests for numerical assertion helpers."""

from __future__ import annotations

import pytest
import torch

from tklab.harness.tolerances import assert_close, assert_relative_frobenius


def test_assert_close_accepts_small_dtype_appropriate_error() -> None:
    """Accept expected float16 rounding error."""
    output = torch.tensor([1.0, 2.0], dtype=torch.float16)
    reference = torch.tensor([1.001, 2.001], dtype=torch.float32)
    assert_close(output, reference)


def test_assert_close_rejects_non_finite_output() -> None:
    """Reject NaN even if a permissive tolerance is supplied."""
    output = torch.tensor([float("nan")], dtype=torch.float32)
    with pytest.raises(AssertionError, match="non-finite"):
        assert_close(output, torch.zeros_like(output), rtol=1.0, atol=1.0)


def test_relative_frobenius_handles_zero_reference() -> None:
    """Use absolute norm when the reference norm is zero."""
    output = torch.zeros(4)
    reference = torch.zeros(4)
    assert_relative_frobenius(output, reference, max_relative_error=1e-3)


def test_relative_frobenius_rejects_large_error() -> None:
    """Reject aggregate error above the configured threshold."""
    with pytest.raises(AssertionError, match="Frobenius"):
        assert_relative_frobenius(
            torch.ones(4),
            torch.zeros(4),
            max_relative_error=0.5,
        )
