"""CPU-only tests for Flash Attention memory-study calculations."""

from __future__ import annotations

import math

import pytest

from benchmarks.flash_attention_memory import (
    WorkerResult,
    _dedicated_memory_crossing,
    _empirical_slope,
    _quadratic_bytes_per_n_squared,
)


def _row(sequence_length: int, peak_bytes: int) -> WorkerResult:
    """Create one successful synthetic worker result."""
    return {
        "implementation": "flash",
        "sequence_length": sequence_length,
        "baseline_allocated_bytes": 0,
        "peak_allocated_bytes": peak_bytes,
        "peak_increment_bytes": peak_bytes,
        "output_finite": True,
        "oom": False,
        "error": None,
    }


def test_quadratic_memory_model_counts_scores_and_probabilities() -> None:
    """Count two FP16 BxHxNxN buffers for B=1 and H=16."""
    assert _quadratic_bytes_per_n_squared() == 64


def test_dedicated_memory_crossing_uses_quadratic_buffers_only() -> None:
    """Compute the analytical crossing independently of allocator overhead."""
    total_bytes = 8 * 2**30
    assert _dedicated_memory_crossing(total_bytes) == pytest.approx(math.sqrt(total_bytes / 64))
    with pytest.raises(ValueError, match="positive"):
        _dedicated_memory_crossing(0)


def test_empirical_slope_recovers_linear_and_quadratic_growth() -> None:
    """Recover expected log-log slopes from synthetic observations."""
    assert _empirical_slope([_row(512, 1), _row(1024, 2)]) == pytest.approx(1.0)
    assert _empirical_slope([_row(512, 1), _row(1024, 4)]) == pytest.approx(2.0)
