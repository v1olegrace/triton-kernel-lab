"""CPU-only tests for roofline formulas and profiles."""

from __future__ import annotations

import pytest

from tklab.harness.roofline import (
    RTX4060_PROFILE,
    theoretical_tflops_fp16_fp16acc,
    theoretical_tflops_fp16_fp32acc,
)


def test_rtx4060_rated_fp32_accumulate_ceiling() -> None:
    """Reproduce the audited 30.22848 TFLOP/s rated ceiling."""
    assert theoretical_tflops_fp16_fp32acc(RTX4060_PROFILE.rated_boost_mhz) == pytest.approx(
        30.22848
    )


def test_fp16_accumulate_ceiling_is_double() -> None:
    """Reflect Ada's two-to-one accumulator-throughput ratio."""
    clock = 2625
    fp32_accumulate = theoretical_tflops_fp16_fp32acc(clock)
    assert theoretical_tflops_fp16_fp16acc(clock) == pytest.approx(2.0 * fp32_accumulate)
