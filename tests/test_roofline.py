"""CPU-only tests for roofline formulas and profiles."""

from __future__ import annotations

from pathlib import Path

import pytest

from tklab.harness.jsonio import JsonObject
from tklab.harness.roofline import (
    PEAKS_SCHEMA_VERSION,
    RTX4060_PROFILE,
    TflopsSample,
    _maximum_observed_sm_clock_mhz,
    _validate_stored_peaks,
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


def _clock_sample(*, clock_mhz: int) -> TflopsSample:
    """Create a minimal typed calibration sample."""
    return {
        "n": 1024,
        "ms": 1.0,
        "tflops": 1.0,
        "sm_clock_mhz": clock_mhz,
        "sm_clock_min_mhz": clock_mhz,
        "sm_clock_max_mhz": clock_mhz,
        "sm_clock_samples_mhz": [clock_mhz],
    }


def test_theoretical_ceiling_uses_maximum_clock_across_sweeps() -> None:
    """Keep sparse winner-clock sampling from lowering a theoretical upper bound."""
    assert (
        _maximum_observed_sm_clock_mhz(
            [_clock_sample(clock_mhz=2505)],
            [_clock_sample(clock_mhz=2715)],
        )
        == 2715
    )


def _cached_peaks() -> JsonObject:
    """Return the minimum valid cache payload used by provenance tests."""
    return {
        "schema_version": PEAKS_SCHEMA_VERSION,
        "gpu_name": "NVIDIA GeForce RTX 4060",
        "compute_capability": "8.9",
        "torch_version": "2.12.1+cu130",
        "triton_version": "3.7.1",
        "calibration_started_at_utc": "2026-06-21T18:00:00+00:00",
        "calibration_preflight_gpu_utilization_pct": [3, 3, 3, 3, 3],
        "calibration_preflight_gpu_utilization_limit_pct": 10,
        "peak_bw_gbps": 247.8,
        "cublas_tflops_fp16_fp32acc": 30.9,
        "theoretical_tflops_fp16_fp32acc_at_measured_clock": 32.0,
        "theoretical_tflops_fp16_fp16acc_at_measured_clock": 64.0,
        "theoretical_ceiling_sm_clock_mhz": 2625,
        "theoretical_provenance": {
            "model": "NVIDIA GeForce RTX 4060",
            "rated_boost_clock_mhz": 2460,
            "streaming_multiprocessors": 24,
            "tensor_cores_per_sm": 4,
            "fp16_fp32acc_flops_per_clock_per_tensor_core": 128,
            "fp16_fp32acc_rated_tflops": 30.22848,
            "fp16_fp16acc_rated_tflops": 60.45696,
            "ceiling_clock_policy": "maximum observed clock",
            "product_spec_url": "https://example.com/product",
            "architecture_whitepaper_url": "https://example.com/architecture",
        },
    }


def test_cached_peaks_accept_matching_environment() -> None:
    """Reuse calibration only when every provenance discriminator matches."""
    _validate_stored_peaks(
        _cached_peaks(),
        path=Path("peaks.json"),
        expected_gpu_name="NVIDIA GeForce RTX 4060",
        expected_compute_capability="8.9",
        expected_torch_version="2.12.1+cu130",
        expected_triton_version="3.7.1",
    )


def test_cached_peaks_reject_stale_software_stack() -> None:
    """Prevent benchmark rows from mixing peaks across torch/Triton upgrades."""
    with pytest.raises(ValueError, match=r"active environment.*torch_version"):
        _validate_stored_peaks(
            _cached_peaks(),
            path=Path("peaks.json"),
            expected_gpu_name="NVIDIA GeForce RTX 4060",
            expected_compute_capability="8.9",
            expected_torch_version="2.13.0+cu131",
            expected_triton_version="3.7.1",
        )


def test_cached_peaks_reject_non_positive_measurement() -> None:
    """Reject a corrupt cache before it reaches roofline divisions."""
    payload = _cached_peaks()
    payload["peak_bw_gbps"] = 0.0
    with pytest.raises(ValueError, match="invalid positive numeric fields"):
        _validate_stored_peaks(
            payload,
            path=Path("peaks.json"),
            expected_gpu_name="NVIDIA GeForce RTX 4060",
            expected_compute_capability="8.9",
            expected_torch_version="2.12.1+cu130",
            expected_triton_version="3.7.1",
        )
