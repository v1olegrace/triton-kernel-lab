"""CPU-only tests for benchmark metric semantics."""

from __future__ import annotations

import pytest
import torch

import tklab.kernels  # noqa: F401
from tklab.harness.bench import BenchRow, _add_performance_metrics
from tklab.kernels.flash_attention import ATTENTION_CAUSAL, ATTENTION_NONCAUSAL
from tklab.kernels.matmul import MATMUL_FP32ACC
from tklab.kernels.residual_rms_norm import RESIDUAL_RMS_NORM
from tklab.kernels.rms_norm import RMS_NORM
from tklab.registry import REGISTRY


def test_compute_metrics_separate_same_size_cublas_from_peak() -> None:
    """Do not conflate same-shape cuBLAS efficiency with peak utilization."""
    row: BenchRow = {
        "kernel": "matmul_fp32acc",
        "dtype": "float16",
        "size": 512,
        "ms": 0.04,
        "ms_lo": 0.04,
        "ms_hi": 0.04,
        "torch_ms": 0.02,
        "speedup": 0.5,
    }

    _add_performance_metrics(
        row,
        spec=MATMUL_FP32ACC,
        size=512,
        dtype=torch.float16,
        milliseconds=0.04,
        peak_bw_gbps=None,
        cublas_tflops=32.0,
        theoretical_tflops=64.0,
    )

    assert row["reference_tflops"] == pytest.approx(2 * row["tflops"])
    assert row["pct_cublas_same_size"] == pytest.approx(50.0)
    assert row["pct_cublas_peak"] == pytest.approx(100.0 * row["tflops"] / 32.0)


def test_attention_flops_use_full_and_lower_triangular_pair_counts() -> None:
    """Count both attention matmuls and the exact causal lower triangle."""
    assert ATTENTION_NONCAUSAL.flops is not None
    assert ATTENTION_CAUSAL.flops is not None
    sequence_length = 8
    full_pairs = sequence_length**2
    causal_pairs = sequence_length * (sequence_length + 1) // 2
    fixed_factor = 4 * 1 * 16 * 64

    assert ATTENTION_NONCAUSAL.flops(sequence_length, torch.float16) == (fixed_factor * full_pairs)
    assert ATTENTION_CAUSAL.flops(sequence_length, torch.float16) == (fixed_factor * causal_pairs)


def test_norm_specs_expose_native_and_naive_baseline_metadata() -> None:
    """Keep future norm benchmark JSONs explicit about baseline semantics."""
    for spec in (RMS_NORM, RESIDUAL_RMS_NORM):
        assert spec.naive_fn is not None or spec.naive_call_factory is not None
        assert spec.benchmark_metadata is not None
        metadata = spec.benchmark_metadata(1024, torch.float16)
        assert metadata["reference_allocates_output"] is True
        assert isinstance(metadata["reference_baseline"], str)
        assert isinstance(metadata["naive_baseline"], str)


def test_memory_specs_expose_two_labeled_baselines() -> None:
    """Require both native/reference and decomposed comparisons in final JSONs."""
    for spec in REGISTRY.values():
        if spec.bound != "memory":
            continue
        assert spec.naive_fn is not None or spec.naive_call_factory is not None
        assert spec.benchmark_metadata is not None
        metadata = spec.benchmark_metadata(spec.sizes[0], spec.dtypes[0])
        assert isinstance(metadata["reference_baseline"], str)
        assert isinstance(metadata["naive_baseline"], str)
        assert metadata["reference_baseline"]
        assert metadata["naive_baseline"]
        assert isinstance(metadata["reference_allocates_output"], bool)
        assert isinstance(metadata["naive_allocates_output"], bool)
