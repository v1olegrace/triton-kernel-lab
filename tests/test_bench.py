"""CPU-only tests for benchmark metric semantics."""

from __future__ import annotations

import pytest
import torch

from tklab.harness.bench import BenchRow, _add_performance_metrics
from tklab.kernels.matmul import MATMUL_FP32ACC


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
