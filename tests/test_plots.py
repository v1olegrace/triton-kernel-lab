"""Unit tests for headless benchmark plotting."""

from __future__ import annotations

from pathlib import Path

import pytest

from tklab.harness.bench import BenchRow
from tklab.harness.plots import (
    AttentionMemoryRow,
    BenchmarkPayload,
    plot_attention_memory,
    plot_matmul_accumulation_comparison,
    plot_speedup,
    plot_throughput,
)


def _row(
    *,
    kernel: str,
    size: int,
    speedup: float,
    tflops: float | None = None,
) -> BenchRow:
    """Create a compact benchmark row for plot tests."""
    row: BenchRow = {
        "kernel": kernel,
        "dtype": "float16",
        "size": size,
        "ms": 1.0,
        "ms_lo": 0.9,
        "ms_hi": 1.1,
        "torch_ms": speedup,
        "speedup": speedup,
    }
    if tflops is not None:
        row["tflops"] = tflops
    return row


def test_plot_speedup_writes_png(tmp_path: Path) -> None:
    """Render a speedup plot without a display server."""
    payload: BenchmarkPayload = {
        "rows": [
            _row(kernel="example", size=128, speedup=1.0),
            _row(kernel="example", size=256, speedup=1.2),
        ]
    }
    output = plot_speedup(payload, tmp_path)
    assert output.is_file()
    assert output.stat().st_size > 0


def test_plot_throughput_ignores_memory_payload(tmp_path: Path) -> None:
    """Return ``None`` when TFLOP/s data is absent."""
    payload: BenchmarkPayload = {"rows": [_row(kernel="memory_kernel", size=128, speedup=1.0)]}
    assert plot_throughput(payload, tmp_path) is None


def test_matmul_comparison_requires_matching_sizes(tmp_path: Path) -> None:
    """Reject comparisons whose observations are not aligned."""
    fp32: BenchmarkPayload = {"rows": [_row(kernel="fp32", size=512, speedup=1.0, tflops=10.0)]}
    fp16: BenchmarkPayload = {"rows": [_row(kernel="fp16", size=1024, speedup=1.0, tflops=15.0)]}
    with pytest.raises(ValueError, match="matching sizes"):
        plot_matmul_accumulation_comparison(fp32, fp16, tmp_path)


def test_attention_memory_plot_handles_oom_marker(tmp_path: Path) -> None:
    """Render successful memory points plus a terminal OOM observation."""
    rows: list[AttentionMemoryRow] = [
        {
            "implementation": "flash",
            "sequence_length": 512,
            "peak_increment_bytes": 1 << 20,
            "oom": False,
        },
        {
            "implementation": "materialized",
            "sequence_length": 512,
            "peak_increment_bytes": 25 << 20,
            "oom": False,
        },
        {
            "implementation": "materialized",
            "sequence_length": 1024,
            "peak_increment_bytes": None,
            "oom": True,
        },
    ]
    output = plot_attention_memory(rows, tmp_path)
    assert output.is_file()
    assert output.stat().st_size > 0
