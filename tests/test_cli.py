"""Unit tests for benchmark CLI argument handling."""

from __future__ import annotations

import pytest

from tklab.cli import _gpu_utilization_samples, parse_args


def test_parse_args_selects_multiple_kernels() -> None:
    """Preserve repeated ``--kernel`` selections."""
    options = parse_args(
        [
            "--kernel",
            "vector_add",
            "--kernel",
            "fused_softmax",
            "--warmup-ms",
            "10",
            "--repetition-ms",
            "20",
        ]
    )
    assert options.kernels == ("vector_add", "fused_softmax")
    assert options.warmup_ms == 10
    assert options.repetition_ms == 20
    assert options.allow_busy_gpu is False


def test_parse_args_allows_explicit_busy_gpu_override() -> None:
    """Expose an explicit escape hatch without weakening the default guard."""
    options = parse_args(["--allow-busy-gpu"])
    assert options.allow_busy_gpu is True


def test_parse_args_rejects_unknown_kernel() -> None:
    """Let argparse reject unknown registry keys."""
    with pytest.raises(SystemExit):
        parse_args(["--kernel", "missing"])


def test_gpu_preflight_collects_repeated_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the complete sample window rather than a single idle reading."""
    values = iter((3, 4, 5))
    sleeps: list[float] = []
    monkeypatch.setattr(
        "tklab.cli.gpu_utilization_pct",
        lambda device: next(values),
    )
    monkeypatch.setattr("tklab.cli.time.sleep", sleeps.append)
    assert _gpu_utilization_samples(0, sample_count=3, interval_seconds=0.25) == (
        3,
        4,
        5,
    )
    assert sleeps == [0.25, 0.25]
