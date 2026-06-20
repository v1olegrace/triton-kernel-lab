"""Unit tests for benchmark CLI argument handling."""

from __future__ import annotations

import pytest

from tklab.cli import parse_args


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


def test_parse_args_rejects_unknown_kernel() -> None:
    """Let argparse reject unknown registry keys."""
    with pytest.raises(SystemExit):
        parse_args(["--kernel", "missing"])
