"""Headless benchmark visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from tklab.harness.bench import BenchRow


class BenchmarkPayload(TypedDict):
    """Minimal benchmark payload required by plot functions."""

    rows: list[BenchRow]


class AttentionMemoryRow(TypedDict):
    """One peak-memory observation for an attention implementation."""

    implementation: str
    sequence_length: int
    peak_increment_bytes: int | None
    oom: bool


def plot_speedup(payload: BenchmarkPayload, output_dir: Path) -> Path:
    """Plot Triton speedup against PyTorch and an optional naive baseline.

    Args:
        payload: Benchmark payload containing at least one row.
        output_dir: Directory where the PNG is written.

    Returns:
        Path to the generated PNG.

    Raises:
        ValueError: If the payload has no rows.
    """
    rows = payload["rows"]
    if not rows:
        raise ValueError("cannot plot an empty benchmark")

    output_dir.mkdir(parents=True, exist_ok=True)
    kernel = rows[0]["kernel"]
    output_path = output_dir / f"{kernel}_speedup.png"
    dtypes = sorted({row["dtype"] for row in rows})
    has_naive = all("speedup_vs_naive" in row for row in rows)

    if has_naive:
        figure, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        vendor_axis, naive_axis = axes
    else:
        figure, vendor_axis = plt.subplots(figsize=(8, 5))
        naive_axis = None

    for dtype in dtypes:
        dtype_rows = sorted(
            (row for row in rows if row["dtype"] == dtype),
            key=lambda row: row["size"],
        )
        sizes = [row["size"] for row in dtype_rows]
        vendor_axis.plot(
            sizes,
            [row["speedup"] for row in dtype_rows],
            marker="o",
            label=dtype,
        )
        if naive_axis is not None:
            naive_axis.plot(
                sizes,
                [row["speedup_vs_naive"] for row in dtype_rows],
                marker="s",
                label=dtype,
            )

    _configure_speedup_axis(vendor_axis, ylabel="vs torch")
    vendor_axis.set_title(f"{kernel}: Triton speedup")
    bottom_axis = vendor_axis
    if naive_axis is not None:
        _configure_speedup_axis(naive_axis, ylabel="vs naive")
        bottom_axis = naive_axis

    bottom_axis.set_xlabel("Size")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_throughput(payload: BenchmarkPayload, output_dir: Path) -> Path | None:
    """Plot TFLOP/s for a compute-bound benchmark.

    Args:
        payload: Benchmark payload.
        output_dir: Directory where the PNG is written.

    Returns:
        Generated PNG path, or ``None`` for a non-compute payload.
    """
    rows = payload["rows"]
    if not rows or "tflops" not in rows[0]:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    kernel = rows[0]["kernel"]
    output_path = output_dir / f"{kernel}_tflops.png"
    ordered = sorted(rows, key=lambda row: row["size"])

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        [row["size"] for row in ordered],
        [row["tflops"] for row in ordered],
        marker="o",
        label="Triton",
    )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Problem size")
    axis.set_ylabel("TFLOP/s")
    axis.set_title(f"{kernel}: throughput")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_matmul_accumulation_comparison(
    fp32_payload: BenchmarkPayload,
    fp16_payload: BenchmarkPayload,
    output_dir: Path,
) -> Path:
    """Compare FP32- and FP16-accumulating matmul variants.

    Args:
        fp32_payload: FP32-accumulation benchmark.
        fp16_payload: FP16-accumulation benchmark.
        output_dir: Directory where the PNG is written.

    Returns:
        Path to the comparison PNG.

    Raises:
        ValueError: If either payload is empty, lacks TFLOP/s, or uses
            different matrix sizes.
    """
    fp32_rows = sorted(fp32_payload["rows"], key=lambda row: row["size"])
    fp16_rows = sorted(fp16_payload["rows"], key=lambda row: row["size"])
    if not fp32_rows or not fp16_rows:
        raise ValueError("matmul comparison requires non-empty payloads")
    if not all("tflops" in row for row in (*fp32_rows, *fp16_rows)):
        raise ValueError("matmul comparison requires TFLOP/s observations")

    sizes = [row["size"] for row in fp32_rows]
    if sizes != [row["size"] for row in fp16_rows]:
        raise ValueError("matmul comparison requires matching sizes")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "matmul_accumulation_comparison.png"
    fp32_tflops = [row["tflops"] for row in fp32_rows]
    fp16_tflops = [row["tflops"] for row in fp16_rows]
    gaps = [
        fp16_value / fp32_value
        for fp16_value, fp32_value in zip(fp16_tflops, fp32_tflops, strict=True)
    ]

    figure, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    throughput_axis, gap_axis = axes
    throughput_axis.plot(sizes, fp32_tflops, marker="o", label="FP32 accumulate")
    throughput_axis.plot(sizes, fp16_tflops, marker="s", label="FP16 accumulate")
    throughput_axis.set_xscale("log", base=2)
    throughput_axis.set_ylabel("TFLOP/s")
    throughput_axis.set_title("Matmul accumulation-mode comparison")
    throughput_axis.grid(True, which="both", alpha=0.3)
    throughput_axis.legend()

    gap_axis.plot(sizes, gaps, marker="o", color="tab:purple")
    gap_axis.axhline(2.0, color="black", linestyle="--", linewidth=1, label="2x ideal")
    gap_axis.set_xscale("log", base=2)
    gap_axis.set_xlabel("Matrix size (M=N=K)")
    gap_axis.set_ylabel("FP16-acc / FP32-acc")
    gap_axis.grid(True, which="both", alpha=0.3)
    gap_axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_attention_memory(
    rows: list[AttentionMemoryRow],
    output_dir: Path,
) -> Path:
    """Plot measured attention peak allocation against sequence length."""
    if not rows:
        raise ValueError("attention memory plot requires observations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "flash_attention_memory.png"

    figure, axis = plt.subplots(figsize=(8, 5))
    for implementation in sorted({row["implementation"] for row in rows}):
        implementation_rows = sorted(
            (
                row
                for row in rows
                if row["implementation"] == implementation
                and row["peak_increment_bytes"] is not None
            ),
            key=lambda row: row["sequence_length"],
        )
        axis.plot(
            [row["sequence_length"] for row in implementation_rows],
            [cast(int, row["peak_increment_bytes"]) / 2**20 for row in implementation_rows],
            marker="o",
            label=implementation,
        )
        oom_rows = [row for row in rows if row["implementation"] == implementation and row["oom"]]
        if oom_rows and implementation_rows:
            marker_y = cast(int, implementation_rows[-1]["peak_increment_bytes"]) / 2**20
            for row in oom_rows:
                axis.scatter(
                    row["sequence_length"],
                    marker_y,
                    marker="x",
                    s=80,
                    color="red",
                    zorder=5,
                )
                axis.annotate(
                    "OOM",
                    (row["sequence_length"], marker_y),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    color="red",
                )

    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Peak incremental allocation (MiB)")
    axis.set_title("Attention forward peak memory")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def _configure_speedup_axis(axis: Axes, *, ylabel: str) -> None:
    """Apply shared formatting to a speedup axis."""
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xscale("log", base=2)
    axis.set_ylabel(ylabel)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
