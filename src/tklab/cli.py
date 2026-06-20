"""Command-line interface for reproducible local benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

import tklab.kernels  # noqa: F401
from tklab.harness.bench import BenchRow, bench_spec
from tklab.harness.jsonio import JsonObject, write_json_atomic
from tklab.harness.plots import (
    BenchmarkPayload,
    plot_matmul_accumulation_comparison,
    plot_speedup,
    plot_throughput,
)
from tklab.harness.roofline import PeakResults, gpu_utilization_pct, load_or_measure_peaks
from tklab.registry import REGISTRY, KernelSpec

RESULT_SCHEMA_VERSION = 4
MAX_IDLE_GPU_UTILIZATION_PCT = 10


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    """Validated command-line benchmark options."""

    device: int
    force_peaks: bool
    kernels: tuple[str, ...] | None
    warmup_ms: int
    repetition_ms: int
    list_kernels: bool
    allow_busy_gpu: bool


def parse_args(argv: list[str] | None = None) -> BenchmarkOptions:
    """Parse benchmark CLI arguments.

    Args:
        argv: Optional argument list. ``None`` reads from ``sys.argv``.

    Returns:
        Immutable benchmark options.
    """
    parser = argparse.ArgumentParser(
        prog="tklab-bench",
        description="Benchmark registered Triton kernels and write JSON/PNG results.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument(
        "--force-peaks",
        action="store_true",
        help="Recalibrate bandwidth, cuBLAS, clocks, and theoretical ceilings.",
    )
    parser.add_argument(
        "--kernel",
        action="append",
        dest="kernels",
        choices=sorted(REGISTRY),
        help="Kernel to benchmark. Repeat the option to select multiple kernels.",
    )
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=25,
        help="Approximate Triton warmup duration per observation.",
    )
    parser.add_argument(
        "--repetition-ms",
        type=int,
        default=100,
        help="Approximate Triton measurement duration per observation.",
    )
    parser.add_argument(
        "--list-kernels",
        action="store_true",
        help="Print registered kernels and exit.",
    )
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Run despite pre-existing GPU utilization above 10%%.",
    )
    namespace = parser.parse_args(argv)
    return BenchmarkOptions(
        device=namespace.device,
        force_peaks=namespace.force_peaks,
        kernels=tuple(namespace.kernels) if namespace.kernels else None,
        warmup_ms=namespace.warmup_ms,
        repetition_ms=namespace.repetition_ms,
        list_kernels=namespace.list_kernels,
        allow_busy_gpu=namespace.allow_busy_gpu,
    )


def run_benchmarks(
    options: BenchmarkOptions,
    *,
    project_root: Path | None = None,
) -> dict[str, BenchmarkPayload]:
    """Execute selected benchmarks and persist result artifacts.

    Args:
        options: Validated CLI options.
        project_root: Repository root. Defaults to the installed source tree's
            repository root.

    Returns:
        Mapping from kernel name to its in-memory plot payload.

    Raises:
        RuntimeError: If CUDA is unavailable.
        ValueError: If no selected kernel exists.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarks")
    utilization = gpu_utilization_pct(options.device)
    if not options.allow_busy_gpu and utilization > MAX_IDLE_GPU_UTILIZATION_PCT:
        raise RuntimeError(
            f"GPU {options.device} is already {utilization}% utilized; "
            "close GPU workloads or pass --allow-busy-gpu to accept contaminated results"
        )
    selected_specs = _select_specs(options.kernels)
    root = project_root or Path(__file__).resolve().parents[2]
    results_root = root / "results"
    peaks, peaks_path = load_or_measure_peaks(
        results_root,
        device=options.device,
        force=options.force_peaks,
    )
    output_dir = peaks_path.parent
    _print_peak_summary(peaks, path=peaks_path)

    payloads: dict[str, BenchmarkPayload] = {}
    for spec in selected_specs:
        cublas_tflops, theoretical_tflops = _compute_baselines(spec, peaks)
        rows = bench_spec(
            spec,
            peak_bw_gbps=peaks["peak_bw_gbps"],
            cublas_tflops=cublas_tflops,
            theoretical_tflops=theoretical_tflops,
            device=options.device,
            warmup_ms=options.warmup_ms,
            repetition_ms=options.repetition_ms,
        )
        plot_payload: BenchmarkPayload = {"rows": rows}
        payloads[spec.name] = plot_payload
        result_payload = _result_payload(spec, rows=rows, peaks=peaks)
        output_path = output_dir / f"{spec.name}.json"
        write_json_atomic(output_path, result_payload)

        speedup_path = plot_speedup(plot_payload, output_dir)
        throughput_path = plot_throughput(plot_payload, output_dir)
        _print_result_summary(
            spec,
            rows=rows,
            output_path=output_path,
            speedup_path=speedup_path,
            throughput_path=throughput_path,
            has_cublas_baseline=cublas_tflops is not None,
        )

    if {"matmul_fp32acc", "matmul_fp16acc"}.issubset(payloads):
        comparison_path = plot_matmul_accumulation_comparison(
            payloads["matmul_fp32acc"],
            payloads["matmul_fp16acc"],
            output_dir,
        )
        print(f"matmul comparison plot: {comparison_path}")
    return payloads


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit status.
    """
    options = parse_args(argv)
    if options.list_kernels:
        for spec in REGISTRY.values():
            print(f"{spec.name}: {spec.description}")
        return 0
    run_benchmarks(options)
    return 0


def _select_specs(kernel_names: tuple[str, ...] | None) -> list[KernelSpec]:
    """Resolve selected kernel names while preserving registry order."""
    if kernel_names is None:
        return list(REGISTRY.values())
    names = set(kernel_names)
    specs = [spec for spec in REGISTRY.values() if spec.name in names]
    if not specs:
        raise ValueError("no kernels selected")
    return specs


def _compute_baselines(
    spec: KernelSpec,
    peaks: PeakResults,
) -> tuple[float | None, float | None]:
    """Select fair compute baselines for a kernel specification."""
    if spec.compute_mode == "fp16_fp32acc":
        return (
            peaks["cublas_tflops_fp16_fp32acc"],
            peaks["theoretical_tflops_fp16_fp32acc_at_measured_clock"],
        )
    if spec.compute_mode == "fp16_fp16acc":
        return (
            None,
            peaks["theoretical_tflops_fp16_fp16acc_at_measured_clock"],
        )
    if spec.compute_mode == "attention_fp16_fp32acc":
        return (
            None,
            peaks["theoretical_tflops_fp16_fp32acc_at_measured_clock"],
        )
    return None, None


def _result_payload(
    spec: KernelSpec,
    *,
    rows: list[BenchRow],
    peaks: PeakResults,
) -> JsonObject:
    """Build the versioned JSON payload for one kernel."""
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kernel": spec.name,
        "description": spec.description,
        "gpu": peaks["gpu_name"],
        "compute_capability": peaks["compute_capability"],
        "torch_version": peaks["torch_version"],
        "triton_version": peaks["triton_version"],
        "peak_bw_gbps": peaks["peak_bw_gbps"],
        "cublas_tflops_fp16_fp32acc": peaks["cublas_tflops_fp16_fp32acc"],
        "cublas_tflops_fp16_reduced_precision_allowed": peaks[
            "cublas_tflops_fp16_reduced_precision_allowed"
        ],
        "cublas_fp16acc_available_through_torch": peaks["cublas_fp16acc_available_through_torch"],
        "theoretical_tflops_fp16_fp32acc_at_measured_clock": peaks[
            "theoretical_tflops_fp16_fp32acc_at_measured_clock"
        ],
        "theoretical_tflops_fp16_fp16acc_at_measured_clock": peaks[
            "theoretical_tflops_fp16_fp16acc_at_measured_clock"
        ],
        "compute_mode": spec.compute_mode,
        "rows": rows,
    }
    return cast(JsonObject, payload)


def _print_peak_summary(peaks: PeakResults, *, path: Path) -> None:
    """Print a concise roofline summary."""
    print(f"peaks: {path}")
    print(f"measured bandwidth peak: {peaks['peak_bw_gbps']:.3f} GB/s")
    print(
        f"measured cuBLAS FP16/FP32-acc baseline: {peaks['cublas_tflops_fp16_fp32acc']:.3f} TFLOP/s"
    )
    print(
        "clock-scaled theoretical FP16/FP32-acc: "
        f"{peaks['theoretical_tflops_fp16_fp32acc_at_measured_clock']:.3f} TFLOP/s"
    )
    print(
        "clock-scaled theoretical FP16/FP16-acc: "
        f"{peaks['theoretical_tflops_fp16_fp16acc_at_measured_clock']:.3f} TFLOP/s"
    )


def _print_result_summary(
    spec: KernelSpec,
    *,
    rows: list[BenchRow],
    output_path: Path,
    speedup_path: Path,
    throughput_path: Path | None,
    has_cublas_baseline: bool,
) -> None:
    """Print result paths and the best primary metric."""
    if spec.bound == "memory":
        metric_value, baseline_name = _pct_peak, "bandwidth peak"
    elif has_cublas_baseline:
        metric_value, baseline_name = _pct_cublas_same_size, "same-size cuBLAS"
    else:
        metric_value, baseline_name = _pct_theoretical, "theoretical"

    best = max(rows, key=metric_value)
    performance = best.get("gbps", best.get("tflops", 0.0))
    print(f"results: {output_path}")
    print(f"speedup plot: {speedup_path}")
    if throughput_path is not None:
        print(f"throughput plot: {throughput_path}")
    print(f"best {spec.name}: {metric_value(best):.2f}% {baseline_name} ({performance:.3f})")


def _pct_peak(row: BenchRow) -> float:
    """Return a row's bandwidth-roofline percentage."""
    return row["pct_peak"]


def _pct_cublas_same_size(row: BenchRow) -> float:
    """Return a row's throughput relative to cuBLAS at the same shape."""
    return row["pct_cublas_same_size"]


def _pct_theoretical(row: BenchRow) -> float:
    """Return a row's theoretical-compute percentage."""
    return row["pct_theoretical"]


if __name__ == "__main__":
    raise SystemExit(main())
