"""Benchmark execution and metric calculation."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TypedDict, cast

import torch
import triton
from typing_extensions import NotRequired

from tklab.registry import KernelSpec


class BenchRow(TypedDict):
    """One serialized benchmark observation."""

    kernel: str
    dtype: str
    size: int
    ms: float
    ms_lo: float
    ms_hi: float
    torch_ms: float
    speedup: float
    gbps: NotRequired[float]
    tflops: NotRequired[float]
    pct_peak: NotRequired[float]
    naive_ms: NotRequired[float]
    speedup_vs_naive: NotRequired[float]
    reference_baseline: NotRequired[str]
    naive_baseline: NotRequired[str]
    reference_allocates_output: NotRequired[bool]
    reference_tflops: NotRequired[float]
    pct_cublas_same_size: NotRequired[float]
    pct_cublas_peak: NotRequired[float]
    pct_theoretical: NotRequired[float]
    block_m: NotRequired[int]
    block_n: NotRequired[int]
    block_k: NotRequired[int]
    group_m: NotRequired[int]
    num_warps: NotRequired[int]
    num_stages: NotRequired[int]
    mma_opcode: NotRequired[str]
    heads: NotRequired[int]
    head_dim: NotRequired[int]
    causal: NotRequired[bool]


def bench_spec(
    spec: KernelSpec,
    *,
    peak_bw_gbps: float | None,
    cublas_tflops: float | None,
    theoretical_tflops: float | None,
    device: int = 0,
    warmup_ms: int = 25,
    repetition_ms: int = 100,
) -> list[BenchRow]:
    """Benchmark every dtype and size in a kernel specification.

    Args:
        spec: Kernel contract to benchmark.
        peak_bw_gbps: Empirical bandwidth roofline for memory-bound kernels.
        cublas_tflops: Optional practical cuBLAS baseline.
        theoretical_tflops: Clock-scaled compute ceiling.
        device: CUDA device index.
        warmup_ms: Approximate warmup duration used by Triton.
        repetition_ms: Approximate measurement duration used by Triton.

    Returns:
        Ordered benchmark observations suitable for JSON serialization.

    Raises:
        ValueError: If timing parameters or required baselines are invalid.
        RuntimeError: If the selected CUDA device is unavailable.
    """
    _validate_benchmark_request(
        device=device,
        warmup_ms=warmup_ms,
        repetition_ms=repetition_ms,
    )
    rows: list[BenchRow] = []
    torch_device = torch.device("cuda", device)

    for dtype in spec.dtypes:
        for size in spec.sizes:
            args = spec.make_inputs(size, torch_device, dtype)
            output = spec.make_output(args)
            launch = (
                spec.benchmark_call_factory(args, output)
                if spec.benchmark_call_factory is not None
                else partial(spec.launch_fn, args, output)
            )

            median_ms, low_ms, high_ms = triton.testing.do_bench(
                launch,
                quantiles=[0.5, 0.2, 0.8],
                warmup=warmup_ms,
                rep=repetition_ms,
            )
            torch_ms = _benchmark_reference(
                spec,
                args=args,
                output=output,
                warmup_ms=warmup_ms,
                repetition_ms=repetition_ms,
            )
            row: BenchRow = {
                "kernel": spec.name,
                "dtype": str(dtype).removeprefix("torch."),
                "size": size,
                "ms": float(median_ms),
                "ms_lo": float(low_ms),
                "ms_hi": float(high_ms),
                "torch_ms": torch_ms,
                "speedup": torch_ms / float(median_ms),
            }

            if spec.benchmark_metadata is not None:
                metadata = cast(BenchRow, dict(spec.benchmark_metadata(size, dtype)))
                row.update(metadata)

            naive_call: Callable[[], object] | None = None
            if spec.naive_call_factory is not None:
                naive_call = spec.naive_call_factory(args, output)
            elif spec.naive_fn is not None:
                naive_call = partial(spec.naive_fn, *args)
            if naive_call is not None:
                naive_ms = float(
                    triton.testing.do_bench(
                        naive_call,
                        warmup=warmup_ms,
                        rep=repetition_ms,
                        return_mode="median",
                    )
                )
                row["naive_ms"] = naive_ms
                row["speedup_vs_naive"] = naive_ms / float(median_ms)

            _add_performance_metrics(
                row,
                spec=spec,
                size=size,
                dtype=dtype,
                milliseconds=float(median_ms),
                peak_bw_gbps=peak_bw_gbps,
                cublas_tflops=cublas_tflops,
                theoretical_tflops=theoretical_tflops,
            )
            rows.append(row)

    return rows


def _benchmark_reference(
    spec: KernelSpec,
    *,
    args: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    warmup_ms: int,
    repetition_ms: int,
) -> float:
    """Benchmark the PyTorch reference with allocation removed when possible."""
    reference_call: Callable[[], object]
    if spec.reference_call_factory is not None:
        reference_call = spec.reference_call_factory(args, output)
    elif spec.reference_launch_fn is not None:
        reference_call = partial(spec.reference_launch_fn, args, output)
    else:
        reference_call = partial(spec.ref_fn, *args)
    return float(
        triton.testing.do_bench(
            reference_call,
            warmup=warmup_ms,
            rep=repetition_ms,
            return_mode="median",
        )
    )


def _add_performance_metrics(
    row: BenchRow,
    *,
    spec: KernelSpec,
    size: int,
    dtype: torch.dtype,
    milliseconds: float,
    peak_bw_gbps: float | None,
    cublas_tflops: float | None,
    theoretical_tflops: float | None,
) -> None:
    """Add roofline metrics to a benchmark row.

    Raises:
        ValueError: If the specification lacks its required cost model or
            baseline.
    """
    seconds = milliseconds * 1e-3
    if spec.bound == "memory":
        if spec.bytes_moved is None or peak_bw_gbps is None:
            raise ValueError(f"{spec.name}: missing memory cost model or bandwidth peak")
        gigabytes_per_second = spec.bytes_moved(size, dtype) / seconds / 1e9
        row["gbps"] = gigabytes_per_second
        row["pct_peak"] = 100.0 * gigabytes_per_second / peak_bw_gbps
        return

    if spec.flops is None or theoretical_tflops is None:
        raise ValueError(f"{spec.name}: missing compute cost model or theoretical peak")
    teraflops = spec.flops(size, dtype) / seconds / 1e12
    reference_teraflops = spec.flops(size, dtype) / (row["torch_ms"] * 1e-3) / 1e12
    row["tflops"] = teraflops
    row["reference_tflops"] = reference_teraflops
    row["pct_theoretical"] = 100.0 * teraflops / theoretical_tflops
    if cublas_tflops is not None:
        row["pct_cublas_same_size"] = 100.0 * teraflops / reference_teraflops
        row["pct_cublas_peak"] = 100.0 * teraflops / cublas_tflops


def _validate_benchmark_request(
    *,
    device: int,
    warmup_ms: int,
    repetition_ms: int,
) -> None:
    """Validate benchmark runtime parameters.

    Raises:
        ValueError: If an integer parameter is negative or zero.
        RuntimeError: If CUDA or the selected device is unavailable.
    """
    if device < 0:
        raise ValueError("device index must be non-negative")
    if warmup_ms <= 0 or repetition_ms <= 0:
        raise ValueError("warmup and repetition durations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU benchmarks")
    if device >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {device} is unavailable; found {torch.cuda.device_count()} device(s)"
        )
