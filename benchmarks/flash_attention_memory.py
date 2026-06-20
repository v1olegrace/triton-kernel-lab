"""Measure Flash Attention and materialized-attention peak CUDA allocation."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, cast

import torch
import triton

from tklab.harness.jsonio import JsonObject, write_json_atomic
from tklab.harness.plots import AttentionMemoryRow, plot_attention_memory
from tklab.harness.roofline import gpu_slug, gpu_utilization_pct
from tklab.kernels.flash_attention import attention_noncausal

_DEFAULT_SIZES = (512, 1024, 2048, 4096, 8192, 12288, 16384, 24576)


class WorkerResult(TypedDict):
    """One isolated-process memory measurement."""

    implementation: str
    sequence_length: int
    baseline_allocated_bytes: int
    peak_allocated_bytes: int | None
    peak_increment_bytes: int | None
    output_finite: bool | None
    oom: bool
    error: str | None


def _parse_args() -> argparse.Namespace:
    """Parse parent and worker options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--size", type=int, action="append", dest="sizes")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--implementation",
        choices=("flash", "materialized"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-size", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def _materialized_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Compute attention while explicitly retaining the quadratic matrices."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)


def _worker(implementation: str, sequence_length: int, device_index: int) -> int:
    """Measure one implementation in a fresh process so OOM is recoverable."""
    torch.cuda.set_device(device_index)
    torch.manual_seed(31)
    shape = (1, 16, sequence_length, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)
    if implementation == "flash":
        warm_output = attention_noncausal(q, k, v)
        torch.cuda.synchronize()
        del warm_output
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    result: WorkerResult
    try:
        output = (
            attention_noncausal(q, k, v)
            if implementation == "flash"
            else _materialized_attention(q, k, v)
        )
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device_index)
        result = {
            "implementation": implementation,
            "sequence_length": sequence_length,
            "baseline_allocated_bytes": baseline,
            "peak_allocated_bytes": peak,
            "peak_increment_bytes": peak - baseline,
            "output_finite": bool(torch.isfinite(output).all().item()),
            "oom": False,
            "error": None,
        }
    except torch.OutOfMemoryError as error:
        result = {
            "implementation": implementation,
            "sequence_length": sequence_length,
            "baseline_allocated_bytes": baseline,
            "peak_allocated_bytes": None,
            "peak_increment_bytes": None,
            "output_finite": None,
            "oom": True,
            "error": str(error).splitlines()[0],
        }
    print(json.dumps(result, separators=(",", ":")))
    return 0


def _run_worker(
    script: Path,
    *,
    implementation: str,
    sequence_length: int,
    device: int,
) -> WorkerResult:
    """Run and parse one isolated memory worker."""
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--worker",
            "--implementation",
            implementation,
            "--worker-size",
            str(sequence_length),
            "--device",
            str(device),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    line = completed.stdout.strip().splitlines()[-1]
    return cast(WorkerResult, json.loads(line))


def _empirical_slope(rows: list[WorkerResult]) -> float | None:
    """Estimate the log-log slope from the first and last successful points."""
    successful = [
        row
        for row in rows
        if not row["oom"]
        and row["peak_increment_bytes"] is not None
        and row["peak_increment_bytes"] > 0
    ]
    if len(successful) < 2:
        return None
    first, last = successful[0], successful[-1]
    size_ratio = last["sequence_length"] / first["sequence_length"]
    memory_ratio = cast(int, last["peak_increment_bytes"]) / cast(
        int,
        first["peak_increment_bytes"],
    )
    return math.log(memory_ratio) / math.log(size_ratio)


def main() -> int:
    """Run workers, persist JSON evidence, and generate the memory curve."""
    options = _parse_args()
    if options.worker:
        if options.implementation is None or options.worker_size is None:
            raise ValueError("worker mode requires implementation and size")
        return _worker(options.implementation, options.worker_size, options.device)

    utilization = gpu_utilization_pct(options.device)
    if not options.allow_busy_gpu and utilization > 10:
        raise RuntimeError(
            f"GPU {options.device} is already {utilization}% utilized; "
            "close GPU workloads or pass --allow-busy-gpu"
        )
    sizes = tuple(options.sizes or _DEFAULT_SIZES)
    if any(size <= 0 for size in sizes):
        raise ValueError("sequence lengths must be positive")

    script = Path(__file__).resolve()
    measurements: list[WorkerResult] = []
    for implementation in ("flash", "materialized"):
        for size in sizes:
            result = _run_worker(
                script,
                implementation=implementation,
                sequence_length=size,
                device=options.device,
            )
            measurements.append(result)
            peak = result["peak_increment_bytes"]
            status = "OOM" if result["oom"] else f"{cast(int, peak) / 2**20:.2f} MiB"
            print(f"{implementation} N={size}: {status}")

    flash_rows = [row for row in measurements if row["implementation"] == "flash"]
    materialized_rows = [row for row in measurements if row["implementation"] == "materialized"]
    major, minor = torch.cuda.get_device_capability(options.device)
    payload = {
        "schema_version": 1,
        "study": "flash_attention_memory",
        "gpu": torch.cuda.get_device_name(options.device),
        "compute_capability": f"{major}.{minor}",
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "batch": 1,
        "heads": 16,
        "head_dim": 64,
        "dtype": "float16",
        "measurements": measurements,
        "empirical_log_log_slope": {
            "flash": _empirical_slope(flash_rows),
            "materialized": _empirical_slope(materialized_rows),
        },
    }
    root = script.parents[1]
    output_dir = root / "results" / gpu_slug(options.device)
    json_path = output_dir / "flash_attention_memory.json"
    write_json_atomic(json_path, cast(JsonObject, payload))
    plot_rows: list[AttentionMemoryRow] = [
        {
            "implementation": row["implementation"],
            "sequence_length": row["sequence_length"],
            "peak_increment_bytes": row["peak_increment_bytes"],
            "oom": row["oom"],
        }
        for row in measurements
    ]
    plot_path = plot_attention_memory(plot_rows, output_dir)
    print(f"results: {json_path}")
    print(f"plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
