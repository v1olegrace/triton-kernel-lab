"""Benchmark LayerNorm backward stages and serialize gradient evidence."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import TypedDict, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import triton

from tklab.harness.jsonio import JsonObject, write_json_atomic
from tklab.harness.roofline import gpu_slug, gpu_utilization_pct
from tklab.kernels.layer_norm import (
    _EPS,
    _launch_backward_stage1,
    _launch_backward_stage2,
    _launch_forward,
    _make_backward_buffers,
    layer_norm,
)

_DEFAULT_ROWS = 4096
_DEFAULT_COLUMNS = (1024, 2048, 4096, 8192, 16384)


class BackwardRow(TypedDict):
    """One two-stage backward timing observation."""

    rows: int
    columns: int
    stage1_ms: float
    stage2_standalone_ms: float
    total_kernel_ms: float
    stage2_incremental_ms: float
    stage2_incremental_pct_total: float
    effective_gbps: float
    group_size: int
    group_count: int


def _parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=_DEFAULT_ROWS)
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--repetition-ms", type=int, default=200)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    return parser.parse_args()


def _gradient_errors(device: torch.device) -> dict[str, float | int]:
    """Measure FP32 forward and gradient errors on a strided tail case."""
    torch.manual_seed(7)
    rows, columns = 67, 1000
    x_storage = torch.randn(rows * 2, columns, device=device, dtype=torch.float32)
    dy_storage = torch.randn(rows * 3, columns, device=device, dtype=torch.float32)
    x = x_storage[::2].detach().requires_grad_(True)
    dy = dy_storage[::3]
    weight = torch.randn(columns, device=device, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(columns, device=device, dtype=torch.float32, requires_grad=True)

    reference = F.layer_norm(x, (columns,), weight, bias, _EPS)
    dx_ref, dw_ref, db_ref = torch.autograd.grad(
        reference,
        (x, weight, bias),
        grad_outputs=dy,
    )

    x_tri = x.detach().requires_grad_(True)
    weight_tri = weight.detach().requires_grad_(True)
    bias_tri = bias.detach().requires_grad_(True)
    output = layer_norm(x_tri, weight_tri, bias_tri, _EPS)
    dx_tri, dw_tri, db_tri = torch.autograd.grad(
        output,
        (x_tri, weight_tri, bias_tri),
        grad_outputs=dy,
    )
    return {
        "rows": rows,
        "columns": columns,
        "forward_relative_frobenius": _relative_error(output, reference),
        "dx_relative_frobenius": _relative_error(dx_tri, dx_ref),
        "dweight_relative_frobenius": _relative_error(dw_tri, dw_ref),
        "dbias_relative_frobenius": _relative_error(db_tri, db_ref),
    }


def _relative_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    """Return relative Frobenius error."""
    numerator = torch.linalg.vector_norm(output.float() - reference.float())
    denominator = torch.linalg.vector_norm(reference.float())
    return float((numerator / denominator).item())


def _benchmark_size(
    rows: int,
    columns: int,
    *,
    device: torch.device,
    warmup_ms: int,
    repetition_ms: int,
) -> BackwardRow:
    """Benchmark stage 1, stage 2, and their combined kernel sequence."""
    x = torch.randn(rows, columns, device=device, dtype=torch.float16)
    weight = torch.randn(columns, device=device, dtype=torch.float16)
    bias = torch.randn(columns, device=device, dtype=torch.float16)
    dy = torch.randn_like(x)
    output = torch.empty_like(x)
    mean = torch.empty(rows, device=device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    _launch_forward(x, weight, bias, output, mean, rstd, eps=_EPS, store_stats=True)
    buffers = _make_backward_buffers(x, weight)

    launch_stage1 = partial(_launch_backward_stage1, dy, x, weight, mean, rstd, buffers)
    stage2 = partial(_launch_backward_stage2, buffers, columns)

    def stage1() -> None:
        """Reset lock/count state, then launch stage 1."""
        buffers.locks.zero_()
        launch_stage1()

    def total() -> None:
        """Launch both backward stages from clean lock/count state."""
        stage1()
        stage2()

    stage1()
    stage2()
    stage1_ms = float(
        triton.testing.do_bench(
            stage1,
            warmup=warmup_ms,
            rep=repetition_ms,
            return_mode="median",
        )
    )
    stage2_standalone_ms = float(
        triton.testing.do_bench(
            stage2,
            warmup=warmup_ms,
            rep=repetition_ms,
            return_mode="median",
        )
    )
    total_ms = float(
        triton.testing.do_bench(
            total,
            warmup=warmup_ms,
            rep=repetition_ms,
            return_mode="median",
        )
    )
    stage2_incremental_ms = max(total_ms - stage1_ms, 0.0)
    effective_bytes = 3 * x.numel() * x.element_size()
    return {
        "rows": rows,
        "columns": columns,
        "stage1_ms": stage1_ms,
        "stage2_standalone_ms": stage2_standalone_ms,
        "total_kernel_ms": total_ms,
        "stage2_incremental_ms": stage2_incremental_ms,
        "stage2_incremental_pct_total": 100.0 * stage2_incremental_ms / total_ms,
        "effective_gbps": effective_bytes / (total_ms * 1e-3) / 1e9,
        "group_size": buffers.group_size,
        "group_count": buffers.group_count,
    }


def main() -> int:
    """Run the backward study and write one per-GPU JSON artifact."""
    options = _parse_args()
    utilization = gpu_utilization_pct(options.device)
    if not options.allow_busy_gpu and utilization > 10:
        raise RuntimeError(
            f"GPU {options.device} is already {utilization}% utilized; "
            "close GPU workloads or pass --allow-busy-gpu"
        )
    if options.rows <= 0:
        raise ValueError("rows must be positive")
    device = torch.device("cuda", options.device)
    rows = [
        _benchmark_size(
            options.rows,
            columns,
            device=device,
            warmup_ms=options.warmup_ms,
            repetition_ms=options.repetition_ms,
        )
        for columns in _DEFAULT_COLUMNS
    ]
    major, minor = torch.cuda.get_device_capability(options.device)
    payload = {
        "schema_version": 1,
        "kernel": "layer_norm_backward",
        "gpu": torch.cuda.get_device_name(options.device),
        "compute_capability": f"{major}.{minor}",
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "methodology": {
            "allocations_timed": False,
            "stage1_includes_lock_reset": True,
            "stage2_incremental_definition": "total_kernel_ms - stage1_ms",
            "stage2_standalone_has_separate_l2_flush": True,
        },
        "gradient_validation": _gradient_errors(device),
        "rows": rows,
    }
    root = Path(__file__).resolve().parents[1]
    output = root / "results" / gpu_slug(options.device) / "layer_norm_backward.json"
    write_json_atomic(output, cast(JsonObject, payload))
    plot_path = _plot_rows(rows, output.parent)
    print(f"results: {output}")
    print(f"stage plot: {plot_path}")
    for row in rows:
        print(
            f"N={row['columns']}: total={row['total_kernel_ms']:.4f} ms, "
            f"stage2 incremental={row['stage2_incremental_ms']:.4f} ms "
            f"({row['stage2_incremental_pct_total']:.2f}%)"
        )
    return 0


def _plot_rows(rows: list[BackwardRow], output_dir: Path) -> Path:
    """Plot backward stage timing and incremental stage-2 overhead."""
    output_path = output_dir / "layer_norm_backward_stages.png"
    columns = [row["columns"] for row in rows]
    stage1 = [row["stage1_ms"] for row in rows]
    incremental = [max(row["stage2_incremental_ms"], 1e-6) for row in rows]
    overhead = [row["stage2_incremental_pct_total"] for row in rows]

    figure, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    timing_axis, overhead_axis = axes
    timing_axis.plot(columns, stage1, marker="o", label="Stage 1: dx + partials")
    timing_axis.plot(columns, incremental, marker="s", label="Stage 2 incremental")
    timing_axis.set_xscale("log", base=2)
    timing_axis.set_yscale("log")
    timing_axis.set_ylabel("Milliseconds")
    timing_axis.set_title("LayerNorm backward stages")
    timing_axis.grid(True, which="both", alpha=0.3)
    timing_axis.legend()

    overhead_axis.plot(columns, overhead, marker="o", color="tab:purple")
    overhead_axis.set_xscale("log", base=2)
    overhead_axis.set_xlabel("Feature width")
    overhead_axis.set_ylabel("Stage 2 / total (%)")
    overhead_axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    raise SystemExit(main())
