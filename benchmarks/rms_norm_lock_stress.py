"""Stress RMSNorm's single-buffer lock reduction under heavy contention."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import TypedDict, cast

import torch
import triton

from tklab.harness.jsonio import JsonObject, write_json_atomic
from tklab.harness.roofline import gpu_slug, gpu_utilization_pct
from tklab.kernels.rms_norm import (
    _EPS,
    _launch_backward_stage1,
    _launch_backward_stage2,
    _launch_forward,
    _make_backward_buffers,
)

_DEFAULT_GROUP_SIZES = (1, 8, 256, 2048)
_MAX_REFERENCE_ERROR = 1e-2
_MAX_REPEAT_DRIFT = 1e-4
_MAX_GROUP_DRIFT = 1e-4


class GroupResult(TypedDict):
    """Stress result for one lock-group count."""

    group_size: int
    rows_per_slot: int
    runs: int
    locks_released_after_stage1: bool
    active_counts_initialized: bool
    inactive_counts_zero: bool
    bitwise_identical: bool
    unique_output_hashes: int
    max_repeat_relative_drift_dweight: float
    dweight_relative_frobenius: float


def _parse_args() -> argparse.Namespace:
    """Parse stress-test options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=1025)
    parser.add_argument("--columns", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument(
        "--group-size",
        type=int,
        action="append",
        dest="group_sizes",
        help="Lock-group count. Repeat to test several values.",
    )
    parser.add_argument("--allow-busy-gpu", action="store_true")
    return parser.parse_args()


def _relative_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    """Return relative Frobenius error, including an all-zero reference."""
    numerator = torch.linalg.vector_norm(output.float() - reference.float())
    denominator = torch.linalg.vector_norm(reference.float())
    if denominator.item() == 0.0:
        return float(numerator.item())
    return float((numerator / denominator).item())


def _output_hash(dweight: torch.Tensor) -> str:
    """Return a stable hash of one small CPU gradient vector."""
    return hashlib.sha256(dweight.contiguous().numpy().tobytes()).hexdigest()


def _run_group(
    *,
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    reference_dw: torch.Tensor,
    group_size: int,
    runs: int,
) -> tuple[GroupResult, torch.Tensor]:
    """Run repeated lock reductions and inspect lock/count state."""
    buffers = _make_backward_buffers(x, weight, group_size=group_size)
    baseline_dw: torch.Tensor | None = None
    hashes: set[str] = set()
    bitwise_identical = True
    max_dw_drift = 0.0
    locks_released = True
    active_counts_initialized = True
    inactive_counts_zero = True

    for _ in range(runs):
        buffers.locks.zero_()
        _launch_backward_stage1(dy, x, weight, rstd, buffers)
        torch.cuda.synchronize()

        state = buffers.locks.detach().cpu()
        locks = state[: buffers.group_size]
        counts = state[buffers.group_size :]
        locks_released = locks_released and bool(torch.all(locks == 0))
        active_counts_initialized = active_counts_initialized and bool(
            torch.all(counts[: buffers.group_count] == 1)
        )
        inactive_counts_zero = inactive_counts_zero and bool(
            torch.all(counts[buffers.group_count :] == 0)
        )

        _launch_backward_stage2(buffers, x.shape[1])
        torch.cuda.synchronize()
        current_dw = buffers.dw.detach().cpu()
        hashes.add(_output_hash(current_dw))
        if baseline_dw is None:
            baseline_dw = current_dw
            continue
        bitwise_identical = bitwise_identical and torch.equal(current_dw, baseline_dw)
        max_dw_drift = max(max_dw_drift, _relative_error(current_dw, baseline_dw))

    if baseline_dw is None:
        raise RuntimeError("stress test produced no outputs")
    result: GroupResult = {
        "group_size": group_size,
        "rows_per_slot": (x.shape[0] + group_size - 1) // group_size,
        "runs": runs,
        "locks_released_after_stage1": locks_released,
        "active_counts_initialized": active_counts_initialized,
        "inactive_counts_zero": inactive_counts_zero,
        "bitwise_identical": bitwise_identical,
        "unique_output_hashes": len(hashes),
        "max_repeat_relative_drift_dweight": max_dw_drift,
        "dweight_relative_frobenius": _relative_error(
            baseline_dw,
            reference_dw.cpu(),
        ),
    }
    return result, baseline_dw


def main() -> int:
    """Execute and persist the contention stress study."""
    options = _parse_args()
    utilization = gpu_utilization_pct(options.device)
    if not options.allow_busy_gpu and utilization > 10:
        raise RuntimeError(
            f"GPU {options.device} is already {utilization}% utilized; "
            "close GPU workloads or pass --allow-busy-gpu"
        )
    if options.rows <= 0 or options.columns <= 0 or options.runs <= 0:
        raise ValueError("rows, columns, and runs must be positive")
    group_sizes = tuple(options.group_sizes or _DEFAULT_GROUP_SIZES)
    if any(group_size <= 0 for group_size in group_sizes):
        raise ValueError("group sizes must be positive")
    if len(set(group_sizes)) != len(group_sizes):
        raise ValueError("group sizes must be unique")

    torch.manual_seed(17)
    device = torch.device("cuda", options.device)
    x_storage = torch.randn(
        options.rows * 2,
        options.columns,
        device=device,
        dtype=torch.float32,
    )
    dy_storage = torch.randn(
        options.rows * 3,
        options.columns,
        device=device,
        dtype=torch.float32,
    )
    x = x_storage[::2]
    dy = dy_storage[::3]
    weight = torch.randn(options.columns, device=device, dtype=torch.float32)
    output = torch.empty_like(x)
    rstd = torch.empty(options.rows, device=device, dtype=torch.float32)
    _launch_forward(x, weight, output, rstd, eps=_EPS, store_stats=True)
    reference_rstd = torch.rsqrt(torch.mean(x * x, dim=1) + _EPS)
    reference_dw = torch.sum(dy * x * reference_rstd[:, None], dim=0)

    group_results: list[GroupResult] = []
    group_outputs: dict[int, torch.Tensor] = {}
    for group_size in group_sizes:
        result, dweight = _run_group(
            x=x,
            dy=dy,
            weight=weight,
            rstd=rstd,
            reference_dw=reference_dw,
            group_size=group_size,
            runs=options.runs,
        )
        group_results.append(result)
        group_outputs[group_size] = dweight
        print(
            f"group={group_size}, rows/slot={result['rows_per_slot']}, "
            f"lock={result['locks_released_after_stage1']}, "
            f"count={result['active_counts_initialized']}, "
            f"inactive={result['inactive_counts_zero']}, "
            f"hashes={result['unique_output_hashes']}, "
            f"dw_error={result['dweight_relative_frobenius']:.3e}, "
            f"dw_drift={result['max_repeat_relative_drift_dweight']:.3e}"
        )

    first_group = group_sizes[0]
    first_dw = group_outputs[first_group]
    invariance = [
        {
            "reference_group_size": first_group,
            "group_size": group_size,
            "dweight_relative_frobenius": _relative_error(
                group_outputs[group_size],
                first_dw,
            ),
        }
        for group_size in group_sizes[1:]
    ]
    if any(
        not result["locks_released_after_stage1"]
        or not result["active_counts_initialized"]
        or not result["inactive_counts_zero"]
        for result in group_results
    ):
        raise AssertionError("lock/count state violated the stage-1 protocol")
    if any(
        result["dweight_relative_frobenius"] >= _MAX_REFERENCE_ERROR for result in group_results
    ):
        raise AssertionError("lock stress exceeded the PyTorch-reference error threshold")
    if any(
        result["max_repeat_relative_drift_dweight"] >= _MAX_REPEAT_DRIFT for result in group_results
    ):
        raise AssertionError("lock stress exceeded the repeat-drift threshold")
    if any(result["dweight_relative_frobenius"] >= _MAX_GROUP_DRIFT for result in invariance):
        raise AssertionError("lock stress exceeded the group-invariance threshold")

    major, minor = torch.cuda.get_device_capability(options.device)
    payload = {
        "schema_version": 1,
        "kernel": "rms_norm_lock_stress",
        "gpu": torch.cuda.get_device_name(options.device),
        "compute_capability": f"{major}.{minor}",
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "rows": options.rows,
        "columns": options.columns,
        "row_strided_input": True,
        "row_strided_upstream_gradient": True,
        "runs_per_group": options.runs,
        "group_results": group_results,
        "group_invariance": invariance,
        "interpretation": {
            "bitwise_identity_expected": False,
            "reason": "lock acquisition order changes floating-point reduction order",
            "race_signal": "invalid lock/count state, large drift, or reference error",
            "max_reference_error": _MAX_REFERENCE_ERROR,
            "max_repeat_drift": _MAX_REPEAT_DRIFT,
            "max_group_drift": _MAX_GROUP_DRIFT,
        },
    }
    root = Path(__file__).resolve().parents[1]
    output_path = root / "results" / gpu_slug(options.device) / "rms_norm_lock_stress.json"
    write_json_atomic(output_path, cast(JsonObject, payload))
    print(f"results: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
