"""Stress LayerNorm's global-memory lock reduction under heavy contention."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import TypedDict, cast

import torch
import triton

from tklab.harness.jsonio import JsonObject, write_json_atomic
from tklab.harness.roofline import gpu_slug, gpu_utilization_pct
from tklab.kernels._norm_common import EPS as _EPS
from tklab.kernels.layer_norm import (
    _launch_backward_stage1,
    _launch_backward_stage2,
    _launch_forward,
    _make_backward_buffers,
)

_DEFAULT_GROUP_SIZES = (8, 32, 128)
_MAX_REFERENCE_ERROR = 1e-2
_MAX_REPEAT_DRIFT = 1e-4
_MAX_GROUP_DRIFT = 1e-4


class GroupResult(TypedDict):
    """Stress result for one lock-group count."""

    group_size: int
    rows_per_slot: int
    runs: int
    bitwise_identical: bool
    unique_output_hashes: int
    max_repeat_relative_drift_dweight: float
    max_repeat_relative_drift_dbias: float
    dweight_relative_frobenius: float
    dbias_relative_frobenius: float


def _parse_args() -> argparse.Namespace:
    """Parse stress-test options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=1 << 16)
    parser.add_argument("--columns", type=int, default=1024)
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


def _output_hash(dweight: torch.Tensor, dbias: torch.Tensor) -> str:
    """Return a stable hash of two small CPU gradient vectors."""
    digest = hashlib.sha256()
    digest.update(dweight.contiguous().numpy().tobytes())
    digest.update(dbias.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _torch_reference(
    x: torch.Tensor,
    dy: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute parameter gradients with PyTorch FP32 reductions."""
    with torch.no_grad():
        x_hat = (x - mean[:, None]) * rstd[:, None]
        dweight = torch.sum(dy * x_hat, dim=0)
        dbias = torch.sum(dy, dim=0)
    return dweight, dbias


def _run_group(
    *,
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    reference_dw: torch.Tensor,
    reference_db: torch.Tensor,
    group_size: int,
    runs: int,
) -> tuple[GroupResult, torch.Tensor, torch.Tensor]:
    """Run repeated lock reductions for one group count."""
    buffers = _make_backward_buffers(x, weight, group_size=group_size)
    baseline_dw: torch.Tensor | None = None
    baseline_db: torch.Tensor | None = None
    hashes: set[str] = set()
    bitwise_identical = True
    max_dw_drift = 0.0
    max_db_drift = 0.0

    for _ in range(runs):
        buffers.locks.zero_()
        _launch_backward_stage1(dy, x, weight, mean, rstd, buffers)
        _launch_backward_stage2(buffers, x.shape[1])
        torch.cuda.synchronize()
        current_dw = buffers.dw.detach().cpu()
        current_db = buffers.db.detach().cpu()
        hashes.add(_output_hash(current_dw, current_db))
        if baseline_dw is None or baseline_db is None:
            baseline_dw = current_dw
            baseline_db = current_db
            continue
        bitwise_identical = bitwise_identical and torch.equal(current_dw, baseline_dw)
        bitwise_identical = bitwise_identical and torch.equal(current_db, baseline_db)
        max_dw_drift = max(max_dw_drift, _relative_error(current_dw, baseline_dw))
        max_db_drift = max(max_db_drift, _relative_error(current_db, baseline_db))

    if baseline_dw is None or baseline_db is None:
        raise RuntimeError("stress test produced no outputs")
    result: GroupResult = {
        "group_size": group_size,
        "rows_per_slot": (x.shape[0] + group_size - 1) // group_size,
        "runs": runs,
        "bitwise_identical": bitwise_identical,
        "unique_output_hashes": len(hashes),
        "max_repeat_relative_drift_dweight": max_dw_drift,
        "max_repeat_relative_drift_dbias": max_db_drift,
        "dweight_relative_frobenius": _relative_error(baseline_dw, reference_dw.cpu()),
        "dbias_relative_frobenius": _relative_error(baseline_db, reference_db.cpu()),
    }
    return result, baseline_dw, baseline_db


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

    torch.manual_seed(11)
    device = torch.device("cuda", options.device)
    x = torch.randn(options.rows, options.columns, device=device, dtype=torch.float32)
    dy = torch.randn_like(x)
    weight = torch.randn(options.columns, device=device, dtype=torch.float32)
    bias = torch.randn(options.columns, device=device, dtype=torch.float32)
    output = torch.empty_like(x)
    mean = torch.empty(options.rows, device=device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    _launch_forward(x, weight, bias, output, mean, rstd, eps=_EPS, store_stats=True)
    reference_dw, reference_db = _torch_reference(x, dy, mean, rstd)

    group_results: list[GroupResult] = []
    group_outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for group_size in group_sizes:
        result, dweight, dbias = _run_group(
            x=x,
            dy=dy,
            weight=weight,
            mean=mean,
            rstd=rstd,
            reference_dw=reference_dw,
            reference_db=reference_db,
            group_size=group_size,
            runs=options.runs,
        )
        group_results.append(result)
        group_outputs[group_size] = (dweight, dbias)
        print(
            f"group={group_size}, rows/slot={result['rows_per_slot']}, "
            f"bitwise={result['bitwise_identical']}, hashes={result['unique_output_hashes']}, "
            f"dw_error={result['dweight_relative_frobenius']:.3e}, "
            f"db_error={result['dbias_relative_frobenius']:.3e}, "
            f"dw_drift={result['max_repeat_relative_drift_dweight']:.3e}, "
            f"db_drift={result['max_repeat_relative_drift_dbias']:.3e}"
        )

    first_group = group_sizes[0]
    first_dw, first_db = group_outputs[first_group]
    invariance = [
        {
            "reference_group_size": first_group,
            "group_size": group_size,
            "dweight_relative_frobenius": _relative_error(
                group_outputs[group_size][0],
                first_dw,
            ),
            "dbias_relative_frobenius": _relative_error(
                group_outputs[group_size][1],
                first_db,
            ),
        }
        for group_size in group_sizes[1:]
    ]
    if any(
        result["dweight_relative_frobenius"] >= _MAX_REFERENCE_ERROR
        or result["dbias_relative_frobenius"] >= _MAX_REFERENCE_ERROR
        for result in group_results
    ):
        raise AssertionError("lock stress exceeded the PyTorch-reference error threshold")
    if any(
        result["max_repeat_relative_drift_dweight"] >= _MAX_REPEAT_DRIFT
        or result["max_repeat_relative_drift_dbias"] >= _MAX_REPEAT_DRIFT
        for result in group_results
    ):
        raise AssertionError("lock stress exceeded the repeat-drift threshold")
    if any(
        result["dweight_relative_frobenius"] >= _MAX_GROUP_DRIFT
        or result["dbias_relative_frobenius"] >= _MAX_GROUP_DRIFT
        for result in invariance
    ):
        raise AssertionError("lock stress exceeded the group-invariance threshold")
    major, minor = torch.cuda.get_device_capability(options.device)
    payload = {
        "schema_version": 1,
        "kernel": "layer_norm_lock_stress",
        "gpu": torch.cuda.get_device_name(options.device),
        "compute_capability": f"{major}.{minor}",
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "rows": options.rows,
        "columns": options.columns,
        "runs_per_group": options.runs,
        "group_results": group_results,
        "group_invariance": invariance,
        "interpretation": {
            "bitwise_identity_expected": False,
            "reason": "lock acquisition order changes floating-point reduction order",
            "race_signal": "large repeat drift, reference error, or group-dependent error",
            "max_reference_error": _MAX_REFERENCE_ERROR,
            "max_repeat_drift": _MAX_REPEAT_DRIFT,
            "max_group_drift": _MAX_GROUP_DRIFT,
        },
    }
    root = Path(__file__).resolve().parents[1]
    output_path = root / "results" / gpu_slug(options.device) / "layer_norm_lock_stress.json"
    write_json_atomic(output_path, cast(JsonObject, payload))
    print(f"results: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
