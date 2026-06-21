"""Real-GPU forward and gradient tests for Triton RMSNorm."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tklab.harness.tolerances import assert_relative_frobenius
from tklab.kernels.rms_norm import (
    _launch_backward_stage1,
    _launch_backward_stage2,
    _launch_forward,
    _make_backward_buffers,
    rms_norm,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required"),
]

_EPS = 1e-5


def _strided_problem(
    *,
    rows: int = 37,
    columns: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create FP32 leaves with row-strided input and upstream gradient."""
    x_storage = torch.randn(rows * 2, columns, device="cuda", dtype=torch.float32)
    dy_storage = torch.randn(rows * 3, columns, device="cuda", dtype=torch.float32)
    x = x_storage[::2].detach().requires_grad_(True)
    dy = dy_storage[::3]
    weight = torch.randn(columns, device="cuda", dtype=torch.float32, requires_grad=True)
    return x, weight, dy


def test_rms_norm_forward_matches_pytorch_on_strided_tail() -> None:
    """Validate FP32 forward on non-power-of-two, row-strided input."""
    torch.manual_seed(0)
    x, weight, _ = _strided_problem()
    output = rms_norm(x, weight)
    reference = F.rms_norm(x, (x.shape[-1],), weight, _EPS)
    assert_relative_frobenius(output, reference, max_relative_error=1e-4)


def test_rms_norm_gradients_match_pytorch() -> None:
    """Validate ``dx`` and ``dweight`` against autograd."""
    torch.manual_seed(1)
    x, weight, dy = _strided_problem(rows=67, columns=1000)

    reference = F.rms_norm(x, (x.shape[-1],), weight, _EPS)
    dx_ref, dw_ref = torch.autograd.grad(
        reference,
        (x, weight),
        grad_outputs=dy,
    )

    x_tri = x.detach().requires_grad_(True)
    weight_tri = weight.detach().requires_grad_(True)
    output = rms_norm(x_tri, weight_tri)
    dx_tri, dw_tri = torch.autograd.grad(
        output,
        (x_tri, weight_tri),
        grad_outputs=dy,
    )

    assert_relative_frobenius(dx_tri, dx_ref, max_relative_error=1e-2)
    assert_relative_frobenius(dw_tri, dw_ref, max_relative_error=1e-2)


def test_rms_norm_gradients_match_under_strided_tail_contention() -> None:
    """Combine a non-power-of-two width with multiple updates per default lock."""
    torch.manual_seed(2)
    x, weight, dy = _strided_problem(rows=1025, columns=1000)

    reference = F.rms_norm(x, (x.shape[-1],), weight, _EPS)
    dx_ref, dw_ref = torch.autograd.grad(
        reference,
        (x, weight),
        grad_outputs=dy,
    )

    x_tri = x.detach().requires_grad_(True)
    weight_tri = weight.detach().requires_grad_(True)
    output = rms_norm(x_tri, weight_tri)
    dx_tri, dw_tri = torch.autograd.grad(
        output,
        (x_tri, weight_tri),
        grad_outputs=dy,
    )

    assert_relative_frobenius(dx_tri, dx_ref, max_relative_error=1e-2)
    assert_relative_frobenius(dw_tri, dw_ref, max_relative_error=1e-2)


def test_rms_norm_single_lock_count_and_gradient_under_contention() -> None:
    """Serialize 1,025 rows through one lock and verify its count state."""
    torch.manual_seed(3)
    rows, columns, runs = 1025, 1000, 10
    x, weight, dy = _strided_problem(rows=rows, columns=columns)
    output = torch.empty_like(x)
    rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
    _launch_forward(x, weight, output, rstd, eps=_EPS, store_stats=True)
    buffers = _make_backward_buffers(x, weight, group_size=1)
    reference_dw = torch.sum(dy * x * rstd[:, None], dim=0)

    for _ in range(runs):
        buffers.locks.zero_()
        _launch_backward_stage1(dy, x, weight, rstd, buffers)
        torch.cuda.synchronize()

        lock_state = buffers.locks.cpu()
        assert lock_state.tolist() == [0, 1]

        _launch_backward_stage2(buffers, columns)
        torch.cuda.synchronize()
        assert_relative_frobenius(
            buffers.dw,
            reference_dw,
            max_relative_error=1e-2,
        )


def test_rms_norm_backward_rejects_deterministic_mode() -> None:
    """Honor PyTorch's deterministic-algorithm contract."""
    previous_mode = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        x, weight, dy = _strided_problem(rows=8, columns=16)
        output = rms_norm(x, weight)
        with pytest.raises(RuntimeError, match="not deterministic"):
            torch.autograd.grad(output, (x, weight), grad_outputs=dy)
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)


def test_rms_norm_backward_warns_in_deterministic_warn_only_mode() -> None:
    """Follow PyTorch's warn-only deterministic policy without hiding the risk."""
    previous_mode = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        x, weight, dy = _strided_problem(rows=8, columns=16)
        output = rms_norm(x, weight)
        with pytest.warns(RuntimeWarning, match="not deterministic"):
            torch.autograd.grad(output, (x, weight), grad_outputs=dy)
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)
