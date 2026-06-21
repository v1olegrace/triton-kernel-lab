"""Real-GPU tests for fused residual addition and RMSNorm."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tklab.harness.tolerances import assert_relative_frobenius
from tklab.kernels.residual_rms_norm import (
    _launch_forward,
    residual_rms_norm,
)
from tklab.kernels.rms_norm import (
    _EPS,
    _launch_backward_stage1,
    _launch_backward_stage2,
    _make_backward_buffers,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required"),
]


def _strided_problem(
    *,
    rows: int = 37,
    columns: int = 1000,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Create independent row-strided leaves and output gradients."""
    x_storage = torch.randn(rows * 2, columns, device="cuda", dtype=torch.float32)
    residual_storage = torch.randn(
        rows * 3,
        columns,
        device="cuda",
        dtype=torch.float32,
    )
    dsum_storage = torch.randn(
        rows * 4,
        columns,
        device="cuda",
        dtype=torch.float32,
    )
    doutput_storage = torch.randn(
        rows * 5,
        columns,
        device="cuda",
        dtype=torch.float32,
    )
    x = x_storage[::2].detach().requires_grad_(True)
    residual = residual_storage[::3].detach().requires_grad_(True)
    dsum = dsum_storage[::4]
    doutput = doutput_storage[::5]
    weight = torch.randn(columns, device="cuda", dtype=torch.float32, requires_grad=True)
    return x, residual, weight, dsum, doutput


def test_residual_rms_norm_forward_matches_pytorch() -> None:
    """Validate both outputs on independent row strides and a masked tail."""
    torch.manual_seed(20)
    x, residual, weight, _, _ = _strided_problem()
    residual_sum, output = residual_rms_norm(x, residual, weight)
    reference_sum = x + residual
    reference_output = F.rms_norm(
        reference_sum,
        (reference_sum.shape[-1],),
        weight,
        _EPS,
    )
    assert_relative_frobenius(
        residual_sum,
        reference_sum,
        max_relative_error=1e-7,
    )
    assert_relative_frobenius(
        output,
        reference_output,
        max_relative_error=1e-4,
    )


def test_residual_rms_norm_fp16_normalizes_materialized_sum() -> None:
    """Normalize the same FP16-rounded residual sum returned to the caller."""
    torch.manual_seed(24)
    rows, columns = 37, 1000
    x = torch.randn(rows, columns, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(columns, device="cuda", dtype=torch.float16)
    residual_sum, output = residual_rms_norm(x, residual, weight)
    reference_sum = x + residual
    reference_output = F.rms_norm(
        reference_sum,
        (columns,),
        weight,
        _EPS,
    )
    torch.testing.assert_close(residual_sum, reference_sum, rtol=0, atol=0)
    assert_relative_frobenius(
        output,
        reference_output,
        max_relative_error=1e-3,
    )


def test_residual_rms_norm_combined_output_gradients_match_pytorch() -> None:
    """Validate the direct residual path plus the RMSNorm backward path."""
    torch.manual_seed(21)
    x, residual, weight, dsum, doutput = _strided_problem(rows=67)

    reference_sum = x + residual
    reference_output = F.rms_norm(
        reference_sum,
        (reference_sum.shape[-1],),
        weight,
        _EPS,
    )
    dx_ref, dresidual_ref, dw_ref = torch.autograd.grad(
        (reference_sum, reference_output),
        (x, residual, weight),
        grad_outputs=(dsum, doutput),
    )

    x_tri = x.detach().requires_grad_(True)
    residual_tri = residual.detach().requires_grad_(True)
    weight_tri = weight.detach().requires_grad_(True)
    residual_sum_tri, output_tri = residual_rms_norm(
        x_tri,
        residual_tri,
        weight_tri,
    )
    dx_tri, dresidual_tri, dw_tri = torch.autograd.grad(
        (residual_sum_tri, output_tri),
        (x_tri, residual_tri, weight_tri),
        grad_outputs=(dsum, doutput),
    )

    assert_relative_frobenius(dx_tri, dx_ref, max_relative_error=1e-2)
    assert_relative_frobenius(
        dresidual_tri,
        dresidual_ref,
        max_relative_error=1e-2,
    )
    assert_relative_frobenius(dw_tri, dw_ref, max_relative_error=1e-2)
    torch.testing.assert_close(dx_tri, dresidual_tri, rtol=0, atol=0)


def test_residual_rms_norm_direct_sum_gradient_matches_both_inputs() -> None:
    """Return the incoming sum gradient unchanged when norm output is unused."""
    torch.manual_seed(22)
    x, residual, weight, dsum, _ = _strided_problem(rows=17)
    residual_sum, _ = residual_rms_norm(x, residual, weight)
    dx, dresidual = torch.autograd.grad(
        residual_sum,
        (x, residual),
        grad_outputs=dsum,
    )
    torch.testing.assert_close(dx, dsum, rtol=0, atol=0)
    torch.testing.assert_close(dresidual, dsum, rtol=0, atol=0)


def test_residual_sum_only_is_deterministic() -> None:
    """Avoid the lock reduction when the normalized output is unused."""
    previous_mode = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        x, residual, weight, dsum, _ = _strided_problem(rows=17)
        residual_sum, _ = residual_rms_norm(x, residual, weight)
        dx, dresidual = torch.autograd.grad(
            residual_sum,
            (x, residual),
            grad_outputs=dsum,
        )
        torch.testing.assert_close(dx, dsum, rtol=0, atol=0)
        torch.testing.assert_close(dresidual, dsum, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)


def test_residual_rms_norm_single_lock_adds_incoming_gradient() -> None:
    """Stress 1,025 rows on one lock while adding the direct output gradient."""
    torch.manual_seed(23)
    rows, columns, runs = 1025, 1000, 10
    x, residual, weight, dsum, doutput = _strided_problem(
        rows=rows,
        columns=columns,
    )
    residual_sum = torch.empty_like(x)
    output = torch.empty_like(x)
    rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
    _launch_forward(
        x,
        residual,
        weight,
        residual_sum,
        output,
        rstd,
        eps=_EPS,
        store_residual=True,
        store_stats=True,
    )
    buffers = _make_backward_buffers(residual_sum, weight, group_size=1)
    x_hat = residual_sum * rstd[:, None]
    g = weight * doutput
    dx_reference = dsum + rstd[:, None] * (g - x_hat * torch.mean(g * x_hat, dim=1, keepdim=True))
    dw_reference = torch.sum(doutput * x_hat, dim=0)

    for _ in range(runs):
        buffers.locks.zero_()
        _launch_backward_stage1(
            doutput,
            residual_sum,
            weight,
            rstd,
            buffers,
            incoming_dx=dsum,
        )
        torch.cuda.synchronize()
        assert buffers.locks.cpu().tolist() == [0, 1]
        _launch_backward_stage2(buffers, columns)
        torch.cuda.synchronize()
        assert_relative_frobenius(
            buffers.dx,
            dx_reference,
            max_relative_error=1e-2,
        )
        assert_relative_frobenius(
            buffers.dw,
            dw_reference,
            max_relative_error=1e-2,
        )


def test_residual_rms_norm_backward_rejects_deterministic_mode() -> None:
    """Propagate the lock reduction's deterministic-algorithm contract."""
    previous_mode = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        x, residual, weight, dsum, doutput = _strided_problem(rows=8, columns=16)
        residual_sum, output = residual_rms_norm(x, residual, weight)
        with pytest.raises(RuntimeError, match="not deterministic"):
            torch.autograd.grad(
                (residual_sum, output),
                (x, residual, weight),
                grad_outputs=(dsum, doutput),
            )
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)
