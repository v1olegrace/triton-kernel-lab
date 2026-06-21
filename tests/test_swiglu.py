"""Real-GPU correctness and gradient tests for SwiGLU."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tklab.harness.tolerances import assert_relative_frobenius
from tklab.kernels.swiglu import swiglu

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required"),
]


def _strided_problem(
    *,
    rows: int = 37,
    columns: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create independent row-strided inputs and upstream gradient."""
    value_storage = torch.randn(rows * 2, columns, device="cuda", dtype=torch.float32)
    gate_storage = torch.randn(rows * 3, columns, device="cuda", dtype=torch.float32)
    dy_storage = torch.randn(rows * 4, columns, device="cuda", dtype=torch.float32)
    value = value_storage[::2].detach().requires_grad_(True)
    gate = gate_storage[::3].detach().requires_grad_(True)
    return value, gate, dy_storage[::4]


def test_swiglu_forward_matches_pytorch_on_strided_tail() -> None:
    """Validate FP32 forward with row strides and a masked final block."""
    torch.manual_seed(30)
    value, gate, _ = _strided_problem()
    output = swiglu(value, gate)
    reference = value * F.silu(gate)
    assert_relative_frobenius(output, reference, max_relative_error=1e-6)


def test_swiglu_gradients_match_pytorch() -> None:
    """Validate both deterministic elementwise gradients."""
    torch.manual_seed(31)
    value, gate, dy = _strided_problem(rows=67)
    reference = value * F.silu(gate)
    dvalue_ref, dgate_ref = torch.autograd.grad(
        reference,
        (value, gate),
        grad_outputs=dy,
    )

    value_tri = value.detach().requires_grad_(True)
    gate_tri = gate.detach().requires_grad_(True)
    output = swiglu(value_tri, gate_tri)
    dvalue_tri, dgate_tri = torch.autograd.grad(
        output,
        (value_tri, gate_tri),
        grad_outputs=dy,
    )
    assert_relative_frobenius(
        dvalue_tri,
        dvalue_ref,
        max_relative_error=1e-6,
    )
    assert_relative_frobenius(
        dgate_tri,
        dgate_ref,
        max_relative_error=1e-6,
    )


def test_swiglu_extreme_gates_remain_finite() -> None:
    """Exercise both stable sigmoid branches far beyond exp overflow."""
    value = torch.randn(4, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    gate_values = torch.tensor(
        [-100.0, -80.0, -20.0, -1.0, 0.0, 1.0, 20.0, 100.0],
        device="cuda",
    )
    gate = gate_values.repeat(4, 1).requires_grad_(True)
    output = swiglu(value, gate)
    dvalue, dgate = torch.autograd.grad(
        output,
        (value, gate),
        grad_outputs=torch.ones_like(output),
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(dvalue).all()
    assert torch.isfinite(dgate).all()
    reference = value * F.silu(gate)
    assert_relative_frobenius(output, reference, max_relative_error=1e-6)


def test_swiglu_backward_is_deterministic() -> None:
    """Remain valid when PyTorch deterministic algorithms are required."""
    previous_mode = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        value, gate, dy = _strided_problem(rows=17, columns=128)
        output = swiglu(value, gate)
        torch.autograd.grad(output, (value, gate), grad_outputs=dy)
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)
