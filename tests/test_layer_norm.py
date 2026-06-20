"""Real-GPU forward and gradient tests for Triton LayerNorm."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tklab.harness.tolerances import assert_relative_frobenius
from tklab.kernels.layer_norm import layer_norm

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required"),
]


def _strided_problem(
    *,
    rows: int = 37,
    columns: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create FP32 leaves with row-strided input and upstream gradient."""
    x_storage = torch.randn(rows * 2, columns, device="cuda", dtype=torch.float32)
    dy_storage = torch.randn(rows * 3, columns, device="cuda", dtype=torch.float32)
    x = x_storage[::2].detach().requires_grad_(True)
    dy = dy_storage[::3]
    weight = torch.randn(columns, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(columns, device="cuda", dtype=torch.float32, requires_grad=True)
    return x, weight, bias, dy


def test_layer_norm_forward_matches_pytorch_on_strided_tail() -> None:
    """Validate FP32 forward on non-power-of-two, row-strided input."""
    torch.manual_seed(0)
    x, weight, bias, _ = _strided_problem()
    output = layer_norm(x, weight, bias)
    reference = F.layer_norm(x, (x.shape[-1],), weight, bias)
    assert_relative_frobenius(output, reference, max_relative_error=1e-4)


def test_layer_norm_gradients_match_pytorch() -> None:
    """Validate ``dx``, ``dweight``, and ``dbias`` against autograd."""
    torch.manual_seed(1)
    x, weight, bias, dy = _strided_problem(rows=67, columns=1000)

    reference = F.layer_norm(x, (x.shape[-1],), weight, bias)
    dx_ref, dw_ref, db_ref = torch.autograd.grad(
        reference,
        (x, weight, bias),
        grad_outputs=dy,
    )

    x_tri = x.detach().requires_grad_(True)
    weight_tri = weight.detach().requires_grad_(True)
    bias_tri = bias.detach().requires_grad_(True)
    output = layer_norm(x_tri, weight_tri, bias_tri)
    dx_tri, dw_tri, db_tri = torch.autograd.grad(
        output,
        (x_tri, weight_tri, bias_tri),
        grad_outputs=dy,
    )

    assert_relative_frobenius(dx_tri, dx_ref, max_relative_error=1e-2)
    assert_relative_frobenius(dw_tri, dw_ref, max_relative_error=1e-2)
    assert_relative_frobenius(db_tri, db_ref, max_relative_error=1e-2)
