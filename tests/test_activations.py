"""Real-GPU correctness and gradient tests for elementwise activations."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from tklab.harness.tolerances import assert_relative_frobenius
from tklab.kernels.activations import gelu, relu, silu, tanh

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required"),
]

TensorFn = Callable[[torch.Tensor], torch.Tensor]

_ACTIVATIONS: tuple[tuple[str, TensorFn, TensorFn], ...] = (
    ("relu", relu, torch.relu),
    ("gelu", gelu, F.gelu),
    ("silu", silu, F.silu),
    ("tanh", tanh, torch.tanh),
)
_SEEDS = {
    "relu": 101,
    "gelu": 102,
    "silu": 103,
    "tanh": 104,
}


def _strided_problem(
    *,
    rows: int = 37,
    columns: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a row-strided input and an independently strided upstream grad."""
    x_storage = torch.randn(rows * 2, columns, device="cuda", dtype=torch.float32)
    dy_storage = torch.randn(rows * 3, columns, device="cuda", dtype=torch.float32)
    return x_storage[::2], dy_storage[::3]


@pytest.mark.parametrize(("name", "triton_fn", "reference_fn"), _ACTIVATIONS)
def test_activation_forward_matches_pytorch_on_strided_tail(
    name: str,
    triton_fn: TensorFn,
    reference_fn: TensorFn,
) -> None:
    """Validate FP32 forward with row strides and a masked final block."""
    torch.manual_seed(_SEEDS[name])
    x, _ = _strided_problem()
    output = triton_fn(x)
    reference = reference_fn(x)
    assert_relative_frobenius(output, reference, max_relative_error=1e-6)


@pytest.mark.parametrize(("name", "triton_fn", "reference_fn"), _ACTIVATIONS)
def test_activation_gradient_matches_pytorch(
    name: str,
    triton_fn: TensorFn,
    reference_fn: TensorFn,
) -> None:
    """Validate the deterministic elementwise input gradient."""
    torch.manual_seed(_SEEDS[name])
    x_storage, dy = _strided_problem(rows=67)

    x_ref = x_storage.detach().requires_grad_(True)
    (dx_ref,) = torch.autograd.grad(reference_fn(x_ref), x_ref, grad_outputs=dy)

    x_tri = x_storage.detach().requires_grad_(True)
    (dx_tri,) = torch.autograd.grad(triton_fn(x_tri), x_tri, grad_outputs=dy)

    assert_relative_frobenius(dx_tri, dx_ref, max_relative_error=1e-5)


@pytest.mark.parametrize(("name", "triton_fn", "reference_fn"), _ACTIVATIONS)
def test_activation_extreme_inputs_remain_finite(
    name: str,
    triton_fn: TensorFn,
    reference_fn: TensorFn,
) -> None:
    """Exercise saturating inputs far beyond exp overflow."""
    extreme = torch.tensor(
        [-100.0, -80.0, -20.0, -1.0, 0.0, 1.0, 20.0, 100.0],
        device="cuda",
    )
    x = extreme.repeat(4, 1).requires_grad_(True)
    output = triton_fn(x)
    (dx,) = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output))
    assert torch.isfinite(output).all()
    assert torch.isfinite(dx).all()
    assert_relative_frobenius(output, reference_fn(x), max_relative_error=1e-5)


def test_activation_fp16_correctness_path() -> None:
    """Confirm the registered FP16 dtype path matches a float reference."""
    torch.manual_seed(7)
    x = torch.randn(128, 512, device="cuda", dtype=torch.float16)
    for triton_fn, reference_fn in (
        (relu, torch.relu),
        (gelu, F.gelu),
        (silu, F.silu),
        (tanh, torch.tanh),
    ):
        output = triton_fn(x)
        assert output.dtype == torch.float16
        reference = reference_fn(x.float())
        assert_relative_frobenius(output, reference, max_relative_error=1e-2)
