"""Real-GPU correctness and gradient tests for rotate-half RoPE."""

from __future__ import annotations

import pytest
import torch

from tklab.harness.tolerances import assert_relative_frobenius
from tklab.kernels.rope import rope

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required"),
]


def _reference(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply the repository's rotate-half convention in PyTorch."""
    x_first, x_second = x.chunk(2, dim=-1)
    return torch.cat(
        (
            x_first * cos - x_second * sin,
            x_second * cos + x_first * sin,
        ),
        dim=-1,
    )


def _strided_problem(
    *,
    rows: int = 37,
    columns: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create row-strided x, angle tables, and upstream gradient."""
    x_storage = torch.randn(rows * 2, columns, device="cuda", dtype=torch.float32)
    dy_storage = torch.randn(rows * 3, columns, device="cuda", dtype=torch.float32)
    angle_storage = torch.randn(
        rows * 4,
        columns // 2,
        device="cuda",
        dtype=torch.float32,
    )
    angles = angle_storage[::4]
    x = x_storage[::2].detach().requires_grad_(True)
    cos = torch.cos(angles).detach().requires_grad_(True)
    sin = torch.sin(angles).detach().requires_grad_(True)
    return x, cos, sin, dy_storage[::3]


def test_rope_forward_matches_rotate_half_reference() -> None:
    """Validate the chosen convention on row strides and a masked tail."""
    torch.manual_seed(40)
    x, cos, sin, _ = _strided_problem()
    output = rope(x, cos, sin)
    reference = _reference(x, cos, sin)
    assert_relative_frobenius(output, reference, max_relative_error=1e-7)


def test_rope_gradients_match_pytorch() -> None:
    """Validate x, cosine, and sine gradients."""
    torch.manual_seed(41)
    x, cos, sin, dy = _strided_problem(rows=67)
    reference = _reference(x, cos, sin)
    dx_ref, dcos_ref, dsin_ref = torch.autograd.grad(
        reference,
        (x, cos, sin),
        grad_outputs=dy,
    )

    x_tri = x.detach().requires_grad_(True)
    cos_tri = cos.detach().requires_grad_(True)
    sin_tri = sin.detach().requires_grad_(True)
    output = rope(x_tri, cos_tri, sin_tri)
    dx_tri, dcos_tri, dsin_tri = torch.autograd.grad(
        output,
        (x_tri, cos_tri, sin_tri),
        grad_outputs=dy,
    )
    assert_relative_frobenius(dx_tri, dx_ref, max_relative_error=1e-7)
    assert_relative_frobenius(dcos_tri, dcos_ref, max_relative_error=1e-7)
    assert_relative_frobenius(dsin_tri, dsin_ref, max_relative_error=1e-7)


def test_rope_x_backward_is_forward_with_negated_sine() -> None:
    """Confirm the orthogonal inverse-rotation identity used by backward."""
    torch.manual_seed(42)
    x, cos, sin, dy = _strided_problem(rows=17, columns=128)
    output = rope(x, cos, sin)
    (dx,) = torch.autograd.grad(output, (x,), grad_outputs=dy)
    inverse_reference = _reference(dy, cos, -sin)
    assert_relative_frobenius(
        dx,
        inverse_reference,
        max_relative_error=1e-7,
    )


def test_rope_preserves_row_norm_for_valid_angles() -> None:
    """Check the defining orthogonality invariant of rotary embeddings."""
    torch.manual_seed(43)
    x, cos, sin, _ = _strided_problem(rows=17, columns=128)
    output = rope(x, cos, sin)
    input_norm = torch.sum(x * x, dim=-1)
    output_norm = torch.sum(output * output, dim=-1)
    torch.testing.assert_close(output_norm, input_norm, rtol=1e-5, atol=1e-5)


def test_rope_backward_is_deterministic() -> None:
    """Remain valid under PyTorch deterministic mode."""
    previous_mode = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        x, cos, sin, dy = _strided_problem(rows=17, columns=128)
        output = rope(x, cos, sin)
        torch.autograd.grad(output, (x, cos, sin), grad_outputs=dy)
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)
