"""Real-GPU gradient tests for Flash Attention backward."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from tklab.harness.tolerances import assert_relative_frobenius
from tklab.kernels.flash_attention import attention_causal, attention_noncausal

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required"),
]

Kernel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _inputs(sequence_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create contiguous FP16 leaves and a strided upstream gradient."""
    shape = (1, 16, sequence_length, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    do = torch.randn(shape, device="cuda", dtype=torch.float16)
    return q, k, v, do


def _manual_fp32_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    *,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the attention gradient algebra directly in FP32."""
    q_float, k_float, v_float, do_float = (tensor.detach().float() for tensor in (q, k, v, do))
    scale = q.shape[-1] ** -0.5
    scores = (q_float @ k_float.transpose(-2, -1)) * scale
    if causal:
        sequence_length = q.shape[-2]
        upper_triangle = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=q.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(upper_triangle, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    output = probabilities @ v_float
    dv = probabilities.transpose(-2, -1) @ do_float
    dprobabilities = do_float @ v_float.transpose(-2, -1)
    delta = torch.sum(do_float * output, dim=-1, keepdim=True)
    dscores = probabilities * (dprobabilities - delta)
    dq = (dscores @ k_float) * scale
    dk = (dscores.transpose(-2, -1) @ q_float) * scale
    return dq, dk, dv


@pytest.mark.parametrize(
    ("causal", "kernel"),
    [(False, attention_noncausal), (True, attention_causal)],
)
@pytest.mark.parametrize("sequence_length", [128, 1000])
def test_flash_attention_backward_matches_sdpa(
    causal: bool,
    kernel: Kernel,
    sequence_length: int,
) -> None:
    """Validate dQ, dK, and dV against SDPA autograd on full and partial tiles."""
    torch.manual_seed(41 + sequence_length + int(causal))
    q, k, v, do = _inputs(sequence_length)

    reference = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(reference, (q, k, v), grad_outputs=do)

    output = kernel(q, k, v)
    dq_tri, dk_tri, dv_tri = torch.autograd.grad(output, (q, k, v), grad_outputs=do)

    for triton_grad, reference_grad in (
        (dq_tri, dq_ref),
        (dk_tri, dk_ref),
        (dv_tri, dv_ref),
    ):
        assert torch.isfinite(triton_grad).all()
        assert_relative_frobenius(triton_grad, reference_grad, max_relative_error=3e-2)


@pytest.mark.parametrize(
    ("causal", "kernel"),
    [(False, attention_noncausal), (True, attention_causal)],
)
def test_flash_attention_backward_matches_manual_fp32_algebra(
    causal: bool,
    kernel: Kernel,
) -> None:
    """Cross-check dQ/dK/dV without relying on the SDPA backward implementation."""
    torch.manual_seed(79 + int(causal))
    q, k, v, do = _inputs(128)
    expected = _manual_fp32_backward(q, k, v, do, causal=causal)
    output = kernel(q, k, v)
    actual = torch.autograd.grad(output, (q, k, v), grad_outputs=do)
    for triton_grad, reference_grad in zip(actual, expected, strict=True):
        assert_relative_frobenius(triton_grad, reference_grad, max_relative_error=1e-3)
