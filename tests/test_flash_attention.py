"""Real-GPU correctness tests for Flash Attention forward."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from tklab.harness.tolerances import assert_relative_frobenius
from tklab.kernels.flash_attention import (
    _launch_forward,
    attention_causal,
    attention_noncausal,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required"),
]


@pytest.mark.parametrize(
    ("causal", "kernel"),
    [(False, attention_noncausal), (True, attention_causal)],
)
@pytest.mark.parametrize("sequence_length", [128, 1000])
def test_flash_attention_matches_sdpa(
    causal: bool,
    kernel: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    sequence_length: int,
) -> None:
    """Validate regular and partial query/key tiles against SDPA."""
    torch.manual_seed(23 + sequence_length + int(causal))
    shape = (1, 16, sequence_length, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)

    output = kernel(q, k, v)
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    assert torch.isfinite(output).all()
    assert_relative_frobenius(output, reference, max_relative_error=2e-2)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_saved_statistic_is_base2_logsumexp(causal: bool) -> None:
    """Lock the saved statistic to normalized LSE, never the running maximum."""
    torch.manual_seed(91 + int(causal))
    shape = (1, 2, 129, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)
    output = torch.empty_like(q)
    softmax_lse = torch.empty(shape[:-1], device="cuda", dtype=torch.float32)

    _launch_forward(
        q,
        k,
        v,
        output,
        softmax_lse,
        causal=causal,
        store_lse=True,
    )

    scores = (q.float() @ k.float().transpose(-2, -1)) * (shape[-1] ** -0.5)
    if causal:
        upper_triangle = torch.triu(
            torch.ones(
                shape[-2],
                shape[-2],
                device=q.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(upper_triangle, -torch.inf)
    expected = torch.logsumexp(scores, dim=-1) * math.log2(math.e)

    assert torch.isfinite(softmax_lse).all()
    assert_relative_frobenius(softmax_lse, expected, max_relative_error=1e-3)
