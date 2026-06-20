"""Real-GPU correctness tests for Flash Attention forward."""

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
