"""Shared numerically stable device math for elementwise kernels."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit  # type: ignore[untyped-decorator]
def stable_sigmoid(values: tl.tensor) -> tl.tensor:
    """Return sigmoid without overflowing either exponential branch."""
    exp_negative_abs = tl.exp(-tl.abs(values))
    denominator = 1.0 + exp_negative_abs
    return tl.where(
        values >= 0.0,
        1.0 / denominator,
        exp_negative_abs / denominator,
    )
