"""Shared sizing helpers for the fused normalization kernels.

LayerNorm, RMSNorm, and the fused residual RMSNorm share the same epsilon
policy, single-pass feature-row budget, block/warp selection, two-stage
reduction tiling, and lock-group heuristic. Centralizing those invariants here
prevents the three implementations from drifting.
"""

from __future__ import annotations

import math

import triton

# Each normalized feature row is processed in one program, so the row must fit
# the single-pass on-chip budget.
MAX_FEATURE_BYTES = 65_536
EPS = 1e-5

# Tile shape for the stage-two parameter-gradient reduction.
STAGE2_BLOCK_M = 32
STAGE2_BLOCK_N = 128


def validate_epsilon(eps: float) -> None:
    """Require a finite, strictly positive normalization epsilon.

    A zero epsilon makes an all-zero row singular, while a negative or
    non-finite epsilon can produce invalid reciprocal standard deviations.

    Args:
        eps: Stabilizer added to the variance or mean square.

    Raises:
        ValueError: If ``eps`` is not finite and strictly positive.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")


def block_size_and_warps(columns: int, element_size: int) -> tuple[int, int]:
    """Return a power-of-two row block and its warp count.

    Args:
        columns: Feature-dimension width.
        element_size: Bytes per element of the input dtype.

    Returns:
        The power-of-two ``BLOCK_SIZE`` covering ``columns`` and a warp count
        scaled to that block.

    Raises:
        ValueError: If ``columns`` exceeds the single-pass feature budget.
    """
    max_elements = MAX_FEATURE_BYTES // element_size
    block_size = min(max_elements, triton.next_power_of_2(columns))
    if columns > block_size:
        raise ValueError("feature dimension exceeds the fused normalization limit")
    num_warps = min(max(block_size // 256, 1), 8)
    return block_size, num_warps


def group_size_m(columns: int) -> int:
    """Select the number of independent parameter-gradient lock groups.

    Wider feature rows touch more bytes per program, so fewer lock groups keep
    the stage-two reduction balanced against stage-one contention.
    """
    if columns <= 1024:
        return 256
    if columns <= 4096:
        return 128
    if columns <= 8192:
        return 96
    return 64
