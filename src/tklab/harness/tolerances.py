"""Numerical correctness policies shared by kernel tests."""

from __future__ import annotations

import torch

_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.6e-2, 1e-2),
    torch.float32: (1e-4, 1e-5),
}


def assert_close(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Assert elementwise agreement using dtype-aware defaults.

    Args:
        output: Kernel result.
        reference: Higher-precision reference result.
        rtol: Optional relative tolerance override.
        atol: Optional absolute tolerance override.

    Raises:
        ValueError: If the output dtype has no configured tolerance.
        AssertionError: If tensors contain non-finite values or differ beyond
            the selected tolerances.
    """
    if output.dtype not in _TOLERANCES:
        raise ValueError(f"no default tolerance configured for {output.dtype}")
    if not torch.isfinite(output).all():
        raise AssertionError("output contains non-finite values")
    default_rtol, default_atol = _TOLERANCES[output.dtype]
    torch.testing.assert_close(
        output,
        reference.to(output.dtype),
        rtol=default_rtol if rtol is None else rtol,
        atol=default_atol if atol is None else atol,
    )


def assert_relative_frobenius(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    max_relative_error: float,
) -> None:
    """Assert that relative Frobenius error stays below a threshold.

    This metric is appropriate for large reductions such as GEMM, where a few
    elementwise outliers can be expected even when aggregate error is small.

    Args:
        output: Kernel result.
        reference: Higher-precision reference result.
        max_relative_error: Strict upper bound on ``||output-reference|| /
            ||reference||``.

    Raises:
        ValueError: If ``max_relative_error`` is not positive.
        AssertionError: If output is non-finite or exceeds the threshold.
    """
    if max_relative_error <= 0:
        raise ValueError("max_relative_error must be positive")
    output_float = output.float()
    reference_float = reference.float()
    if not torch.isfinite(output_float).all():
        raise AssertionError("output contains non-finite values")

    error_norm = torch.linalg.vector_norm(output_float - reference_float)
    reference_norm = torch.linalg.vector_norm(reference_float)
    relative_error = error_norm if reference_norm.item() == 0.0 else error_norm / reference_norm
    if relative_error.item() >= max_relative_error:
        raise AssertionError(
            f"relative Frobenius error {relative_error.item():.6g} exceeds {max_relative_error:.6g}"
        )
