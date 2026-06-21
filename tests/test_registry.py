"""Unit tests for registry invariants."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

import tklab.kernels  # noqa: F401
from tklab.registry import REGISTRY, KernelSpec


def _identity(x: torch.Tensor) -> torch.Tensor:
    """Return an input unchanged for lightweight contract tests."""
    return x


def _launch(
    args: tuple[torch.Tensor, ...],
    output: torch.Tensor,
) -> None:
    """Copy the first argument into a supplied output."""
    output.copy_(args[0])


def _make_inputs(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Create one simple tensor."""
    return (torch.zeros(size, device=device, dtype=dtype),)


def _make_output(args: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Allocate an output matching the first argument."""
    return torch.empty_like(args[0])


def _memory_spec(**overrides: object) -> KernelSpec:
    """Build a minimal valid memory-bound specification."""
    values: dict[str, object] = {
        "name": "example_kernel",
        "description": "Example kernel used by unit tests.",
        "triton_fn": _identity,
        "launch_fn": _launch,
        "ref_fn": _identity,
        "make_inputs": _make_inputs,
        "make_output": _make_output,
        "sizes": (1, 2),
        "bound": "memory",
        "bytes_moved": lambda size, dtype: size * torch.empty((), dtype=dtype).element_size(),
    }
    values.update(overrides)
    return KernelSpec(**cast(dict[str, Any], values))


def test_registry_contains_expected_kernels() -> None:
    """Ensure importing the package registers every built-in kernel."""
    assert set(REGISTRY) == {
        "attention_causal",
        "attention_noncausal",
        "vector_add",
        "fused_softmax",
        "gelu_forward",
        "layer_norm_forward",
        "relu_forward",
        "residual_rms_norm_forward",
        "rms_norm_forward",
        "rope_forward",
        "silu_forward",
        "swiglu_forward",
        "tanh_forward",
        "matmul_fp32acc",
        "matmul_fp16acc",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": ""}, "name"),
        ({"description": " "}, "description"),
        ({"sizes": ()}, "sizes"),
        ({"sizes": (1, 1)}, "duplicates"),
        ({"sizes": (True, 2)}, "positive integers"),
        ({"dtypes": (torch.float16, torch.float16)}, "dtypes"),
        ({"bytes_moved": None}, "bytes_moved"),
        ({"compute_mode": "fp16_fp32acc"}, "compute metadata"),
        ({"bound": "other", "bytes_moved": None}, "unsupported bound"),
        (
            {
                "bound": "compute",
                "bytes_moved": None,
                "flops": lambda size, dtype: size,
                "compute_mode": "other",
            },
            "unsupported compute mode",
        ),
    ],
)
def test_invalid_memory_specs_are_rejected(
    overrides: dict[str, object],
    message: str,
) -> None:
    """Reject malformed or internally inconsistent specifications."""
    with pytest.raises(ValueError, match=message):
        _memory_spec(**overrides)


def test_validation_sizes_default_to_benchmark_sizes() -> None:
    """Use benchmark sizes when no reduced correctness set is supplied."""
    spec = _memory_spec()
    assert tuple(spec.validation_sizes()) == (1, 2)


def test_spec_normalizes_mutable_sequences_to_tuples() -> None:
    """Keep a frozen specification immutable even when callers pass lists."""
    spec = _memory_spec(sizes=[1, 2], dtypes=[torch.float16])
    assert spec.sizes == (1, 2)
    assert spec.dtypes == (torch.float16,)
