"""GPU and Triton-interpreter correctness tests for registered kernels."""

from __future__ import annotations

import os

import pytest
import torch

import tklab.kernels  # noqa: F401
from tklab.harness.tolerances import assert_close
from tklab.registry import REGISTRY, KernelSpec


def _assert_spec_output(
    spec: KernelSpec,
    output: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    """Apply a kernel-specific or dtype-aware correctness policy."""
    if spec.assert_fn is None:
        assert_close(output, reference, rtol=spec.rtol, atol=spec.atol)
    else:
        spec.assert_fn(output, reference)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")
@pytest.mark.parametrize("name", REGISTRY)
def test_matches_reference_on_gpu(name: str) -> None:
    """Compare every registered kernel against a float32 reference on GPU."""
    spec = REGISTRY[name]
    device = torch.device("cuda")
    torch.manual_seed(0)

    for dtype in spec.dtypes:
        for size in spec.validation_sizes():
            args = spec.make_inputs(size, device, dtype)
            output = spec.triton_fn(*args)
            reference = spec.ref_fn(*(argument.float() for argument in args))
            _assert_spec_output(spec, output, reference)

    if spec.make_adversarial is not None:
        args = spec.make_adversarial(device)
        output = spec.triton_fn(*args)
        reference = spec.ref_fn(*(argument.float() for argument in args))
        _assert_spec_output(spec, output, reference)


@pytest.mark.interpret
@pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="TRITON_INTERPRET=1 is required",
)
@pytest.mark.parametrize("name", REGISTRY)
def test_adversarial_case_in_interpreter(name: str) -> None:
    """Exercise each interpreter-compatible adversarial case on CPU."""
    spec = REGISTRY[name]
    if not spec.supports_interpreter:
        pytest.skip(f"{name}: interpreter is not supported")
    if spec.make_adversarial is None:
        pytest.skip(f"{name}: no adversarial case")

    torch.manual_seed(0)
    args = spec.make_adversarial(torch.device("cpu"))
    output = spec.triton_fn(*args)
    reference = spec.ref_fn(*(argument.float() for argument in args))
    _assert_spec_output(spec, output, reference)
