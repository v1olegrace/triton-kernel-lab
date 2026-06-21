"""Focused real-GPU workloads for Compute Sanitizer tools."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from tklab.kernels.flash_attention import attention_causal, attention_noncausal
from tklab.kernels.fused_softmax import softmax
from tklab.kernels.layer_norm import layer_norm
from tklab.kernels.matmul import matmul_fp16acc, matmul_fp32acc
from tklab.kernels.residual_rms_norm import residual_rms_norm
from tklab.kernels.rms_norm import rms_norm
from tklab.kernels.rope import rope
from tklab.kernels.swiglu import swiglu
from tklab.kernels.vector_add import vector_add

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required"),
]


def test_sanitizer_vector_add_strided_tail() -> None:
    """Exercise non-unit strides and a masked vector tail."""
    left = torch.randn(2018, device="cuda", dtype=torch.float32)[::2]
    right = torch.randn(3027, device="cuda", dtype=torch.float32)[::3]
    vector_add(left, right)
    torch.cuda.synchronize()


def test_sanitizer_softmax_row_stride_and_tail() -> None:
    """Exercise row stride and a non-power-of-two reduction width."""
    values = torch.randn(32, 1000, device="cuda", dtype=torch.float32)[::2]
    softmax(values)
    torch.cuda.synchronize()


@pytest.mark.parametrize("kernel", [matmul_fp32acc, matmul_fp16acc])
def test_sanitizer_matmul_contiguous_and_strided(
    kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    """Exercise contiguous specialization plus M/N/K strided tails."""
    left = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    right = torch.randn_like(left)
    kernel(left, right)

    left_storage = torch.randn(129, 386, device="cuda", dtype=torch.float16)
    right_storage = torch.randn(386, 257, device="cuda", dtype=torch.float16)
    kernel(left_storage[:, ::2], right_storage[::2, :])
    torch.cuda.synchronize()


def test_sanitizer_layer_norm_forward_backward_strided_tail() -> None:
    """Exercise a row-strided tail with multiple updates per lock slot."""
    rows, columns = 513, 1000
    x_storage = torch.randn(rows * 2, columns, device="cuda", dtype=torch.float32)
    dy_storage = torch.randn(rows * 3, columns, device="cuda", dtype=torch.float32)
    x = x_storage[::2].detach().requires_grad_(True)
    dy = dy_storage[::3]
    weight = torch.randn(columns, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(columns, device="cuda", dtype=torch.float32, requires_grad=True)
    output = layer_norm(x, weight, bias)
    torch.autograd.grad(output, (x, weight, bias), grad_outputs=dy)
    torch.cuda.synchronize()


def test_sanitizer_rms_norm_forward_backward_strided_tail() -> None:
    """Exercise RMSNorm's single-buffer lock reduction and masked tail."""
    rows, columns = 513, 1000
    x_storage = torch.randn(rows * 2, columns, device="cuda", dtype=torch.float32)
    dy_storage = torch.randn(rows * 3, columns, device="cuda", dtype=torch.float32)
    x = x_storage[::2].detach().requires_grad_(True)
    dy = dy_storage[::3]
    weight = torch.randn(columns, device="cuda", dtype=torch.float32, requires_grad=True)
    output = rms_norm(x, weight)
    torch.autograd.grad(output, (x, weight), grad_outputs=dy)
    torch.cuda.synchronize()


def test_sanitizer_residual_rms_norm_forward_backward_strided_tail() -> None:
    """Exercise both fused outputs and their shared backward gradient."""
    rows, columns = 513, 1000
    x_storage = torch.randn(rows * 2, columns, device="cuda", dtype=torch.float32)
    residual_storage = torch.randn(
        rows * 3,
        columns,
        device="cuda",
        dtype=torch.float32,
    )
    dsum_storage = torch.randn(
        rows * 4,
        columns,
        device="cuda",
        dtype=torch.float32,
    )
    doutput_storage = torch.randn(
        rows * 5,
        columns,
        device="cuda",
        dtype=torch.float32,
    )
    x = x_storage[::2].detach().requires_grad_(True)
    residual = residual_storage[::3].detach().requires_grad_(True)
    dsum = dsum_storage[::4]
    doutput = doutput_storage[::5]
    weight = torch.randn(columns, device="cuda", dtype=torch.float32, requires_grad=True)
    residual_sum, output = residual_rms_norm(x, residual, weight)
    torch.autograd.grad(
        (residual_sum, output),
        (x, residual, weight),
        grad_outputs=(dsum, doutput),
    )
    torch.cuda.synchronize()


def test_sanitizer_swiglu_forward_backward_strided_tail() -> None:
    """Exercise stable nonlinear math and independent row strides."""
    rows, columns = 37, 1000
    value_storage = torch.randn(rows * 2, columns, device="cuda", dtype=torch.float32)
    gate_storage = torch.randn(rows * 3, columns, device="cuda", dtype=torch.float32)
    dy_storage = torch.randn(rows * 4, columns, device="cuda", dtype=torch.float32)
    value = value_storage[::2].detach().requires_grad_(True)
    gate = gate_storage[::3].detach().requires_grad_(True)
    dy = dy_storage[::4]
    output = swiglu(value, gate)
    torch.autograd.grad(output, (value, gate), grad_outputs=dy)
    torch.cuda.synchronize()


def test_sanitizer_rope_forward_backward_strided_tail() -> None:
    """Exercise rotate-half masks, row strides, and inverse rotation."""
    rows, columns = 37, 1000
    x_storage = torch.randn(rows * 2, columns, device="cuda", dtype=torch.float32)
    dy_storage = torch.randn(rows * 3, columns, device="cuda", dtype=torch.float32)
    angles_storage = torch.randn(
        rows * 4,
        columns // 2,
        device="cuda",
        dtype=torch.float32,
    )
    x = x_storage[::2].detach().requires_grad_(True)
    dy = dy_storage[::3]
    angles = angles_storage[::4]
    cos = torch.cos(angles).detach().requires_grad_(True)
    sin = torch.sin(angles).detach().requires_grad_(True)
    output = rope(x, cos, sin)
    torch.autograd.grad(output, (x, cos, sin), grad_outputs=dy)
    torch.cuda.synchronize()


@pytest.mark.parametrize("kernel", [attention_noncausal, attention_causal])
def test_sanitizer_flash_attention_partial_tiles(
    kernel: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    """Exercise query/key tails and the partial causal diagonal block."""
    shape = (1, 16, 1000, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)
    output = kernel(q, k, v)
    if not torch.isfinite(output).all():
        raise AssertionError("attention sanitizer workload produced non-finite output")
    torch.cuda.synchronize()
