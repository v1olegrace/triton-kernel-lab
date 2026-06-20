"""Flash Attention v2-style forward kernels with online softmax."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tklab.harness.tolerances import assert_relative_frobenius
from tklab.registry import (
    BenchmarkMetadataFn,
    BenchmarkScalar,
    CostFn,
    KernelSpec,
    LaunchFn,
    TensorArgs,
    TensorFn,
    register,
)

_BATCH = 1
_HEADS = 16
_HEAD_DIM = 64
_MAX_INT32_OFFSET = 2**31 - 1
_LOG2_E = tl.constexpr(1.4426950408889634)


def _configs() -> list[triton.Config]:
    """Return an Ada-compatible forward autotune space."""
    space = (
        # BLOCK_M, BLOCK_N, stages, warps
        (64, 32, 2, 4),
        (64, 64, 2, 4),
        (64, 64, 3, 4),
        (128, 32, 2, 4),
        (128, 64, 2, 4),
        (128, 64, 3, 8),
    )
    return [
        triton.Config(
            {"BLOCK_M": block_m, "BLOCK_N": block_n},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_m, block_n, num_stages, num_warps in space
    ]


@triton.jit  # type: ignore[untyped-decorator]
def _attention_inner(
    accumulator: tl.tensor,
    row_sum: tl.tensor,
    row_max: tl.tensor,
    q: tl.tensor,
    k_ptr: tl.tensor,
    v_ptr: tl.tensor,
    k_base: int,
    v_base: int,
    stride_kn: int,
    stride_kd: int,
    stride_vn: int,
    stride_vd: int,
    start_m: int,
    n_ctx: tl.constexpr,
    sm_scale_log2: float,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor, tl.tensor]:
    """Stream full or diagonal K/V blocks into online-softmax state."""
    offsets_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    offsets_d = tl.arange(0, HEAD_DIM)
    if STAGE == 1:
        low = 0
        high = start_m * BLOCK_M
    elif STAGE == 2:
        low = start_m * BLOCK_M
        high = (start_m + 1) * BLOCK_M
    else:
        low = 0
        high = n_ctx

    for start_n in tl.range(low, high, BLOCK_N):
        current_n = start_n + offsets_n
        key_mask = current_n < n_ctx
        k = tl.load(
            k_ptr + k_base + current_n[:, None] * stride_kn + offsets_d[None, :] * stride_kd,
            mask=key_mask[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)) * sm_scale_log2
        if STAGE == 2:
            causal_mask = offsets_m[:, None] >= current_n[None, :]
            scores = tl.where(causal_mask & key_mask[None, :], scores, -float("inf"))
        else:
            scores = tl.where(key_mask[None, :], scores, -float("inf"))

        new_max = tl.maximum(row_max, tl.max(scores, axis=1))
        probabilities = tl.math.exp2(scores - new_max[:, None])
        correction = tl.math.exp2(row_max - new_max)
        accumulator *= correction[:, None]
        v = tl.load(
            v_ptr + v_base + current_n[:, None] * stride_vn + offsets_d[None, :] * stride_vd,
            mask=key_mask[:, None],
            other=0.0,
        )
        accumulator = tl.dot(probabilities.to(tl.float16), v, accumulator)
        row_sum = row_sum * correction + tl.sum(probabilities, axis=1)
        row_max = new_max
    return accumulator, row_sum, row_max


@triton.autotune(  # type: ignore[untyped-decorator]
    configs=_configs(),
    key=["N_CTX", "CAUSAL"],
)
@triton.jit  # type: ignore[untyped-decorator]
def _flash_attention_forward_kernel(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    v_ptr: tl.tensor,
    output_ptr: tl.tensor,
    stride_qz: int,
    stride_qh: int,
    stride_qn: int,
    stride_qd: int,
    stride_kz: int,
    stride_kh: int,
    stride_kn: int,
    stride_kd: int,
    stride_vz: int,
    stride_vh: int,
    stride_vn: int,
    stride_vd: int,
    stride_oz: int,
    stride_oh: int,
    stride_on: int,
    stride_od: int,
    HEADS: int,
    sm_scale: float,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Compute one query block without materializing the score matrix."""
    tl.static_assert(HEAD_DIM == 64, "the current attention kernel requires head_dim=64")
    tl.static_assert(BLOCK_M % BLOCK_N == 0, "causal staging requires BLOCK_M % BLOCK_N == 0")
    start_m = tl.program_id(axis=0)
    head_batch = tl.program_id(axis=1)
    batch = head_batch // HEADS
    head = head_batch % HEADS
    offsets_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, HEAD_DIM)
    query_mask = offsets_m < N_CTX

    q_base = batch * stride_qz + head * stride_qh
    k_base = batch * stride_kz + head * stride_kh
    v_base = batch * stride_vz + head * stride_vh
    output_base = batch * stride_oz + head * stride_oh
    q = tl.load(
        q_ptr + q_base + offsets_m[:, None] * stride_qn + offsets_d[None, :] * stride_qd,
        mask=query_mask[:, None],
        other=0.0,
    )

    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    sm_scale_log2 = sm_scale * _LOG2_E

    if CAUSAL:
        accumulator, row_sum, row_max = _attention_inner(
            accumulator,
            row_sum,
            row_max,
            q,
            k_ptr,
            v_ptr,
            k_base,
            v_base,
            stride_kn,
            stride_kd,
            stride_vn,
            stride_vd,
            start_m,
            N_CTX,
            sm_scale_log2,
            BLOCK_M,
            BLOCK_N,
            HEAD_DIM,
            STAGE=1,
        )
        accumulator, row_sum, row_max = _attention_inner(
            accumulator,
            row_sum,
            row_max,
            q,
            k_ptr,
            v_ptr,
            k_base,
            v_base,
            stride_kn,
            stride_kd,
            stride_vn,
            stride_vd,
            start_m,
            N_CTX,
            sm_scale_log2,
            BLOCK_M,
            BLOCK_N,
            HEAD_DIM,
            STAGE=2,
        )
    else:
        accumulator, row_sum, row_max = _attention_inner(
            accumulator,
            row_sum,
            row_max,
            q,
            k_ptr,
            v_ptr,
            k_base,
            v_base,
            stride_kn,
            stride_kd,
            stride_vn,
            stride_vd,
            start_m,
            N_CTX,
            sm_scale_log2,
            BLOCK_M,
            BLOCK_N,
            HEAD_DIM,
            STAGE=3,
        )

    output = accumulator / row_sum[:, None]
    tl.store(
        output_ptr + output_base + offsets_m[:, None] * stride_on + offsets_d[None, :] * stride_od,
        output,
        mask=query_mask[:, None],
    )


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """Validate the dense self-attention forward contract."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("attention expects Q, K, and V shaped (batch, heads, sequence, dim)")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q, K, and V must have identical shapes")
    if 0 in q.shape:
        raise ValueError("attention does not support empty dimensions")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K, and V must be on the same device")
    if q.device.type != "cuda":
        raise ValueError("attention requires CUDA tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q, K, and V must have the same dtype")
    if q.dtype != torch.float16:
        raise ValueError("attention currently supports float16 inputs only")
    if q.shape[-1] != _HEAD_DIM:
        raise ValueError(f"attention currently requires head_dim={_HEAD_DIM}")
    for name, tensor in (("Q", q), ("K", k), ("V", v)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        max_relative_offset = sum(
            (dimension - 1) * stride
            for dimension, stride in zip(tensor.shape, tensor.stride(), strict=True)
        )
        if max_relative_offset > _MAX_INT32_OFFSET:
            raise ValueError(f"{name} strides exceed the signed int32 offset range")


def _make_output(args: TensorArgs) -> torch.Tensor:
    """Allocate the dense attention output."""
    return torch.empty_like(args[0])


def _launch_factory(*, causal: bool) -> LaunchFn:
    """Create an allocation-free launcher for one masking mode."""

    def launch(args: TensorArgs, output: torch.Tensor) -> None:
        """Launch online-softmax attention into a preallocated tensor."""
        q, k, v = args
        _validate(q, k, v)
        if output.shape != q.shape or output.device != q.device or output.dtype != q.dtype:
            raise ValueError("attention output metadata must match Q")
        if not output.is_contiguous():
            raise ValueError("attention output must be contiguous")
        batch, heads, n_ctx, head_dim = q.shape

        def grid(metadata: dict[str, Any]) -> tuple[int, int]:
            return triton.cdiv(n_ctx, metadata["BLOCK_M"]), batch * heads

        _flash_attention_forward_kernel[grid](
            q,
            k,
            v,
            output,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *output.stride(),
            heads,
            1.0 / math.sqrt(head_dim),
            N_CTX=n_ctx,
            HEAD_DIM=head_dim,
            CAUSAL=causal,
        )

    return launch


_launch_noncausal = _launch_factory(causal=False)
_launch_causal = _launch_factory(causal=True)


def attention_noncausal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute dense non-causal scaled dot-product attention."""
    output = _make_output((q, k, v))
    _launch_noncausal((q, k, v), output)
    return output


def attention_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute dense causal scaled dot-product attention."""
    output = _make_output((q, k, v))
    _launch_causal((q, k, v), output)
    return output


def _reference_factory(*, causal: bool) -> TensorFn:
    """Create a PyTorch SDPA reference for one masking mode."""

    def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Call PyTorch scaled-dot-product attention."""
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    return reference


_reference_noncausal = _reference_factory(causal=False)
_reference_causal = _reference_factory(causal=True)


def _make_inputs(
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TensorArgs:
    """Create contiguous self-attention inputs."""
    shape = (_BATCH, _HEADS, sequence_length, _HEAD_DIM)
    return (
        torch.randn(shape, device=device, dtype=dtype),
        torch.randn(shape, device=device, dtype=dtype),
        torch.randn(shape, device=device, dtype=dtype),
    )


def _make_adversarial(device: torch.device) -> TensorArgs:
    """Create a non-multiple sequence length for query/key tile tails."""
    return _make_inputs(1000, device, torch.float16)


def _assert_attention(output: torch.Tensor, reference: torch.Tensor) -> None:
    """Validate attention through aggregate FP32-relative error."""
    assert_relative_frobenius(output, reference, max_relative_error=2e-2)


def _flops_factory(*, causal: bool) -> CostFn:
    """Create the two-matmul attention FLOP model."""

    def flops(sequence_length: int, _dtype: torch.dtype) -> int:
        """Count QK-transpose and probability-value multiply-adds."""
        attended_pairs = (
            sequence_length * (sequence_length + 1) // 2 if causal else sequence_length**2
        )
        return 4 * _BATCH * _HEADS * attended_pairs * _HEAD_DIM

    return flops


def _metadata_factory(*, causal: bool) -> BenchmarkMetadataFn:
    """Serialize the winning attention tile configuration."""

    def metadata(_size: int, _dtype: torch.dtype) -> dict[str, BenchmarkScalar]:
        """Return the latest autotune winner and fixed problem dimensions."""
        config = _flash_attention_forward_kernel.best_config
        return {
            "block_m": int(config.kwargs["BLOCK_M"]),
            "block_n": int(config.kwargs["BLOCK_N"]),
            "num_warps": int(config.num_warps),
            "num_stages": int(config.num_stages),
            "heads": _HEADS,
            "head_dim": _HEAD_DIM,
            "causal": causal,
        }

    return metadata


_SIZES = (512, 1024, 2048, 4096, 8192)

ATTENTION_NONCAUSAL = register(
    KernelSpec(
        name="attention_noncausal",
        description="Flash Attention forward with online softmax and no mask.",
        triton_fn=attention_noncausal,
        launch_fn=_launch_noncausal,
        ref_fn=_reference_noncausal,
        make_inputs=_make_inputs,
        make_output=_make_output,
        make_adversarial=_make_adversarial,
        sizes=_SIZES,
        correctness_sizes=(128, 512),
        bound="compute",
        flops=_flops_factory(causal=False),
        dtypes=(torch.float16,),
        assert_fn=_assert_attention,
        benchmark_metadata=_metadata_factory(causal=False),
        compute_mode="attention_fp16_fp32acc",
        supports_interpreter=False,
    )
)

ATTENTION_CAUSAL = register(
    KernelSpec(
        name="attention_causal",
        description="Flash Attention forward with two-stage causal masking.",
        triton_fn=attention_causal,
        launch_fn=_launch_causal,
        ref_fn=_reference_causal,
        make_inputs=_make_inputs,
        make_output=_make_output,
        make_adversarial=_make_adversarial,
        sizes=_SIZES,
        correctness_sizes=(128, 512),
        bound="compute",
        flops=_flops_factory(causal=True),
        dtypes=(torch.float16,),
        assert_fn=_assert_attention,
        benchmark_metadata=_metadata_factory(causal=True),
        compute_mode="attention_fp16_fp32acc",
        supports_interpreter=False,
    )
)
