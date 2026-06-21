"""Flash Attention v2-style forward kernels with online softmax."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from tklab.harness.addressing import assert_int32_addressable
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
    softmax_lse_ptr: tl.tensor,
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
    STORE_LSE: tl.constexpr,
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
    if STORE_LSE:
        # Log-sum-exp in the base-2 domain the forward pass already works in:
        # row_max is the scaled per-row maximum, so
        # softmax_lse = row_max + log2(row_sum) lets the backward recompute
        # P = exp2(scaled_scores - softmax_lse) without a separate division.
        softmax_lse = row_max + tl.math.log2(row_sum)
        tl.store(
            softmax_lse_ptr + head_batch * N_CTX + offsets_m,
            softmax_lse,
            mask=query_mask,
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
        assert_int32_addressable(tensor, name=name)


def _make_output(args: TensorArgs) -> torch.Tensor:
    """Allocate the dense attention output."""
    return torch.empty_like(args[0])


def _launch_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    *,
    causal: bool,
    store_lse: bool,
) -> None:
    """Launch online-softmax attention into preallocated buffers.

    ``softmax_lse`` receives the per-query base-2 log-sum-exp when
    ``store_lse`` is set; otherwise it is an unused placeholder and the
    forward path is byte-for-byte identical to the inference-only launcher.
    """
    _validate(q, k, v)
    if output.shape != q.shape or output.device != q.device or output.dtype != q.dtype:
        raise ValueError("attention output metadata must match Q")
    if not output.is_contiguous():
        raise ValueError("attention output must be contiguous")
    batch, heads, n_ctx, head_dim = q.shape
    if store_lse:
        if (
            softmax_lse.shape != (batch, heads, n_ctx)
            or softmax_lse.device != q.device
            or softmax_lse.dtype != torch.float32
            or not softmax_lse.is_contiguous()
        ):
            raise ValueError("attention log-sum-exp buffer must be contiguous float32 BxHxN")
        assert_int32_addressable(softmax_lse, name="log-sum-exp buffer")

    def grid(metadata: dict[str, Any]) -> tuple[int, int]:
        return triton.cdiv(n_ctx, metadata["BLOCK_M"]), batch * heads

    _flash_attention_forward_kernel[grid](
        q,
        k,
        v,
        output,
        softmax_lse,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        heads,
        1.0 / math.sqrt(head_dim),
        N_CTX=n_ctx,
        HEAD_DIM=head_dim,
        CAUSAL=causal,
        STORE_LSE=store_lse,
    )


def _launch_factory(*, causal: bool) -> LaunchFn:
    """Create an allocation-free, inference-only launcher for one masking mode."""

    def launch(args: TensorArgs, output: torch.Tensor) -> None:
        """Launch forward attention without saving backward statistics."""
        q, k, v = args
        _launch_forward(q, k, v, output, output, causal=causal, store_lse=False)

    return launch


_launch_noncausal = _launch_factory(causal=False)
_launch_causal = _launch_factory(causal=True)

_BWD_BLOCK_M = 64
_BWD_BLOCK_N = 64


@triton.jit  # type: ignore[untyped-decorator]
def _attn_bwd_preprocess_kernel(
    o_ptr: tl.tensor,
    do_ptr: tl.tensor,
    delta_ptr: tl.tensor,
    stride_oz: int,
    stride_oh: int,
    stride_on: int,
    stride_od: int,
    HEADS: int,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
) -> None:
    """Compute the per-query row reduction ``delta = sum_d O * dO``."""
    start_m = tl.program_id(axis=0)
    head_batch = tl.program_id(axis=1)
    batch = head_batch // HEADS
    head = head_batch % HEADS
    offsets_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, HEAD_DIM)
    query_mask = offsets_m < N_CTX
    base = batch * stride_oz + head * stride_oh
    addr = base + offsets_m[:, None] * stride_on + offsets_d[None, :] * stride_od
    o = tl.load(o_ptr + addr, mask=query_mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptr + addr, mask=query_mask[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(delta_ptr + head_batch * N_CTX + offsets_m, delta, mask=query_mask)


@triton.jit  # type: ignore[untyped-decorator]
def _attn_bwd_dkdv_kernel(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    v_ptr: tl.tensor,
    do_ptr: tl.tensor,
    dk_ptr: tl.tensor,
    dv_ptr: tl.tensor,
    softmax_lse_ptr: tl.tensor,
    delta_ptr: tl.tensor,
    stride_z: int,
    stride_h: int,
    stride_n: int,
    stride_d: int,
    HEADS: int,
    sm_scale: float,
    sm_scale_log2: float,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Accumulate ``dK`` and ``dV`` for one key/value block over all queries."""
    start_n = tl.program_id(axis=0)
    head_batch = tl.program_id(axis=1)
    batch = head_batch // HEADS
    head = head_batch % HEADS
    base = batch * stride_z + head * stride_h
    offsets_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_d = tl.arange(0, HEAD_DIM)
    key_mask = offsets_n < N_CTX

    kv_addr = base + offsets_n[:, None] * stride_n + offsets_d[None, :] * stride_d
    k = tl.load(k_ptr + kv_addr, mask=key_mask[:, None], other=0.0)
    v = tl.load(v_ptr + kv_addr, mask=key_mask[:, None], other=0.0)
    dk = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)

    # Causal queries below the diagonal contribute zero, so skip those blocks.
    low = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M if CAUSAL else 0
    for start_m in tl.range(low, N_CTX, BLOCK_M):
        offsets_m = start_m + tl.arange(0, BLOCK_M)
        query_mask = offsets_m < N_CTX
        q_addr = base + offsets_m[:, None] * stride_n + offsets_d[None, :] * stride_d
        q = tl.load(q_ptr + q_addr, mask=query_mask[:, None], other=0.0)
        do = tl.load(do_ptr + q_addr, mask=query_mask[:, None], other=0.0)
        softmax_lse_i = tl.load(
            softmax_lse_ptr + head_batch * N_CTX + offsets_m,
            mask=query_mask,
            other=0.0,
        )
        delta_i = tl.load(delta_ptr + head_batch * N_CTX + offsets_m, mask=query_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k)) * sm_scale_log2
        probabilities = tl.math.exp2(scores - softmax_lse_i[:, None])
        valid = query_mask[:, None] & key_mask[None, :]
        if CAUSAL:
            valid = valid & (offsets_m[:, None] >= offsets_n[None, :])
        probabilities = tl.where(valid, probabilities, 0.0)

        dv += tl.dot(tl.trans(probabilities).to(tl.float16), do)
        dp = tl.dot(do, tl.trans(v))
        ds = probabilities * (dp - delta_i[:, None])
        dk += tl.dot(tl.trans(ds).to(tl.float16), q)

    dk *= sm_scale
    tl.store(dk_ptr + kv_addr, dk.to(dk_ptr.dtype.element_ty), mask=key_mask[:, None])
    tl.store(dv_ptr + kv_addr, dv.to(dv_ptr.dtype.element_ty), mask=key_mask[:, None])


@triton.jit  # type: ignore[untyped-decorator]
def _attn_bwd_dq_kernel(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    v_ptr: tl.tensor,
    do_ptr: tl.tensor,
    dq_ptr: tl.tensor,
    softmax_lse_ptr: tl.tensor,
    delta_ptr: tl.tensor,
    stride_z: int,
    stride_h: int,
    stride_n: int,
    stride_d: int,
    HEADS: int,
    sm_scale: float,
    sm_scale_log2: float,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Accumulate ``dQ`` for one query block over all key/value blocks."""
    start_m = tl.program_id(axis=0)
    head_batch = tl.program_id(axis=1)
    batch = head_batch // HEADS
    head = head_batch % HEADS
    base = batch * stride_z + head * stride_h
    offsets_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, HEAD_DIM)
    query_mask = offsets_m < N_CTX

    q_addr = base + offsets_m[:, None] * stride_n + offsets_d[None, :] * stride_d
    q = tl.load(q_ptr + q_addr, mask=query_mask[:, None], other=0.0)
    do = tl.load(do_ptr + q_addr, mask=query_mask[:, None], other=0.0)
    softmax_lse_i = tl.load(
        softmax_lse_ptr + head_batch * N_CTX + offsets_m,
        mask=query_mask,
        other=0.0,
    )
    delta_i = tl.load(delta_ptr + head_batch * N_CTX + offsets_m, mask=query_mask, other=0.0)
    dq = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Causal queries attend only keys at or before the block's last query.
    high = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX
    for start_n in tl.range(0, high, BLOCK_N):
        offsets_n = start_n + tl.arange(0, BLOCK_N)
        key_mask = offsets_n < N_CTX
        kv_addr = base + offsets_n[:, None] * stride_n + offsets_d[None, :] * stride_d
        k = tl.load(k_ptr + kv_addr, mask=key_mask[:, None], other=0.0)
        v = tl.load(v_ptr + kv_addr, mask=key_mask[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k)) * sm_scale_log2
        probabilities = tl.math.exp2(scores - softmax_lse_i[:, None])
        valid = query_mask[:, None] & key_mask[None, :]
        if CAUSAL:
            valid = valid & (offsets_m[:, None] >= offsets_n[None, :])
        probabilities = tl.where(valid, probabilities, 0.0)

        dp = tl.dot(do, tl.trans(v))
        ds = probabilities * (dp - delta_i[:, None])
        dq += tl.dot(ds.to(tl.float16), k)

    dq *= sm_scale
    tl.store(dq_ptr + q_addr, dq.to(dq_ptr.dtype.element_ty), mask=query_mask[:, None])


def _launch_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    do: torch.Tensor,
    *,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the recompute-based attention backward and return dQ, dK, dV."""
    if do.shape != q.shape or do.device != q.device or do.dtype != q.dtype:
        raise ValueError("attention upstream gradient metadata must match Q")
    do = do.contiguous()
    batch, heads, n_ctx, head_dim = q.shape
    delta = torch.empty((batch, heads, n_ctx), dtype=torch.float32, device=q.device)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    sm_scale = 1.0 / math.sqrt(head_dim)
    sm_scale_log2 = sm_scale * 1.4426950408889634
    head_batches = batch * heads
    strides = q.stride()

    _attn_bwd_preprocess_kernel[(triton.cdiv(n_ctx, _BWD_BLOCK_M), head_batches)](
        output,
        do,
        delta,
        *strides,
        heads,
        N_CTX=n_ctx,
        HEAD_DIM=head_dim,
        BLOCK_M=_BWD_BLOCK_M,
    )
    _attn_bwd_dkdv_kernel[(triton.cdiv(n_ctx, _BWD_BLOCK_N), head_batches)](
        q,
        k,
        v,
        do,
        dk,
        dv,
        softmax_lse,
        delta,
        *strides,
        heads,
        sm_scale,
        sm_scale_log2,
        N_CTX=n_ctx,
        HEAD_DIM=head_dim,
        CAUSAL=causal,
        BLOCK_M=_BWD_BLOCK_M,
        BLOCK_N=_BWD_BLOCK_N,
    )
    _attn_bwd_dq_kernel[(triton.cdiv(n_ctx, _BWD_BLOCK_M), head_batches)](
        q,
        k,
        v,
        do,
        dq,
        softmax_lse,
        delta,
        *strides,
        heads,
        sm_scale,
        sm_scale_log2,
        N_CTX=n_ctx,
        HEAD_DIM=head_dim,
        CAUSAL=causal,
        BLOCK_M=_BWD_BLOCK_M,
        BLOCK_N=_BWD_BLOCK_N,
    )
    return dq, dk, dv


class _FlashAttentionFunction(torch.autograd.Function):
    """Autograd bridge for Flash Attention forward and backward."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        """Run forward and save Q/K/V, the output, and the log-sum-exp."""
        _validate(q, k, v)
        output = torch.empty_like(q)
        batch, heads, n_ctx, _ = q.shape
        softmax_lse = torch.empty(
            (batch, heads, n_ctx),
            dtype=torch.float32,
            device=q.device,
        )
        _launch_forward(
            q,
            k,
            v,
            output,
            softmax_lse,
            causal=causal,
            store_lse=True,
        )
        ctx.save_for_backward(q, k, v, output, softmax_lse)
        ctx.causal = causal
        return output

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        do: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Compute Q, K, and V gradients with a recompute-based backward."""
        q, k, v, output, softmax_lse = ctx.saved_tensors
        dq, dk, dv = _launch_backward(
            q,
            k,
            v,
            output,
            softmax_lse,
            do,
            causal=ctx.causal,
        )
        return dq, dk, dv, None


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """Dispatch inference without saved statistics and training through autograd."""
    _validate(q, k, v)
    needs_grad = torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, k, v))
    if not needs_grad:
        output = torch.empty_like(q)
        _launch_forward(
            q,
            k,
            v,
            output,
            output,
            causal=causal,
            store_lse=False,
        )
        return output
    result = _FlashAttentionFunction.apply(q, k, v, causal)  # type: ignore[no-untyped-call]
    return cast(torch.Tensor, result)


def attention_noncausal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute dense non-causal scaled dot-product attention with autograd."""
    return _attention(q, k, v, causal=False)


def attention_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute dense causal scaled dot-product attention with autograd."""
    return _attention(q, k, v, causal=True)


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
