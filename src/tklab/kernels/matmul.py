"""Autotuned FP16 matrix multiplication with selectable accumulation precision."""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from tklab.harness.addressing import assert_int32_addressable
from tklab.harness.tolerances import assert_relative_frobenius
from tklab.registry import (
    BenchmarkMetadataFn,
    BenchmarkScalar,
    KernelSpec,
    LaunchFn,
    TensorArgs,
    register,
)


def _configs() -> list[triton.Config]:
    """Return autotune configurations that fit the RTX 4060 SM89 limits.

    The search includes deeper 128x128 pipelines to test whether extra
    software-pipeline stages improve large-matrix throughput on Ada.
    """
    space = (
        # BM, BN, BK, stages, warps, group_m
        (64, 64, 32, 2, 4, 8),
        (64, 64, 32, 3, 4, 1),
        (64, 64, 32, 4, 4, 1),
        (64, 64, 32, 5, 4, 1),
        (64, 64, 32, 4, 4, 8),
        (64, 128, 32, 4, 4, 8),
        (128, 64, 32, 4, 4, 8),
        (128, 128, 32, 4, 4, 8),
        (128, 128, 32, 5, 4, 8),
        (128, 128, 32, 6, 4, 8),
        (128, 128, 32, 4, 8, 8),
        (128, 128, 32, 5, 8, 8),
        (128, 128, 32, 6, 8, 8),
        (128, 128, 32, 4, 4, 4),
        (64, 256, 32, 4, 4, 8),
        (64, 256, 32, 4, 4, 32),
        (256, 64, 32, 4, 4, 8),
        (128, 128, 64, 3, 8, 8),
        (64, 128, 64, 4, 4, 8),
        (128, 64, 64, 4, 4, 8),
        (64, 64, 64, 4, 4, 8),
        (128, 32, 32, 4, 4, 8),
        (32, 128, 32, 4, 4, 8),
        (64, 32, 32, 5, 2, 8),
        (32, 64, 32, 5, 2, 8),
    )
    return [
        triton.Config(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": group_m,
            },
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_m, block_n, block_k, num_stages, num_warps, group_m in space
    ]


@triton.autotune(  # type: ignore[untyped-decorator]
    configs=_configs(),
    key=["M", "N", "K", "ACC_DTYPE", "CONTIGUOUS"],
)
@triton.jit  # type: ignore[untyped-decorator]
def _matmul_kernel(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    c_ptr: tl.tensor,
    M: int,
    N: int,
    K: int,
    stride_am: int,
    stride_ak: int,
    stride_bk: int,
    stride_bn: int,
    stride_cm: int,
    stride_cn: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
) -> None:
    """Compute one grouped output tile and mask M, N, and K tails."""
    tl.static_assert(BLOCK_K >= 16, "Tensor Core K tile must be at least 16")
    tl.static_assert(BLOCK_K % 16 == 0, "Tensor Core K tile must be a multiple of 16")
    program_id = tl.program_id(axis=0)
    num_programs_m = tl.cdiv(M, BLOCK_M)
    num_programs_n = tl.cdiv(N, BLOCK_N)

    programs_per_group = GROUP_M * num_programs_n
    group_id = program_id // programs_per_group
    first_program_m = group_id * GROUP_M
    group_size_m = min(num_programs_m - first_program_m, GROUP_M)
    program_m = first_program_m + ((program_id % programs_per_group) % group_size_m)
    program_n = (program_id % programs_per_group) // group_size_m

    tl.assume(program_m >= 0)
    tl.assume(program_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    if CONTIGUOUS:
        a_block_ptrs = a_ptr + offsets_m[:, None] * K + offsets_k[None, :]
        b_block_ptrs = b_ptr + offsets_k[:, None] * N + offsets_n[None, :]
    else:
        a_block_ptrs = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
        b_block_ptrs = b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        remaining_k = K - k_block * BLOCK_K
        a = tl.load(
            a_block_ptrs,
            mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < remaining_k),
            other=0.0,
        )
        b = tl.load(
            b_block_ptrs,
            mask=(offsets_k[:, None] < remaining_k) & (offsets_n[None, :] < N),
            other=0.0,
        )
        if tl.float32 == ACC_DTYPE:
            accumulator = tl.dot(a, b, accumulator)
        else:
            accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float16)
        if CONTIGUOUS:
            a_block_ptrs += BLOCK_K
            b_block_ptrs += BLOCK_K * N
        else:
            a_block_ptrs += BLOCK_K * stride_ak
            b_block_ptrs += BLOCK_K * stride_bk

    output = accumulator.to(c_ptr.dtype.element_ty)
    output_offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if CONTIGUOUS:
        output_ptrs = c_ptr + output_offsets_m[:, None] * N + output_offsets_n[None, :]
    else:
        output_ptrs = (
            c_ptr + output_offsets_m[:, None] * stride_cm + output_offsets_n[None, :] * stride_cn
        )
    output_mask = (output_offsets_m[:, None] < M) & (output_offsets_n[None, :] < N)
    tl.store(output_ptrs, output, mask=output_mask)


def _validate(a: torch.Tensor, b: torch.Tensor) -> None:
    """Validate the public FP16 matrix-multiplication contract."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects two 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes: {a.shape} and {b.shape}")
    if 0 in (*a.shape, *b.shape):
        raise ValueError("matmul does not support empty dimensions")
    if a.device != b.device:
        raise ValueError(f"device mismatch: {a.device} != {b.device}")
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch: {a.dtype} != {b.dtype}")
    if a.dtype != torch.float16:
        raise ValueError("matmul currently supports float16 inputs only")
    for name, tensor in (("left", a), ("right", b)):
        if any(stride <= 0 for stride in tensor.stride()):
            raise ValueError(f"{name} input requires strictly positive strides")
        assert_int32_addressable(tensor, name=name)
    if a.device.type != "cuda":
        raise ValueError("matmul requires CUDA tensors")


def _make_output(args: TensorArgs) -> torch.Tensor:
    """Allocate a contiguous FP16 output matrix."""
    a, b = args
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)


def _launch_factory(accumulator_dtype: tl.dtype) -> LaunchFn:
    """Create an allocation-free launcher for one accumulator dtype."""

    def launch(args: TensorArgs, output: torch.Tensor) -> None:
        """Launch the autotuned matmul into ``output``."""
        a, b = args
        _validate(a, b)
        if output.shape != (a.shape[0], b.shape[1]):
            raise ValueError("output shape does not match matmul result")
        if output.device != a.device or output.dtype != a.dtype:
            raise ValueError("output metadata must match the inputs")
        if not output.is_contiguous():
            raise ValueError("matmul output must be contiguous")

        rows, inner = a.shape
        _, columns = b.shape

        def grid(metadata: dict[str, Any]) -> tuple[int]:
            return (
                triton.cdiv(rows, metadata["BLOCK_M"]) * triton.cdiv(columns, metadata["BLOCK_N"]),
            )

        _matmul_kernel[grid](
            a,
            b,
            output,
            rows,
            columns,
            inner,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            output.stride(0),
            output.stride(1),
            ACC_DTYPE=accumulator_dtype,
            CONTIGUOUS=a.is_contiguous() and b.is_contiguous(),
        )

    return launch


_launch_fp32acc = _launch_factory(tl.float32)
_launch_fp16acc = _launch_factory(tl.float16)


def matmul_fp32acc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply FP16 matrices using FP32 Tensor Core accumulation."""
    _validate(a, b)
    output = _make_output((a, b))
    _launch_fp32acc((a, b), output)
    return output


def matmul_fp16acc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply FP16 matrices using FP16 Tensor Core accumulation."""
    _validate(a, b)
    output = _make_output((a, b))
    _launch_fp16acc((a, b), output)
    return output


def _launch_reference(args: TensorArgs, output: torch.Tensor) -> None:
    """Launch allocation-free PyTorch matrix multiplication."""
    a, b = args
    torch.mm(a, b, out=output)


def _make_inputs(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TensorArgs:
    """Create square random matrices for correctness and benchmarking."""
    return (
        torch.randn(size, size, device=device, dtype=dtype),
        torch.randn(size, size, device=device, dtype=dtype),
    )


def _make_adversarial(device: torch.device) -> TensorArgs:
    """Create strided matrices with simultaneous M-, N-, and K-tail tiles."""
    left_storage = torch.randn(129, 386, device=device, dtype=torch.float16)
    right_storage = torch.randn(386, 257, device=device, dtype=torch.float16)
    return left_storage[:, ::2], right_storage[::2, :]


def _assert_fp32acc(output: torch.Tensor, reference: torch.Tensor) -> None:
    """Validate FP32-accumulating matmul by relative Frobenius error."""
    assert_relative_frobenius(output, reference, max_relative_error=1e-2)


def _assert_fp16acc(output: torch.Tensor, reference: torch.Tensor) -> None:
    """Validate FP16-accumulating matmul with a looser aggregate bound."""
    assert_relative_frobenius(output, reference, max_relative_error=5e-2)


def _autotune_metadata_factory(mma_opcode: str) -> BenchmarkMetadataFn:
    """Create a callback that serializes the winning autotune configuration."""

    def metadata(_size: int, _dtype: torch.dtype) -> dict[str, BenchmarkScalar]:
        """Return the most recently selected Triton configuration."""
        config = _matmul_kernel.best_config
        return {
            "block_m": int(config.kwargs["BLOCK_M"]),
            "block_n": int(config.kwargs["BLOCK_N"]),
            "block_k": int(config.kwargs["BLOCK_K"]),
            "group_m": int(config.kwargs["GROUP_M"]),
            "num_warps": int(config.num_warps),
            "num_stages": int(config.num_stages),
            "mma_opcode": mma_opcode,
        }

    return metadata


_metadata_fp32acc = _autotune_metadata_factory("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
_metadata_fp16acc = _autotune_metadata_factory("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16")

_SIZES = (512, 1024, 2048, 4096, 8192)


def _flops(size: int, _dtype: torch.dtype) -> int:
    """Return the conventional ``2*M*N*K`` FLOP count for square GEMM."""
    return 2 * size**3


MATMUL_FP32ACC = register(
    KernelSpec(
        name="matmul_fp32acc",
        description="Autotuned FP16 GEMM with FP32 Tensor Core accumulation.",
        triton_fn=matmul_fp32acc,
        launch_fn=_launch_fp32acc,
        ref_fn=torch.matmul,
        reference_launch_fn=_launch_reference,
        make_inputs=_make_inputs,
        make_output=_make_output,
        make_adversarial=_make_adversarial,
        sizes=_SIZES,
        correctness_sizes=(512, 1024),
        bound="compute",
        flops=_flops,
        dtypes=(torch.float16,),
        assert_fn=_assert_fp32acc,
        benchmark_metadata=_metadata_fp32acc,
        compute_mode="fp16_fp32acc",
        supports_interpreter=False,
    )
)

MATMUL_FP16ACC = register(
    KernelSpec(
        name="matmul_fp16acc",
        description="Autotuned FP16 GEMM with FP16 Tensor Core accumulation.",
        triton_fn=matmul_fp16acc,
        launch_fn=_launch_fp16acc,
        ref_fn=torch.matmul,
        reference_launch_fn=_launch_reference,
        make_inputs=_make_inputs,
        make_output=_make_output,
        make_adversarial=_make_adversarial,
        sizes=_SIZES,
        correctness_sizes=(512, 1024),
        bound="compute",
        flops=_flops,
        dtypes=(torch.float16,),
        assert_fn=_assert_fp16acc,
        benchmark_metadata=_metadata_fp16acc,
        compute_mode="fp16_fp16acc",
        supports_interpreter=False,
    )
)
