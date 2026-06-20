"""Single-launch matmul target for Nsight Compute."""

from __future__ import annotations

import argparse

import torch

from tklab.kernels.matmul import matmul_fp16acc, matmul_fp32acc


def main() -> int:
    """Warm one specialization, then execute one profiled launch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fp32acc", "fp16acc"), required=True)
    parser.add_argument("--size", type=int, default=2048)
    options = parser.parse_args()
    if options.size <= 0:
        raise ValueError("size must be positive")

    torch.manual_seed(17)
    left = torch.randn(
        options.size,
        options.size,
        device="cuda",
        dtype=torch.float16,
    )
    right = torch.randn_like(left)
    kernel = matmul_fp32acc if options.mode == "fp32acc" else matmul_fp16acc

    for _ in range(5):
        kernel(left, right)
    torch.cuda.synchronize()

    cuda_runtime = torch.cuda.cudart()  # type: ignore[no-untyped-call]
    cuda_runtime.cudaProfilerStart()
    kernel(left, right)
    cuda_runtime.cudaProfilerStop()
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
