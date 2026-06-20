# Matmul on RTX 4060

The lab registers two variants of the same tiled Triton kernel:

- `matmul_fp32acc` emits dense FP16 Tensor Core MMA with FP32 accumulation.
- `matmul_fp16acc` emits dense FP16 Tensor Core MMA with FP16 accumulation.

`M`, `N`, `K`, and all strides are runtime values. Tile sizes, pipeline
stages, warp count, grouping, and accumulator type are compile-time values.
The K tail loads zero, the additive identity, while the output store masks
the M and N tails.

## Numerical policy

Correctness uses a float32 PyTorch reference and relative Frobenius error,
rather than elementwise `allclose`. The GPU adversarial case multiplies
`129x193` by `193x257`, exercising M, N, and K tails together.

The observed adversarial relative errors were approximately:

- FP32 accumulation: `2.1e-4`
- FP16 accumulation: `7.6e-4`

## Compute baselines

PyTorch 2.12 does not expose a distinct cuBLAS FP16-accumulate GEMM through
`torch.mm`. Enabling `allow_fp16_reduced_precision_reduction` leaves measured
throughput around 30 TFLOP/s, the FP32-accumulate rate. The roofline therefore
stores that sweep but does not label it as an FP16-accumulate baseline.

The theoretical rates are derived from NVIDIA's RTX 4060 product
specification and Ada architecture whitepaper:

- FP16 inputs with FP32 accumulation: about 30.2 TFLOP/s at rated boost.
- FP16 inputs with FP16 accumulation: twice that rate.

Sources:

- <https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/>
- <https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf>

## RTX 4060 result

The best measured results with Triton 3.7.1 were:

- FP32 accumulation: about 20.7 TFLOP/s, or 67% of measured cuBLAS.
- FP16 accumulation: about 27.7 TFLOP/s.
- End-to-end accumulation-mode gap: about 1.34x at 4096.

PTX inspection confirms that the two variants select different instructions:

- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
- `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`

The missing ideal 2x is therefore not a failure to propagate `out_dtype`.
The conventional Triton pipeline remains limited by non-MMA work and code
generation on SM89. Expanded tile searches, grouping from 1 through 32,
`BK=32/64`, two through five stages, two through eight warps, and register
limits did not close the gap. Larger 256x256 candidates either regressed or
exceeded the 101,376-byte per-block shared-memory limit reported by the
compiler.

This result does not meet the original 85% cuBLAS target. Reaching that level
requires a different kernel structure or deeper profiling, such as a
persistent GEMM and Nsight Compute analysis, rather than more tuning of the
same tutorial-style kernel.
