# Matmul on RTX 4060

The lab registers two variants of the same tiled Triton kernel:

- `matmul_fp32acc` emits dense FP16 Tensor Core MMA with FP32 accumulation.
- `matmul_fp16acc` emits dense FP16 Tensor Core MMA with FP16 accumulation.

`M`, `N`, `K`, and all strides are runtime values. A compile-time contiguous
fast path specializes the common row-major layout; arbitrary positive strides
use the generic path. Tile sizes, pipeline stages, warp count, grouping, and
accumulator type are compile-time values. M, N, and K tails load zero, the
additive identity, while the output store masks invalid coordinates.

## Numerical policy

Correctness uses a float32 PyTorch reference and relative Frobenius error,
rather than elementwise `allclose`. The GPU adversarial case multiplies
strided `129x193` by strided `193x257`, exercising the generic-layout
fallback and M, N, and K tails together.

The test suite enforces relative Frobenius error below `1e-2` for FP32
accumulation and below `5e-2` for FP16 accumulation.

FP16 accumulation is an inference-oriented throughput mode, not a free
training optimization. Accumulating thousands of products in FP16 discards
mantissa bits and can fail on ill-conditioned or high-dynamic-range inputs.
The aggregate tolerance and well-conditioned random tests demonstrate the
implemented instruction path; they do not establish training safety.

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

The best clean measurements with Triton 3.7.1 were:

- FP32 accumulation: 30.95 TFLOP/s at N=2048, or 96.5% of the
  clock-scaled theoretical ceiling and 100.2% of cuBLAS at the same size.
- FP16 accumulation: 53.09 TFLOP/s at N=2048, 80.0% of the clock-scaled
  theoretical ceiling.
- End-to-end accumulation-mode gap: 1.72x at N=2048 and 1.63x at N=4096.

PTX inspection confirms that the two variants select different instructions:

- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
- `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`

The missing ideal 2x is therefore not a failure to propagate `out_dtype`.
FP16 accumulation makes non-MMA work proportionally more expensive, but
specializing contiguous strides removed the dominant code-generation penalty:
the compiler can fold `stride_ak=1`, `stride_bk=N`, and contiguous output
addressing instead of carrying fully dynamic stride arithmetic through the
inner loop.

The search covers grouping from 1 through 32, `BK=32/64`, two through six
stages, and two through eight warps. Larger 256x256 candidates either
regressed or exceeded the 101,376-byte per-block shared-memory limit reported
by the compiler.

## Split-K experiment

Split-K with FP32 atomic merging was implemented and measured before deciding
whether to keep it in the production registry. It regressed in this
small-square-GEMM regime on AD107, where atomic merge contention outweighed
the extra wave parallelism.

The wave-quantization premise for N=512 assumed a 128x128 winner and only 16
output programs. The actual FP32-accumulate winner is 64x64, which launches 64
output programs across 24 SMs before any K splitting. In a controlled 64x64,
FP32-output experiment, increasing `SPLIT_K` produced:

| N | split 1 | split 2 | split 4 | split 8 |
|---:|---:|---:|---:|---:|
| 512 | 13.54 | 4.54 | 3.64 | 2.40 TFLOP/s |
| 1024 | 25.30 | 12.54 | 8.80 | 5.41 TFLOP/s |

The extra programs did not compensate for FP32 output traffic and atomic
merge overhead. The autotuner also selected split 1 for the FP16-accumulate
path. Keeping this split-K design in the public kernel would therefore have
made the measured workload more complex and slower. Low-batch inference where
small GEMMs dominate would justify revisiting the problem with stream-K rather
than treating this result as a universal rejection of K splitting.

The production autotune space includes stages 5 and 6. The final large-N
FP32 winners still use four stages; deeper pipelines are retained because a
six-stage configuration wins the small N=512 FP16-accumulate case.

These measurements reject the proposed optimization, but they do not prove a
hardware ceiling. Attributing the remaining gap solely to SM89 would require
counter-level evidence from a profiler.

The single-launch Nsight Compute target and AD107-specific counter selection
are documented in [debugging.md](debugging.md). Counter evidence is kept
separate from throughput inferred from elapsed time and theoretical FLOP
rates.

The clean NCU collection confirms 95.73% of the mode-specific FP16-input /
FP32-accumulate Tensor path peak and 89.91% for FP16 accumulation. FP16
accumulation raises SM throughput from 47.38% to 86.90% and DRAM throughput
from 11.79% to 21.63%, while active-warp occupancy remains approximately 32%.
These values include the different autotuned launch configurations selected
by each production variant.

| Independent measurement | FP32 accumulate | FP16 accumulate |
|---|---:|---:|
| timed throughput / clock-scaled theoretical peak | 96.5% | 80.0% |
| NCU mode-specific Tensor operations / peak | 95.73% | 89.91% |
| NCU SM throughput / full-rate peak | 47.38% | 86.90% |
| NCU DRAM throughput / peak | 11.79% | 21.63% |
| NCU active warps / peak | 32.26% | 31.95% |

For FP32 accumulation, the Ada Tensor path has half the FP16-accumulate
operation rate. Normalizing `47.38%` from the full-rate denominator gives
`94.76%`, agreeing with both the mode-specific 95.73% hardware counter and
the independently timed 96.5% result. There is no material FP32-accumulate
Tensor-pipe headroom at this shape.

FP16 accumulation nearly doubles HMMA issue and raises DRAM pressure by 1.83x
without changing active-warp occupancy. The counters rule out occupancy as
the cause of the sub-2x speedup and support increased operand-feed pressure.
They do not, by themselves, identify a particular memory-stall source; that
would require scheduler-stall and cache-level profiling.

The contiguous fast path uses signed int32 address arithmetic. Public
validation rejects layouts whose maximum relative offset exceeds
`2^31 - 1`; larger tensors require an explicitly audited int64-addressing
variant.

## cuBLAS denominator

Result schema 4 separates two quantities:

- `pct_cublas_same_size`: Triton divided by cuBLAS at the same N;
- `pct_cublas_peak`: Triton divided by the best cuBLAS result in the full
  size sweep.

For example, the final FP32 kernel reaches 15.42 TFLOP/s at N=512: 49.9% of
the global 30.9 TFLOP/s cuBLAS peak, but 117.7% of cuBLAS at the same size.
The former is a roofline-utilization number; the latter is the valid
same-shape scheduling comparison.
