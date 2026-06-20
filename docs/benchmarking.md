# Benchmark methodology

This document defines what the benchmark numbers mean and which comparisons
are valid.

## Timing

All GPU timings use `triton.testing.do_bench`. It uses CUDA events, performs
warmup, and clears L2 between measured repetitions. Wall-clock timing around
asynchronous CUDA launches is intentionally avoided.

Before allocating benchmark inputs, the CLI checks whole-device utilization
through `nvidia-smi`. It refuses to run above 10% pre-existing utilization,
because desktop applications, games, and animated wallpapers can depress both
Triton and cuBLAS clocks. `--allow-busy-gpu` is an explicit escape hatch for
diagnostics; results produced with it should not be committed as reference
measurements.

Triton launchers and production PyTorch references receive preallocated output
buffers. Allocation is therefore outside the timed region. A pedagogical
multi-pass baseline may allocate intermediates because those intermediates
are the behavior being demonstrated.

## Memory roofline

The bandwidth calibration sweeps several FP32 device-copy working sets. Each
observation counts one read and one write. The maximum sustained median is
stored alongside every sample rather than replacing the curve with a single
number.

Kernel byte models are dtype-aware:

- vector add: two reads and one write;
- fused softmax: one global read and one global write.

Small working sets may be influenced by cache. Large working sets are the
relevant points when interpreting DRAM roofline utilization.

## Compute roofline

Compute calibration:

1. allocates square FP16 matrices and a reusable output;
2. runs GEMM for approximately three seconds to stabilize clocks;
3. samples `clocks.sm` through `nvidia-smi` during measurement;
4. sweeps matrix sizes;
5. stores every sample and reports the maximum.

The JSON separates:

- measured cuBLAS FP16-input/FP32-accumulate throughput;
- theoretical FP16/FP32-accumulate throughput at measured clock;
- theoretical FP16/FP16-accumulate throughput at measured clock.

Theoretical numbers are derived only from an audited hardware profile with
primary-source URLs. Unsupported GPUs fail explicitly rather than inheriting
RTX 4060 constants.

## Numerical correctness

Elementwise kernels use dtype-aware `torch.testing.assert_close` tolerances.
Matmul uses relative Frobenius error against a float32 reference because large
reductions can contain isolated elementwise outliers while retaining low
aggregate error.

Adversarial cases run on real GPU where supported:

- vector add: stride-2/stride-3, length 1009;
- softmax: row stride and 1000 columns;
- matmul: `129×193 @ 193×257`.

## Result interpretation

`speedup` always means Triton divided into the measured PyTorch production
baseline. `speedup_vs_naive` is a separate pedagogical comparison.

For memory-bound kernels, `% peak` is the primary metric. FP32-accumulating
GEMM reports two deliberately separate cuBLAS comparisons:

- `pct_cublas_same_size` compares Triton with the allocation-free cuBLAS
  observation for the same matrix size. This is the valid scheduling and
  implementation-efficiency comparison.
- `pct_cublas_peak` compares Triton with the maximum cuBLAS throughput from
  the complete calibration sweep. This is a practical roofline utilization,
  not a same-shape speedup.

Using the global cuBLAS peak as the denominator for a small matrix can make
launch and wave effects look much worse than the vendor kernel at that same
shape. The JSON therefore never labels the global ratio simply as
`pct_cublas`.

For FP16 accumulation, PyTorch currently does not expose a distinct fair
cuBLAS baseline, so only the theoretical denominator and direct comparison
against the FP32-accumulating Triton kernel are reported.
