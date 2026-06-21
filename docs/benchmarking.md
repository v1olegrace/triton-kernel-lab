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
- LayerNorm forward: one effective input read and one output write; affine
  parameters and FP32 row statistics are amortized across 4096 rows.

Small working sets may be influenced by cache. Large working sets are the
relevant points when interpreting DRAM roofline utilization.
`triton.testing.do_bench` flushes L2 between measured repetitions. Values
slightly above 100%, such as the committed vector-add 100.12%, are treated as
measurement noise rather than super-roofline performance.

## Compute roofline

Compute calibration:

1. allocates square FP16 matrices and a reusable output;
2. runs GEMM for approximately three seconds to stabilize clocks;
3. samples `clocks.sm` through `nvidia-smi` during measurement;
4. sweeps matrix sizes;
5. stores every sample and reports the maximum.

The JSON separates:

- measured cuBLAS FP16-input/FP32-accumulate throughput;
- theoretical FP16/FP32-accumulate throughput at the maximum observed clock;
- theoretical FP16/FP16-accumulate throughput at the maximum observed clock.

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
- LayerNorm: row-strided FP32 input/upstream gradient and 1000 columns,
  including all three backward gradients.

## Result interpretation

`speedup` always means Triton divided into the measured PyTorch production
baseline. `speedup_vs_naive` is a separate pedagogical comparison.

Every registered memory-bound kernel records both comparisons:

- `reference_baseline`: the closest production PyTorch operation or direct
  idiomatic composition;
- `naive_baseline`: an explicitly decomposed multi-pass implementation.

The JSON records whether each comparator allocates its output. RoPE has no
single native PyTorch operator, so its production reference is the direct
half-wise rotate-half composition; its naive comparator materializes the
full-width cosine/sine tables and rotated tensor. The labels make that
distinction explicit rather than calling either composition a native op.

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

## Compute-clock policy

`nvidia-smi` clock telemetry is sampled concurrently with both cuBLAS
calibration sweeps. The per-size median/minimum/maximum samples remain in
`peaks.json`, including the clock paired with the best measured cuBLAS point.

The theoretical roofline denominator uses the **maximum SM clock observed
anywhere across both calibration sweeps**, not the median clock attached to
the throughput winner. A sparse polling sample can miss the active-kernel
boost interval; using that lower median as an upper bound can produce the
physically misleading result `measured throughput > theoretical ceiling`.
The maximum observed clock is still empirical and session-specific, but it is
the appropriate upper-bound clock for a roofline. The selected value and
policy are serialized as `theoretical_ceiling_sm_clock_mhz` and
`theoretical_provenance.ceiling_clock_policy`.

## Session provenance

The benchmark guard records five one-second-spaced whole-GPU utilization
samples before any calibration or timing. The default rejects the session if
any sample exceeds 10%; `--allow-busy-gpu` remains a diagnostic-only override.

`peaks.json` stores the calibration timestamp, all five utilization samples,
and the threshold. Every kernel JSON stores its benchmark-session timestamp
and samples plus the calibration-session provenance it consumed. A reviewer
can therefore verify from committed artifacts whether all kernels and their
roofline denominators came from the same clean run.
