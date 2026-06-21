# GPU debugging and profiling

This document records the real-GPU validation used to complement numerical
comparison against PyTorch. Numerical agreement validates formulas; it does
not, by itself, prove that masked accesses, synchronization, or global-memory
coordination are correct.

## Environment

The committed Phase 5 evidence was collected on:

- NVIDIA GeForce RTX 4060 desktop, compute capability 8.9;
- WSL2 Ubuntu 24.04;
- PyTorch 2.12.1 with CUDA 13.0;
- Triton 3.7.1;
- Compute Sanitizer 2026.2.0;
- Nsight Compute CLI 2026.2.0.

Tool versions are recorded because sanitizer behavior and profiler metric
names change between CUDA releases.

## Compute Sanitizer

The focused workloads in `tests/test_sanitizer_workloads.py` exercise:

- vector-add non-unit strides and a masked tail;
- softmax row stride and a non-power-of-two width;
- both matmul accumulation modes on contiguous tensors and the strided
  `129x193 @ 193x257` M/N/K-tail case;
- LayerNorm row-strided forward and lock-reduced backward;
- RMSNorm row-strided forward and single-buffer lock-reduced backward;
- fused residual RMSNorm with independent strides on both inputs and both
  output-gradient paths;
- SwiGLU and RoPE forward/backward with independent row strides and masked
  tails.

Run them with:

```bash
compute-sanitizer --tool memcheck \
  python -m pytest tests/test_sanitizer_workloads.py

compute-sanitizer --tool initcheck \
  python -m pytest tests/test_sanitizer_workloads.py

compute-sanitizer --tool racecheck \
  python -m pytest tests/test_sanitizer_workloads.py -k layer_norm

compute-sanitizer --tool synccheck \
  python -m pytest tests/test_sanitizer_workloads.py -k layer_norm

compute-sanitizer --tool memcheck \
  python -m pytest tests/test_sanitizer_workloads.py \
  -k "rms_norm and not residual"

compute-sanitizer --tool initcheck \
  python -m pytest tests/test_sanitizer_workloads.py \
  -k "rms_norm and not residual"

compute-sanitizer --tool racecheck \
  python -m pytest tests/test_sanitizer_workloads.py \
  -k "rms_norm and not residual"

compute-sanitizer --tool synccheck \
  python -m pytest tests/test_sanitizer_workloads.py \
  -k "rms_norm and not residual"

compute-sanitizer --tool memcheck \
  python -m pytest tests/test_sanitizer_workloads.py -k residual_rms_norm

compute-sanitizer --tool initcheck \
  python -m pytest tests/test_sanitizer_workloads.py -k residual_rms_norm

compute-sanitizer --tool racecheck \
  python -m pytest tests/test_sanitizer_workloads.py -k residual_rms_norm

compute-sanitizer --tool synccheck \
  python -m pytest tests/test_sanitizer_workloads.py -k residual_rms_norm

compute-sanitizer --tool memcheck \
  python -m pytest tests/test_sanitizer_workloads.py -k "swiglu or rope"

compute-sanitizer --tool initcheck \
  python -m pytest tests/test_sanitizer_workloads.py -k "swiglu or rope"

compute-sanitizer --tool racecheck \
  python -m pytest tests/test_sanitizer_workloads.py -k "swiglu or rope"

compute-sanitizer --tool synccheck \
  python -m pytest tests/test_sanitizer_workloads.py -k "swiglu or rope"
```

The RTX 4060 run completed with:

| Tool | Scope | Result |
|---|---|---|
| memcheck | all focused kernels | 0 errors |
| initcheck | all focused kernels | 0 errors |
| racecheck | LayerNorm | 0 errors, 0 warnings |
| synccheck | LayerNorm | 0 errors |
| memcheck | RMSNorm | 0 errors |
| initcheck | RMSNorm | 0 errors |
| racecheck | RMSNorm | 0 errors, 0 warnings |
| synccheck | RMSNorm | 0 errors |
| memcheck | residual RMSNorm | 0 errors |
| initcheck | residual RMSNorm | 0 errors |
| racecheck | residual RMSNorm | 0 errors, 0 warnings |
| synccheck | residual RMSNorm | 0 errors |
| memcheck | SwiGLU + RoPE | 0 errors |
| initcheck | SwiGLU + RoPE | 0 errors |
| racecheck | SwiGLU + RoPE | 0 errors, 0 warnings |
| synccheck | SwiGLU + RoPE | 0 errors |

The raw summaries are committed under
`results/nvidia_geforce_rtx_4060/compute_sanitizer_*.log`.
RMSNorm-specific summaries use the
`compute_sanitizer_rms_norm_*.log` prefix.
Residual RMSNorm summaries use
`compute_sanitizer_residual_rms_norm_*.log`.
The elementwise summaries use `compute_sanitizer_elementwise_*.log`.

### Coverage limits

`memcheck` detects out-of-bounds and misaligned device-memory accesses.
`initcheck` detects reads from uninitialized global memory.
`racecheck` diagnoses shared-memory hazards within a thread block.
`synccheck` diagnoses invalid synchronization usage.

The LayerNorm partial-gradient lock coordinates independent programs through
global memory. `racecheck` does not independently verify that inter-block
protocol. The implementation uses explicit `acq_rel` atomics at GPU scope,
while a high-contention repeated numerical test validates the complete
protocol on real hardware.

PTX inspection on SM89 confirms `atom.global.acq_rel.gpu.cas` for lock
acquisition, a block barrier before unlock, and
`atom.global.gpu.acq_rel.exch` for release.

## LayerNorm global-lock stress

Reproduce the committed stress study with:

```bash
python benchmarks/layer_norm_lock_stress.py \
  --rows 65536 \
  --columns 1024 \
  --runs 50 \
  --group-size 8 \
  --group-size 32 \
  --group-size 128
```

The most contended configuration assigns 8,192 rows to each lock slot, four
times the requested 2,048-row target. Every group count passed the following
thresholds:

- relative Frobenius error against PyTorch below `1e-2`;
- maximum run-to-run relative drift below `1e-4`;
- relative drift between group counts below `1e-4`.

Results are not expected to be bitwise identical. Competing programs can
acquire the lock in different orders, changing the order of FP32 additions.
That is ordinary floating-point non-associativity, not evidence of lost
updates. A race signal would be large run-to-run drift, large reference error,
or an error that changes materially with the number of lock groups.

The complete values and thresholds are stored in
`results/nvidia_geforce_rtx_4060/layer_norm_lock_stress.json`.

The ordinary gradient suite also combines previously separate dimensions:
1,025 row-strided rows, 1,000 columns, masked tails, and up to five rows per
default lock slot. This catches interactions that a power-of-two contention
test and a low-row-count tail test can each miss in isolation.

## RMSNorm global-lock stress

RMSNorm removes LayerNorm's `dbias` partial but retains a separate count half
inside the lock allocation. The stress study verifies that surgery directly:

```bash
python benchmarks/rms_norm_lock_stress.py \
  --rows 1025 \
  --columns 1000 \
  --runs 50 \
  --group-size 1 \
  --group-size 8 \
  --group-size 256 \
  --group-size 2048
```

The group-size-1 case forces exactly 1,025 row programs through one lock.
After every stage-1 launch, the script checks that all lock slots are released,
all active count slots equal one, and all inactive count slots remain zero.
It then validates `dweight` against a PyTorch FP32 reference and measures
run-to-run and group-size drift.

All 200 launches passed. The one-lock case reached approximately `5.75e-7`
relative reference error and `7.43e-7` maximum repeated drift. Complete
results and thresholds are stored in `rms_norm_lock_stress.json`.

## Triton interpreter limitation

`TRITON_INTERPRET=1` is useful for scalar logic, masks, and indexing, but it
does not reproduce GPU scheduling or the memory-ordering behavior of atomics
between programs. Interpreter tests therefore cannot establish correctness of
the LayerNorm global lock. That claim requires real-GPU stress and tool-based
validation.

## Nsight Compute

`benchmarks/profile_matmul.py` warms a specialization before placing exactly
one matmul launch between `cudaProfilerStart` and `cudaProfilerStop`.

On AD107 with Nsight Compute 2026.2, the relevant metrics are:

- `sm__inst_executed_pipe_tensor_op_hmma_v2.avg.pct_of_peak_sustained_active`;
- `sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.avg.pct_of_peak_sustained_active`;
- `sm__ops_path_tensor_src_fp16_dst_fp16_sparsity_off.avg.pct_of_peak_sustained_active`;
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`;
- `sm__warps_active.avg.pct_of_peak_sustained_active`;
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`.

The older `sm__pipe_tensor_op_hmma_cycles_active...` spelling is not exposed
for this chip/tool combination. Metric availability must be queried for the
target GPU instead of copied across architectures or Nsight versions.

The generic HMMA metric counts issued warp instructions. It is not a direct
measure of useful FLOP throughput when accumulator modes have different
operation rates. The two `sm__ops_path_tensor...` metrics provide the
mode-specific denominators used for the final utilization claim.

Profile both accumulation modes:

```bash
METRICS=sm__inst_executed_pipe_tensor_op_hmma_v2.avg.pct_of_peak_sustained_active,\
sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.avg.pct_of_peak_sustained_active,\
sm__ops_path_tensor_src_fp16_dst_fp16_sparsity_off.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed

ncu --profile-from-start off --target-processes all \
  -k regex:_matmul_kernel -c 1 \
  --metrics "$METRICS" --csv --page raw \
  python benchmarks/profile_matmul.py --mode fp32acc --size 2048

ncu --profile-from-start off --target-processes all \
  -k regex:_matmul_kernel -c 1 \
  --metrics "$METRICS" --csv --page raw \
  python benchmarks/profile_matmul.py --mode fp16acc --size 2048
```

On Windows/WSL, `ERR_NVGPUCTRPERM` means the driver has disabled access to
hardware counters for the current user. Enable access through NVIDIA Control
Panel under Developer settings before interpreting any profiler result.

### RTX 4060 result

The final collection used one `2048x2048` launch per mode while the Windows
screen was locked, after the GPU had returned to idle:

| Metric | FP32 accumulate | FP16 accumulate |
|---|---:|---:|
| mode-specific Tensor ops / peak | 95.73% | 89.91% |
| HMMA instruction issue / peak | 23.93% | 44.95% |
| SM throughput / peak | 47.38% | 86.90% |
| active warps / peak | 32.26% | 31.95% |
| DRAM throughput / peak | 11.79% | 21.63% |
| NCU kernel duration | 602.720 us | 336.352 us |
| registers per thread | 80 | 108 (112 allocated) |
| dynamic shared memory | 24 KiB | 48 KiB |

The counter evidence confirms that FP32 accumulation is already near its
mode-specific Tensor Core ceiling. FP16 accumulation nearly doubles HMMA
instruction issue and raises both SM and DRAM pressure while active-warp
occupancy remains effectively unchanged. Its 89.91% mode-specific utilization
also shows why the end-to-end gain remains below the ideal 2x.

Normalizing the FP32-accumulate SM counter from the full-rate FP16 denominator
to the architecture's half-rate FP16-input/FP32-accumulate ceiling gives
`47.38% x 2 = 94.76%`. That independently agrees with the 95.73%
mode-specific Tensor counter and the 96.5% clock-scaled result from timed
TFLOP/s. The FP16-accumulate path increases DRAM pressure by 1.83x while
active-warp occupancy stays flat, excluding occupancy as the explanation for
the accumulation-mode gap.

These counters support increased operand-feed pressure, but DRAM throughput
alone does not prove a specific memory-stall mechanism. Scheduler stall and
cache-level counters would be required to attribute the remaining gap solely
to memory starvation.

This is not a controlled accumulator-dtype-only experiment: autotuning selects
different launch configurations. FP32 accumulation uses 128 threads and 1,024
blocks; FP16 accumulation uses 256 threads and 256 blocks, with more registers
and shared memory per block. The result characterizes the production variants
as shipped, not an isolated instruction microbenchmark.

Raw NCU CSV files and a compact JSON summary are committed under
`results/nvidia_geforce_rtx_4060/`.

## Kernel-side diagnostics

Compile-time invariants should use `tl.static_assert`. The matmul kernel
asserts that `BLOCK_K` is at least 16 and divisible by 16, matching the K
granularity of its Tensor Core MMA path.

`tl.device_print` is reserved for targeted diagnosis because every active
program can emit output. Restrict it to a single program and a small subset of
lanes, for example while investigating one tile:

```python
if tl.program_id(0) == 0:
    tl.device_print("acc", acc)
```

Debug prints are never left enabled in benchmark or production paths.
