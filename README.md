# Triton Kernel Lab

[![CI](https://github.com/v1olegrace/triton-kernel-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/v1olegrace/triton-kernel-lab/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/v1olegrace/triton-kernel-lab/branch/main/graph/badge.svg)](https://codecov.io/gh/v1olegrace/triton-kernel-lab)
[![Python](https://img.shields.io/badge/python-3.10--3.14-3776AB.svg)](https://www.python.org/)
[![Triton](https://img.shields.io/badge/Triton-3.7%2B-5C4EE5.svg)](https://triton-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](pyproject.toml)

Production-style GPU kernels built with Triton, validated against
high-precision PyTorch references, and measured against empirical rooflines.
The project treats correctness, numerical policy, benchmark provenance, and
performance artifacts as one shared engineering system.

The kernels are not a random sampling. Together they are the compute-bearing
components of a transformer layer: the attention core, its projection GEMMs,
RMS/LayerNorm, the SwiGLU MLP nonlinearity, and rotary positional encoding,
each with a forward and a backward pass. The
[unified roofline](#roofline-at-a-glance) places all of them on one chart
against the ceiling of their own regime — the memory-bound kernels reach
roughly 100% of the 250 GB/s bandwidth roof, and the two compute-bound GEMMs
reach 92% and 80% of the FP32- and FP16-accumulate Tensor-core roofs.

## Contents

- [Why this project exists](#why-this-project-exists)
- [Implemented kernels](#implemented-kernels)
- [Measured RTX 4060 results](#measured-rtx-4060-results)
  - [Roofline at a glance](#roofline-at-a-glance)
  - [Consolidated results](#consolidated-results)
  - [Highlight: the FP16-accumulate speedup is 1.72×, not 2×](#highlight-the-fp16-accumulate-speedup-is-172-not-2)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Testing and quality](#testing-and-quality)
- [Benchmark methodology](#benchmark-methodology)
- [Repository structure](#repository-structure)
- [Known limitations](#known-limitations)
- [Contributing](#contributing)
- [License and contact](#license-and-contact)

## Why this project exists

Short Triton tutorials demonstrate syntax. They rarely demonstrate the
engineering required to trust and maintain a GPU kernel:

- edge-case correctness on real silicon;
- dtype-aware numerical policies;
- allocation-free timing;
- empirical bandwidth and compute calibration;
- transparent baseline selection;
- autotune metadata and hardware provenance;
- versioned JSON and generated plots.

Triton Kernel Lab makes those concerns first-class. A kernel is registered
once through `KernelSpec`; correctness tests, benchmarks, roofline metrics,
and plots then consume the same contract.

## Implemented kernels

| Kernel | Main concepts | Primary metric |
|---|---|---|
| `vector_add` | masked tails, arbitrary 1D strides | effective GB/s |
| `relu` / `gelu` / `silu` / `tanh` | FP32 nonlinear math, deterministic elementwise backward | bandwidth roofline |
| `fused_softmax` | reduction, `-inf` tail sentinel, FP32 reduction | bandwidth roofline |
| `matmul_fp32acc` | Tensor Cores, L2 grouping, autotune, FP32 accumulate | % of cuBLAS |
| `matmul_fp16acc` | FP16 Tensor Core accumulation and accuracy trade-off | % theoretical |
| `layer_norm_forward` | FP32 statistics, autograd, lock-reduced backward | bandwidth roofline |
| `rms_norm_forward` | FP32 RMS statistics, autograd, single-buffer lock reduction | bandwidth roofline |
| `residual_rms_norm_forward` | fused residual stream update, two-output autograd | bandwidth roofline |
| `swiglu_forward` | stable FP32 sigmoid, deterministic gated backward | bandwidth roofline |
| `rope_forward` | rotate-half convention, inverse-rotation backward | bandwidth roofline |
| `attention_noncausal` / `attention_causal` | online softmax, tiled Q/K/V, causal staging, recompute backward | TFLOP/s vs SDPA |

Kernel-specific analysis is available in:

- [Elementwise activations](docs/activations.md)
- [Fused softmax](docs/fused_softmax.md)
- [Matrix multiplication](docs/matmul.md)
- [LayerNorm with backward](docs/layer_norm.md)
- [RMSNorm with backward](docs/rms_norm.md)
- [Fused residual addition and RMSNorm](docs/residual_rms_norm.md)
- [SwiGLU](docs/swiglu.md)
- [Rotate-half RoPE](docs/rope.md)
- [Flash Attention forward and backward](docs/flash_attention.md)
- [Benchmark methodology](docs/benchmarking.md)
- [GPU debugging and profiling](docs/debugging.md)
- [Engineering review](docs/code_review.md)

## Measured RTX 4060 results

Environment:

- NVIDIA GeForce RTX 4060 desktop, 8 GB, SM89, 115 W;
- WSL2 Ubuntu 24.04;
- PyTorch 2.12.1 + CUDA 13.0;
- Triton 3.7.1.

### Roofline at a glance

![Unified roofline: every kernel against the measured RTX 4060 ceilings, plotted
on a log-log compute-vs-intensity chart. Memory-bound kernels sit at the 250 GB/s
bandwidth roof; the matmul GEMMs sit at 92% and 80% of the FP32- and
FP16-accumulate Tensor-core roofs.](results/nvidia_geforce_rtx_4060/roofline.png)

One point per kernel, sourced from the committed benchmark JSON. The vertical
distance from a point to its ceiling is the measured throughput ratio; the
horizontal position is the analytic operational intensity. The chart is
regenerated deterministically by `benchmarks/plot_roofline.py` (two renders
produce one SHA-256). It is the whole result in one frame: every kernel sits
near the ceiling of the regime it was optimized for.

### Consolidated results

Representative committed results:

| Measurement | Result |
|---|---:|
| empirical memory bandwidth | ~250.1 GB/s |
| fused softmax bandwidth utilization | ~100% |
| fused softmax vs `torch.softmax`, FP16/N=4096 | ~1.31× |
| fused softmax vs naive multi-pass baseline | ~1.8–4.5× |
| cuBLAS FP16 input / FP32 accumulate, N=2048 | ~30.8 TFLOP/s |
| Triton matmul FP32 accumulate, N=2048 | ~30.8 TFLOP/s |
| Triton matmul FP16 accumulate, N=2048 | ~53.1 TFLOP/s |
| NCU Tensor ops-to-peak ratio, FP32 / FP16 acc | 95.73% / 89.91% |
| NCU SM throughput, FP32 / FP16 acc | 47.38% / 86.90% |
| LayerNorm forward vs native PyTorch, FP16 | ~1.5–2.3× |
| Flash Attention forward vs PyTorch SDPA | ~0.78–0.96× |
| Flash vs materialized attention at N=16384, WDDM footprint | 32 MiB vs 16.0 GiB |

Matmul rows are at N=2048, the single shape where both accumulation modes are
directly compared and where the FP32-accumulate kernel matches cuBLAS; the FP32
kernel's best shape reaches 32.0 TFLOP/s at N=4096 (95.9% of its roof). Full
per-shape curves are in the [matmul analysis](docs/matmul.md#rtx-4060-result).

These are observations from one machine, not portable promises. Inspect
[`results/nvidia_geforce_rtx_4060/`](results/nvidia_geforce_rtx_4060/) for
full curves, clocks, versions, and autotune winners.

### Highlight: the FP16-accumulate speedup is 1.72×, not 2×

The Ada Tensor cores run FP16-input/FP16-accumulate GEMM at twice the rate of
FP16-input/FP32-accumulate, so the silicon ceiling is exactly 2× — the 66.72
and 33.36 TFLOP/s roofs in the figure above. The kernel does not realize 2×. At
N=2048, the shape where both modes are compared:

- FP32 accumulate: 30.85 TFLOP/s, **92.5%** of its 33.36 TFLOP/s roof;
- FP16 accumulate: 53.09 TFLOP/s, **79.6%** of its 66.72 TFLOP/s roof.

End to end that is **1.72×**, not 2×, and the arithmetic closes exactly:

```text
1.72× = 2× × (79.6% / 92.5%)
```

The 2× is the hardware; the `79.6% / 92.5%` is the realization gap. FP32
accumulate sits closer to its lower ceiling than FP16 accumulate sits to its
higher one, so doubling the math rate does not double delivered throughput.
The shortfall is not codegen: PTX confirms the two variants select different
MMA instructions (`...f32.f16.f16.f32` vs `...f16.f16.f16.f16`), so `out_dtype`
is propagating correctly.

Nsight Compute localizes the limit. Switching to FP16 accumulate:

- drives SM throughput from 47.38% to **86.90%** — the binding counter;
- leaves DRAM throughput at **21.63%** (from 11.79%) — memory bandwidth is
  nowhere near a wall;
- keeps the Tensor pipe active only **44.95%** of cycles (from 23.93%) — the
  Tensor cores are under-fed, not saturated;
- holds active-warp occupancy flat at ~32%, ruling out occupancy.

So the limit is on-chip, not DRAM starvation: at the doubled math rate SM
throughput becomes the binding counter at 86.9% — consumed by the non-Tensor
work of feeding the cores, such as operand staging, shared-memory movement, and
address generation — while DRAM bandwidth idles at 22%. FP16 accumulate is
SM-throughput-bound well before it is DRAM-bound on this part, which caps it at
79.6% of the doubled ceiling. Pinning the exact SM sub-pipe would need
scheduler-stall and cache-level profiling; what the counters establish is the
binding resource — SM throughput, not DRAM bandwidth. A separate NCU counter agrees on the direction — the mode-specific
Tensor-operations-to-peak ratio falls from 95.73% to 89.91%, so FP16 accumulate
realizes a smaller fraction of its instruction peak — but that ratio uses a
different denominator than the roofline, so it corroborates the direction rather
than carrying the 1.72×; the roofline pair (79.6% / 92.5%) carries the number.

This is why the 1.72× is a property of the workload, not a defect: FP16
accumulation is an inference-oriented throughput mode that trades mantissa bits
for rate, and the rate it delivers here is bounded by on-chip operand feed, not
by the 2× math headline. The
[matmul analysis](docs/matmul.md#rtx-4060-result) carries the full counter table
and the PTX disassembly.

## Architecture

```text
KernelSpec
├── correctness inputs and numerical assertion
├── allocation-free Triton launcher
├── high-precision PyTorch reference
├── benchmark sizes and cost model
├── adversarial tail/stride case
└── optional autotune metadata
        │
        ├── pytest GPU/interpreter suites
        ├── benchmark harness
        ├── empirical roofline calibration
        └── JSON + PNG artifacts
```

This design keeps test and benchmark behavior synchronized. Adding a kernel
does not require duplicating loops across scripts.

## Requirements

- Linux or WSL2. Native Windows execution is not supported.
- NVIDIA GPU supported by Triton.
- Recent NVIDIA driver with CUDA passthrough.
- Python 3.10–3.14.
- [`uv`](https://docs.astral.sh/uv/).
- A C compiler and Python development headers if using a system Python.
  `uv python install` provides a managed Python with headers.

Check the GPU before installation:

```bash
nvidia-smi
```

## Installation

Clone and create a reproducible environment:

```bash
git clone https://github.com/v1olegrace/triton-kernel-lab.git
cd triton-kernel-lab

uv python install 3.12.13
uv sync --extra dev --frozen
```

Verify the stack:

```bash
uv run --frozen python -c \
  "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name())"
```

Install pre-commit hooks:

```bash
uv run --frozen pre-commit install
```

### WSL2 note

Keep the virtual environment on the native Linux filesystem when the Windows
drive is constrained or when copying large CUDA libraries through `/mnt/c`
is unreliable:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/triton-kernel-lab"
uv sync --extra dev --frozen
```

### Docker

A pinned CUDA image reproduces the documented stack without hand-assembling a
WSL2 + driver + `uv` environment. The NVIDIA driver is supplied by the host
through the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit);
only the CUDA toolkit lives in the image.

```bash
make docker-build                 # build the pinned image
make docker-test-gpu              # run the real-GPU correctness suite
docker run --rm triton-kernel-lab make all   # static checks + CPU layers, no GPU
```

See [docs/docker.md](docs/docker.md) for the layer layout and caching notes.

## Usage

### Python API

```python
import torch

from tklab.kernels.fused_softmax import softmax
from tklab.kernels.layer_norm import layer_norm
from tklab.kernels.matmul import matmul_fp32acc
from tklab.kernels.residual_rms_norm import residual_rms_norm
from tklab.kernels.rope import rope
from tklab.kernels.rms_norm import rms_norm
from tklab.kernels.swiglu import swiglu
from tklab.kernels.vector_add import vector_add

x = torch.randn(1_000_000, device="cuda", dtype=torch.float16)
y = torch.randn_like(x)
z = vector_add(x, y)

scores = torch.randn(4096, 1024, device="cuda", dtype=torch.float16)
probabilities = softmax(scores)

a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = matmul_fp32acc(a, b)

weight = torch.ones(2048, device="cuda", dtype=torch.float16, requires_grad=True)
bias = torch.zeros_like(weight, requires_grad=True)
normalized = layer_norm(a, weight, bias)
rms_normalized = rms_norm(a, weight)
residual_sum, fused_normalized = residual_rms_norm(a, b, weight)
gated = swiglu(a, b)

angles = torch.randn(a.shape[0], a.shape[1] // 2, device="cuda")
rotated = rope(a, torch.cos(angles).half(), torch.sin(angles).half())
```

### Benchmark CLI

List kernels:

```bash
uv run --frozen tklab-bench --list-kernels
```

Benchmark one kernel using an existing peak cache:

```bash
uv run --frozen tklab-bench --kernel fused_softmax
```

The CLI refuses to benchmark when pre-existing GPU utilization exceeds 10%.
Close games, animated wallpapers, and other GPU workloads before producing
reference artifacts. `--allow-busy-gpu` exists only for intentional diagnostic
runs.

Recalibrate rooflines and benchmark both matmul modes:

```bash
uv run --frozen tklab-bench \
  --force-peaks \
  --kernel matmul_fp32acc \
  --kernel matmul_fp16acc
```

Benchmark LayerNorm forward and its two backward stages:

```bash
uv run --frozen tklab-bench --kernel layer_norm_forward
uv run --frozen python benchmarks/layer_norm_backward.py
```

Benchmark RMSNorm forward and reproduce its lock stress:

```bash
uv run --frozen tklab-bench --kernel rms_norm_forward
uv run --frozen python benchmarks/rms_norm_lock_stress.py
```

Benchmark RMSNorm and residual RMSNorm together during one clean GPU session:

```bash
uv run --frozen tklab-bench \
  --kernel rms_norm_forward \
  --kernel residual_rms_norm_forward
```

The final portfolio benchmark should run the complete registry once after a
clean boot with GPU applications closed:

```bash
uv run --frozen tklab-bench --force-peaks
```

The CLI writes versioned JSON and PNG files under `results/<gpu_slug>/`.

## Testing and quality

```bash
make lint              # Ruff lint + formatting check
make type              # mypy --strict over src, tests, and scripts
make test-unit         # CPU-only unit tests
make test-interpreter  # Triton interpreter edge cases
make test-gpu          # real CUDA correctness tests
make coverage          # unit-test coverage report
make all               # lint + type + CPU tests
```

Apply automatic formatting:

```bash
make format
```

GPU correctness includes non-power-of-two tails, non-contiguous row/element
strides, and the adversarial `129×193 @ 193×257` matmul.
Focused Compute Sanitizer workloads, the LayerNorm and RMSNorm
lock-contention studies, and Nsight Compute commands are documented in
[docs/debugging.md](docs/debugging.md).

## Benchmark methodology

- Timings use `triton.testing.do_bench`, CUDA events, warmup, and L2 flushing.
- Triton and PyTorch production baselines reuse preallocated outputs when the
  native reference exposes an `out` contract.
- PyTorch 2.12 has no `aten.rms_norm.out`; the RMSNorm documentation and
  benchmark interpretation explicitly retain this allocation-matching
  limitation.
- RMSNorm and residual RMSNorm record both their native `F.rms_norm`
  composition and a deliberately decomposed PyTorch baseline.
- Public LayerNorm, RMSNorm, residual RMSNorm, and attention wrappers skip
  backward-only statistics when gradients are disabled or no input requires
  gradients.
- Softmax also measures a deliberately naive multi-pass baseline.
- Memory cost models depend on dtype.
- Compute calibration warms the GPU to steady state and records SM clocks.
- Measured cuBLAS throughput and theoretical silicon ceilings remain separate.
- Matmul reports same-size cuBLAS efficiency separately from global cuBLAS
  peak utilization.
- FP16/FP16-accumulate is not mislabeled as a cuBLAS baseline when PyTorch
  does not expose that mode.
- JSON writes are atomic.

See [docs/benchmarking.md](docs/benchmarking.md) for details.

## Repository structure

```text
.
├── benchmarks/
│   ├── flash_attention_memory.py
│   ├── layer_norm_backward.py  # two-stage backward study
│   ├── layer_norm_lock_stress.py
│   ├── profile_matmul.py
│   ├── rms_norm_lock_stress.py # single-buffer lock validation
│   └── run.py                  # compatibility entry point
├── docs/
│   ├── activations.md
│   ├── benchmarking.md
│   ├── code_review.md
│   ├── debugging.md
│   ├── docker.md
│   ├── flash_attention.md
│   ├── fused_softmax.md
│   ├── layer_norm.md
│   ├── matmul.md
│   ├── residual_rms_norm.md
│   ├── rms_norm.md
│   ├── rope.md
│   └── swiglu.md
├── results/
│   └── <gpu_slug>/             # committed JSON and PNG evidence
├── src/tklab/
│   ├── cli.py                  # installed tklab-bench command
│   ├── registry.py             # KernelSpec and global registry
│   ├── harness/
│   │   ├── addressing.py
│   │   ├── bench.py
│   │   ├── jsonio.py
│   │   ├── plots.py
│   │   ├── roofline.py
│   │   └── tolerances.py
│   └── kernels/
│       ├── _elementwise_math.py
│       ├── _norm_common.py
│       ├── activations.py
│       ├── flash_attention.py
│       ├── fused_softmax.py
│       ├── layer_norm.py
│       ├── matmul.py
│       ├── residual_rms_norm.py
│       ├── rope.py
│       ├── rms_norm.py
│       ├── swiglu.py
│       └── vector_add.py
├── tests/
├── .python-version
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE
├── SECURITY.md
└── pyproject.toml
```

## Known limitations

- The audited theoretical compute profile currently supports RTX 4060 only.
  Other GPUs require a sourced `GpuTheoreticalProfile`.
- The fused softmax supports contiguous columns and up to 65,536 columns.
- Matmul currently accepts FP16 inputs only.
- LayerNorm requires a contiguous final dimension and a feature row no larger
  than 64 KiB. FP16 backward is not claimed as training-grade numerics, and
  higher-order gradients are not supported. Its lock-reduced parameter
  gradients are numerically stable but not bitwise deterministic; backward
  rejects PyTorch deterministic mode.
- RMSNorm has the same 64 KiB row limit and first-order autograd scope. Its
  single-buffer parameter-gradient reduction is likewise non-deterministic
  and rejects deterministic mode.
- Residual RMSNorm shares those limits and returns `(residual_sum,
  normalized)`. Its normalized-output backward is non-deterministic, while a
  residual-sum-only backward bypasses the lock reduction.
- SwiGLU and RoPE currently accept 2D tensors with contiguous final
  dimensions. RoPE uses half-dimension angle tables and requires an even
  feature width. Their custom backward implementations support first-order
  gradients only.
- The standalone ReLU/GELU/SiLU/tanh kernels support first-order gradients
  only.
- Flash Attention supports contiguous FP16 Q/K/V with `head_dim=64` and
  specializes per compile-time sequence length. Forward and a recompute-based
  backward (`dQ`/`dK`/`dV`, causal and non-causal) are implemented. Dropout,
  attention bias, grouped-query attention, variable-length production
  bucketing, larger head dimensions, and higher-order gradients remain out of
  scope.
- The contiguous matmul fast path matches the measured cuBLAS peak in FP32
  accumulation on this RTX 4060. Split-K atomics regressed; persistent/stream-K
  designs and profiler evidence remain future work for broader shapes.
- GitHub-hosted CI has no GPU. Real-GPU tests and benchmark artifacts are run
  locally; CI covers static checks, CPU unit tests, and Triton interpreter
  cases.
- Triton's interpreter does not model real GPU program scheduling or
  inter-program atomic ordering. Lock-protocol claims require real-GPU stress
  and sanitizer evidence.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). Every kernel contribution must
include:

- high-precision correctness reference;
- adversarial tail/stride case;
- explicit numerical policy;
- byte or FLOP cost model;
- benchmark and hardware provenance;
- documentation of unsupported inputs.

## License and contact

Licensed under the [MIT License](LICENSE).

- Maintainer: Mauro de Oliveira Cardoso
- Email: maurulycan@gmail.com
- GitHub: <https://github.com/v1olegrace>

## Credits

Built with [Triton](https://triton-lang.org/),
[PyTorch](https://pytorch.org/), and
[uv](https://docs.astral.sh/uv/). Hardware-theory references are linked in
the kernel documentation and committed peak metadata.
