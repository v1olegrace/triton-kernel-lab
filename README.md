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

## Contents

- [Why this project exists](#why-this-project-exists)
- [Implemented kernels](#implemented-kernels)
- [Measured RTX 4060 results](#measured-rtx-4060-results)
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
| `fused_softmax` | reduction, `-inf` tail sentinel, FP32 reduction | bandwidth roofline |
| `matmul_fp32acc` | Tensor Cores, L2 grouping, autotune, FP32 accumulate | % of cuBLAS |
| `matmul_fp16acc` | FP16 Tensor Core accumulation and accuracy trade-off | % theoretical |
| `layer_norm_forward` | FP32 statistics, autograd, lock-reduced backward | bandwidth roofline |

Kernel-specific analysis is available in:

- [Fused softmax](docs/fused_softmax.md)
- [Matrix multiplication](docs/matmul.md)
- [LayerNorm with backward](docs/layer_norm.md)
- [Benchmark methodology](docs/benchmarking.md)
- [GPU debugging and profiling](docs/debugging.md)
- [Engineering review](docs/code_review.md)

## Measured RTX 4060 results

Environment:

- NVIDIA GeForce RTX 4060 desktop, 8 GB, SM89, 115 W;
- WSL2 Ubuntu 24.04;
- PyTorch 2.12.1 + CUDA 13.0;
- Triton 3.7.1.

Representative committed results:

| Measurement | Result |
|---|---:|
| empirical memory bandwidth | ~247.8 GB/s |
| fused softmax bandwidth utilization | ~100% |
| fused softmax vs `torch.softmax`, FP16/N=4096 | ~1.32× |
| fused softmax vs naive multi-pass baseline | ~1.8–6.4× |
| cuBLAS FP16 input / FP32 accumulate | ~30.9 TFLOP/s |
| Triton matmul FP32 accumulate | ~31.0 TFLOP/s |
| Triton matmul FP16 accumulate | ~53.1 TFLOP/s |
| NCU Tensor-path utilization, FP32 / FP16 accumulate | 95.73% / 89.91% |
| LayerNorm forward vs native PyTorch, FP16 | ~1.5–2.3× |

These are observations from one machine, not portable promises. Inspect
[`results/nvidia_geforce_rtx_4060/`](results/nvidia_geforce_rtx_4060/) for
full curves, clocks, versions, and autotune winners.

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

uv python install 3.12
uv sync --extra dev
```

Verify the stack:

```bash
uv run python -c \
  "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name())"
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

### WSL2 note

Keep the virtual environment on the native Linux filesystem when the Windows
drive is constrained or when copying large CUDA libraries through `/mnt/c`
is unreliable:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/triton-kernel-lab"
uv sync --extra dev
```

## Usage

### Python API

```python
import torch

from tklab.kernels.fused_softmax import softmax
from tklab.kernels.layer_norm import layer_norm
from tklab.kernels.matmul import matmul_fp32acc
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
```

### Benchmark CLI

List kernels:

```bash
uv run tklab-bench --list-kernels
```

Benchmark one kernel using an existing peak cache:

```bash
uv run tklab-bench --kernel fused_softmax
```

The CLI refuses to benchmark when pre-existing GPU utilization exceeds 10%.
Close games, animated wallpapers, and other GPU workloads before producing
reference artifacts. `--allow-busy-gpu` exists only for intentional diagnostic
runs.

Recalibrate rooflines and benchmark both matmul modes:

```bash
uv run tklab-bench \
  --force-peaks \
  --kernel matmul_fp32acc \
  --kernel matmul_fp16acc
```

Benchmark LayerNorm forward and its two backward stages:

```bash
uv run tklab-bench --kernel layer_norm_forward
uv run python benchmarks/layer_norm_backward.py
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
Focused Compute Sanitizer workloads, the LayerNorm lock-contention study, and
Nsight Compute commands are documented in
[docs/debugging.md](docs/debugging.md).

## Benchmark methodology

- Timings use `triton.testing.do_bench`, CUDA events, warmup, and L2 flushing.
- Triton and PyTorch production baselines reuse preallocated outputs.
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
│   ├── layer_norm_backward.py  # two-stage backward study
│   └── run.py                  # compatibility entry point
├── docs/
│   ├── benchmarking.md
│   ├── code_review.md
│   ├── debugging.md
│   ├── fused_softmax.md
│   ├── layer_norm.md
│   └── matmul.md
├── results/
│   └── <gpu_slug>/             # committed JSON and PNG evidence
├── src/tklab/
│   ├── cli.py                  # installed tklab-bench command
│   ├── registry.py             # KernelSpec and global registry
│   ├── harness/
│   │   ├── bench.py
│   │   ├── jsonio.py
│   │   ├── plots.py
│   │   ├── roofline.py
│   │   └── tolerances.py
│   └── kernels/
│       ├── fused_softmax.py
│       ├── layer_norm.py
│       ├── matmul.py
│       └── vector_add.py
├── tests/
├── CONTRIBUTING.md
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
