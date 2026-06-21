# Contributing

Contributions are welcome when they preserve the laboratory's central
contract: every kernel must ship with correctness coverage, an explicit cost
model, reproducible benchmark metadata, and documented numerical assumptions.

## Development setup

```bash
git clone https://github.com/v1olegrace/triton-kernel-lab.git
cd triton-kernel-lab
uv python install 3.12.13
uv sync --extra dev --frozen
uv run --frozen pre-commit install
```

Use Linux or WSL2. Native Windows Triton execution is not supported.

## Before opening a pull request

```bash
make format
make lint
make type
make test-unit
make test-interpreter
make test-gpu
```

GPU tests and benchmarks must include the GPU model, driver, PyTorch version,
Triton version, and generated JSON artifacts.

## Adding a kernel

1. Add a module under `src/tklab/kernels/`.
2. Define one or more `KernelSpec` instances.
3. Import the specs from `src/tklab/kernels/__init__.py`.
4. Provide an adversarial input covering tails and relevant strides.
5. Define bytes moved or FLOPs explicitly.
6. Document numerical tolerances and unsupported layouts.
7. Run correctness tests before committing benchmark results.
8. Autograd kernels must compare every returned gradient with a PyTorch
   reference on real GPU and document reduced-precision limitations.

## Commit and review expectations

- Keep commits focused and explain performance-sensitive design decisions.
- Do not claim speedups without a named and fairly measured baseline.
- Do not compare a measured result with an unaudited theoretical number.
- Preserve unrelated user changes and generated results.

By contributing, you agree that your work is licensed under the MIT License.
