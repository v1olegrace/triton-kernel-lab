# Containerized environment

The repository ships a `Dockerfile` that reproduces the documented stack (CUDA
13.0 toolkit, managed Python 3.12.13, uv 0.11.23, and the
`uv.lock`-resolved torch/triton
wheels). It exists so a reviewer can run the correctness suite without manually
assembling a WSL2 + driver + `uv` environment on the host.

## Design

- **Base image:** `nvidia/cuda:13.0.1-devel-ubuntu24.04`, pinned by manifest
  digest. The `devel` variant provides `nvcc`/`ptxas` and CUDA headers so
  Triton can JIT-compile kernels. torch and triton come from the locked wheels,
  not from the base image.
- **Host-provided driver:** the image carries only the CUDA toolkit. The NVIDIA
  driver and `libcuda.so` are injected at run time by the NVIDIA Container
  Toolkit when the container is started with `--gpus all`.
- **Resolved environment in the image:** `UV_PROJECT_ENVIRONMENT=/opt/venv`
  keeps the virtual environment inside the image and off any bind mount, so the
  container is self-contained and does not depend on a host `.venv`.
- **Reproducibility:** `uv sync --frozen` fails rather than silently re-resolving
  if `uv.lock` and `pyproject.toml` disagree.
- **Pinned installer:** uv is copied from
  `ghcr.io/astral-sh/uv:0.11.23`, pinned by manifest digest rather than the
  mutable `latest` tag.

## Layer caching

Dependencies are resolved before the source is copied so source edits do not
invalidate the expensive wheel-download layer:

```dockerfile
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --extra dev --no-install-project --frozen   # third-party deps only
COPY . .
RUN uv sync --extra dev --frozen                        # install the project
```

`README.md` is copied early because `pyproject.toml` references it through the
`readme` field; metadata parsing fails without it.

## Usage

```bash
# Build the pinned image.
make docker-build
# or: docker build -t triton-kernel-lab .

# Real-GPU correctness suite (requires the NVIDIA Container Toolkit on the host).
make docker-test-gpu
# or: docker run --rm --gpus all triton-kernel-lab uv run --frozen pytest -m gpu -q

# Static checks plus the CPU and interpreter layers, no GPU required.
docker run --rm triton-kernel-lab make all
docker run --rm -e TRITON_INTERPRET=1 triton-kernel-lab \
  uv run --frozen pytest -m interpret -q
```

## Notes

- Benchmark artifacts are still produced from a clean host session, not from the
  container. The image targets correctness and the static quality gate; it does
  not attempt to control clocks, thermals, or background GPU contention the way
  [`docs/benchmarking.md`](benchmarking.md) requires.
- The base tag pins CUDA 13.0.1 to match the stack the committed results were
  measured on. Bumping torch/triton in `pyproject.toml` and `uv.lock` may
  require a matching CUDA base tag.
