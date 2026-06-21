# Reproducible CUDA environment for triton-kernel-lab.
#
# The image pins the documented stack (CUDA 13.0 toolkit, managed Python 3.12.13,
# uv-resolved torch/triton from uv.lock) so a contributor can reproduce the
# correctness suite without hand-assembling a WSL2 + driver + uv environment.
#
# The NVIDIA driver is provided by the host at run time; only the CUDA toolkit
# (ptxas, headers) lives in the image. Build and run with GPU passthrough:
#
#   docker build -t triton-kernel-lab .
#   docker run --rm --gpus all triton-kernel-lab uv run --frozen pytest -m gpu -q
#
# Without a GPU you can still run the static checks and CPU layers:
#
#   docker run --rm triton-kernel-lab make all
#
# The devel base ships nvcc/ptxas so Triton can JIT-compile kernels. torch and
# triton themselves come from the locked wheels, not from the base image.

FROM ghcr.io/astral-sh/uv:0.11.23@sha256:d0a0a753ab981624b49c97abc98821c1c09f4ca69d1ef5cee69c501be3d88479 AS uv

FROM nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92

# Fail fast and keep pipelines honest inside RUN layers.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    # Keep the resolved environment inside the image, never on a bind mount.
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    # uv copies wheels instead of hardlinking across the image/cache boundary.
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH

# git is required by some build backends; build-essential gives Triton a host
# compiler. ca-certificates lets uv fetch the managed Python and wheels.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        git \
        make \
    && rm -rf /var/lib/apt/lists/*

# Install the exact uv release used by CI and local project metadata.
COPY --from=uv /uv /uvx /usr/local/bin/

RUN useradd --create-home --uid 10001 tklab

WORKDIR /workspace

# Resolve and install third-party dependencies first so the heavy layer is
# cached independently of source edits. README.md is referenced by pyproject's
# readme field and must exist for metadata parsing.
COPY pyproject.toml uv.lock README.md ./
RUN uv python install 3.12.13 \
    && uv sync --extra dev --no-install-project --frozen

# Install the project itself against the already-resolved environment.
COPY --chown=tklab:tklab . .
RUN uv sync --extra dev --frozen

# Run tests and JIT compilation without root privileges. The writable home is
# also where Triton stores its compilation cache.
RUN chown tklab:tklab /opt/venv
USER tklab
ENV HOME=/home/tklab

# Default to the full static + CPU quality gate; override with `uv run --frozen pytest
# -m gpu` (and `--gpus all`) for the real-GPU correctness suite.
CMD ["make", "all"]
