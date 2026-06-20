# Engineering review

This document records the repository-wide review performed before the
portfolio refactor.

## Strengths found

- A single `KernelSpec` registry already connected correctness and benchmark
  execution.
- Timed Triton launchers reused output buffers instead of measuring allocator
  overhead.
- Shapes and strides were runtime values rather than accidental JIT
  specialization keys.
- Numerical references used float32 computation.
- Adversarial GPU cases covered masks, tails, and non-contiguous layouts.
- Roofline artifacts preserved raw samples, clocks, versions, and source URLs.
- Benchmark conclusions documented underperformance instead of hiding it.

## Problems found

- Triton was imported without being a direct project dependency.
- NumPy and pandas were declared but unused.
- README, license, contribution, security, and benchmark-method documents were
  missing or incomplete.
- CI and `make test` could succeed with skipped tests and had no CPU unit-test
  layer.
- Public and internal APIs lacked consistent docstrings.
- JSON writes were non-atomic and cache reads trusted unchecked casts.
- CLI, serialization, plotting, baseline selection, and benchmark execution
  were coupled in one script.
- PyTorch benchmark references allocated outputs inside timed regions.
- Validation did not consistently reject empty shapes or unsupported dtypes.
- Plot payloads and autotune metadata relied on loose `object` casts.
- A text-encoding defect appeared in one plot label.
- The repository had no configured coverage collection.

## Refactoring decisions

- Keep the registry architecture; strengthen its invariants and types.
- Preserve tested Triton kernels while improving wrappers and documentation.
- Add allocation-free PyTorch reference launchers.
- Move benchmark orchestration into an installed CLI.
- Add atomic JSON helpers and schema validation.
- Model theoretical hardware facts with an extensible immutable profile.
- Split CPU unit, Triton interpreter, and real-GPU tests.
- Keep generated benchmark evidence committed and regenerate it through the
  same CLI used by contributors.

## Remaining limitations

- Theoretical compute calibration has an audited profile only for RTX 4060.
- GitHub-hosted CI cannot execute real-GPU tests.
- Matmul performance remains below cuBLAS and needs a structurally different
  kernel or deeper Nsight Compute work rather than more tuning of the current
  design.
- Publishing placeholders in README, LICENSE, badges, and security contact
  must be replaced by the repository owner.
