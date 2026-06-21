# RMSNorm with backward

The RMSNorm implementation reuses the audited LayerNorm architecture while
removing every mean-centering dependency: no saved mean, no bias gradient,
and one lock-reduced parameter-gradient buffer.

## Forward

One Triton program owns one row. The mean square, reciprocal RMS, and affine
scaling use FP32 intermediates:

```text
rstd = 1 / sqrt(mean(x^2) + eps)
y = x * rstd * weight
```

The forward stores one FP32 `rstd` value per row for backward. Inputs may use
arbitrary positive row strides, but the final dimension must be contiguous.
Non-power-of-two widths are masked, and one feature row may occupy at most
64 KiB.

PyTorch 2.12 exposes `aten.rms_norm` but no `aten.rms_norm.out` overload.
Consequently, the native PyTorch benchmark baseline returns a newly allocated
output while the Triton launcher reuses preallocated output and statistic
buffers. This baseline limitation is documented explicitly; the comparison
must not be described as strictly allocation-matched.

## Backward formula

For each row, define:

```text
x_hat = x * rstd
g = weight * dy
```

Then:

```text
dx = rstd * (g - x_hat * mean(g * x_hat))
dweight = sum_rows(dy * x_hat)
```

This is equivalent to:

```text
dx = rstd * (g - x * rstd^2 * mean(g * x))
```

Unlike LayerNorm, RMSNorm does not subtract the input mean in forward.
Therefore `dx` has no `mean(g)` term.

The custom `torch.autograd.Function` saves `(x, weight, rstd)` and returns
`(dx, dweight, None)`. It implements first-order gradients only.

## Single-buffer lock reduction

Backward stage 1 computes `dx` and accumulates each row's `dweight` partial
into one of a bounded number of FP32 buffers. The synchronization allocation
contains two logical halves:

```text
locks[0:group_size]
counts[group_size:2*group_size]
```

The lock half serializes programs assigned to the same partial buffer. The
count half identifies the first writer, which initializes rather than reads
the uninitialized partial. Dropping LayerNorm's `dbias` buffer does not alter
this count protocol.

Acquisition and release use explicit `acq_rel` atomics at GPU scope.
`tl.debug_barrier` joins vector stores from all lanes before the unlock
publishes the completed partial. Stage 2 reduces only the
`(group_count, columns)` `dweight` buffer.

## Correctness and contention

The ordinary GPU tests cover:

- FP32 forward against `torch.nn.functional.rms_norm`;
- `dx` and `dweight` against PyTorch autograd;
- independent row-strided `x` and `dy`;
- a masked width of 1000;
- strict and warn-only deterministic-algorithm policies.

A dedicated test sends exactly 1,025 rows through one lock for ten runs. After
every stage-1 launch it directly asserts `[lock, count] == [0, 1]` before
checking `dweight`.

The longer stress study runs 50 repetitions at group sizes 1, 8, 256, and
2048:

```bash
uv run python benchmarks/rms_norm_lock_stress.py
```

On the RTX 4060, all 200 stage-1 launches released every lock, initialized
every active count, and left every inactive count at zero. The most contended
case assigns 1,025 rows to one lock slot. Its relative Frobenius error against
the PyTorch FP32 reference was approximately `6.00e-7`, with maximum
run-to-run drift of `7.39e-7`.

Bitwise identity is not expected because lock acquisition changes the order
of FP32 additions. Large reference error, large repeated drift, group-size
dependence, or invalid lock/count state would indicate a protocol failure.
The complete values and thresholds are stored in
`rms_norm_lock_stress.json`.

## Determinism and numerical scope

The lock reduction is numerically stable but not bitwise deterministic.
Backward rejects `torch.use_deterministic_algorithms(True)` and emits a
warning under `warn_only=True`.

FP32 is the validation mode for backward. FP16 forward is the performance
mode. FP16 backward is available but is not presented as a general
training-grade numerical guarantee.

## Compute Sanitizer

The dedicated row-strided forward/backward workload passes Compute Sanitizer
2026.2 `memcheck`, `initcheck`, `racecheck`, and `synccheck` with zero errors
or warnings. As with LayerNorm, `racecheck` does not independently prove the
inter-program global-memory lock protocol; the repeated real-GPU stress study
provides that complementary evidence.
