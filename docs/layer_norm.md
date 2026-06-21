# LayerNorm with backward

The LayerNorm implementation is the first lab kernel with a custom
`torch.autograd.Function` and a reduction across Triton programs.

## Forward

One program owns one row. Values, mean, biased variance, reciprocal standard
deviation, normalization, and affine transformation are computed with FP32
intermediates:

```text
mean = sum(x) / N
variance = sum((x - mean)^2) / N
rstd = 1 / sqrt(variance + eps)
y = (x - mean) * rstd * weight + bias
```

Mean and reciprocal standard deviation are stored as FP32 vectors for
backward. The final dimension must be contiguous; arbitrary positive row
strides and non-power-of-two widths are supported. A feature row may use at
most 64 KiB so one row remains within the intended fused working set.

The forward benchmark uses the allocation-free
`aten.native_layer_norm.out` CUDA baseline. Triton and PyTorch therefore both
receive preallocated output/statistic buffers.

## Backward formula

For each row:

```text
x_hat = (x - mean) * rstd
weighted_dy = weight * dy
dx = rstd * (
    weighted_dy
    - mean(weighted_dy)
    - x_hat * mean(weighted_dy * x_hat)
)
```

The parameter gradients are:

```text
dweight = sum_rows(dy * x_hat)
dbias = sum_rows(dy)
```

FP32 gradient tests compare `dx`, `dweight`, and `dbias` with
`torch.nn.functional.layer_norm` using relative Frobenius error below
`1e-2`. One adversarial problem uses 67 rows, width 1000, and independent
row-strided `x` and `dy` views. A second combines the same non-power-of-two,
row-strided layout with 1,025 rows, forcing up to five updates per default
lock slot.

## Lock-reduced parameter gradients

Backward stage 1 computes `dx` and assigns each row to one of a bounded number
of partial-gradient buffers. Rows sharing a buffer serialize updates through
an `atomic_cas` spinlock. Acquisition and release explicitly use Triton's
`acq_rel` memory semantics at GPU scope. Triton 3.7.1 already maps omitted
atomic semantics to `acq_rel` and omitted scope to `gpu`; spelling them out
makes the synchronization contract reviewable and guards against accidental
weakening.

The lock protects vector loads, additions, and stores. `tl.debug_barrier` is
an intra-program barrier, not the device fence by itself. It remains necessary
to join stores issued by every lane before the device-scoped atomic unlock
publishes the completed vector to the next program acquiring that lock.

Backward stage 2 reduces the `(group_count, N)` FP32 partial buffers into the
final `dweight` and `dbias`. This differs from split-K GEMM: contention is
bounded by deliberate lock groups, each critical section accumulates a full
vector, and the final reduction has a small fixed number of rows.

Reproduce the stage study with:

```bash
uv run --frozen python benchmarks/layer_norm_backward.py
```

The resulting `layer_norm_backward.json` records all three gradient errors,
stage-1 time, standalone stage-2 time, incremental stage-2 overhead in the
real two-kernel sequence, lock-group count, and effective backward bandwidth.
The standalone value includes a separate L2 flush and is not used as the
end-to-end overhead estimate. Stage 1 includes the required reset of the
lock/count array before every repetition; tensor allocation remains outside
the timed region.

## RTX 4060 measurements

With 4096 rows and FP16 data, clean forward measurements range from 1.51x to
2.33x over allocation-free native PyTorch LayerNorm. At widths 8192 and
16384, the kernel sustains about 247.8-248.2 GB/s, effectively the measured
memory roofline.

On the FP32 strided width-1000 validation problem, relative Frobenius errors
were approximately:

- forward: `5.0e-8`;
- `dx`: `1.0e-7`;
- `dweight`: `1.5e-7`;
- `dbias`: `1.3e-7`.

The incremental stage-2 percentage is reported from the measured two-kernel
sequence, not by adding isolated medians. Separate benchmark invocations
flush L2, while the real stage 2 consumes stage-1 partials that remain hot.
Across widths 1024 through 16384, stage 2 adds approximately 4.26% down to
0.99% of the measured two-kernel sequence.

## Numerical scope

FP32 is the correctness reference mode for backward. FP16 forward is the
performance mode. FP16 backward is available through autograd but is not
presented as a training-grade numerical guarantee; reduced-precision
parameter-gradient accumulation requires workload-specific validation.

Welford or online variance is unnecessary while one row fits the 64 KiB fused
limit. Wider or streaming normalization would require an online algorithm and
is outside this implementation.

The lock-reduction structure follows the official Triton LayerNorm tutorial:
<https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html>.
The custom autograd function implements first-order gradients only; higher-
order differentiation is outside the current contract.

## Determinism

The parameter-gradient result is numerically stable but not bitwise
deterministic. Programs can acquire each lock in different orders, and FP32
addition is not associative. The backward therefore rejects
`torch.use_deterministic_algorithms(True)` and emits a warning in PyTorch's
`warn_only=True` mode.

A deterministic implementation would write one partial row per input row and
reduce those partials in a fixed order. That alternative requires an
`M x N` FP32 buffer instead of the bounded lock-group buffer and is outside
the current memory/performance contract.

## Contention validation

The global-lock protocol is stress-tested at `M=65536`, `N=1024` for 50 runs
each with 8, 32, and 128 lock groups. This reaches 8,192 rows per lock slot in
the most contended case. Reference error, run-to-run drift, and group-count
drift all remain below the committed thresholds.

Bitwise identity is not expected because lock acquisition order changes the
order of floating-point additions. See [debugging.md](debugging.md) and the
committed `layer_norm_lock_stress.json` artifact for the exact methodology and
values.
