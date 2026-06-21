# Fused residual addition and RMSNorm

This kernel fuses the residual addition that advances a transformer residual
stream with RMSNorm of the resulting tensor. It returns both values because
the residual sum is consumed by the next block while the normalized output is
consumed by the current sublayer.

## Forward

For each row:

```text
s = x + residual
rstd = 1 / sqrt(mean(s^2) + eps)
y = s * rstd * weight
```

The public API returns:

```python
residual_sum, normalized = residual_rms_norm(x, residual, weight)
```

One Triton program loads both input rows once and forms `s`. For FP16 inputs,
the sum is rounded to FP16 before being converted back to FP32 for the RMS
statistic; therefore the kernel normalizes the exact same materialized `s`
that it returns. It then stores both `s` and `y`. The final dimensions of `x`
and `residual` must be contiguous, but their row strides may differ. Masking
supports non-power-of-two widths. The same 64 KiB feature-row limit as
RMSNorm applies.

A variant that retained the addition in FP32 before normalization could be
more accurate in isolation, but it would not be equivalent to the drop-in
contract where `x + residual` is materialized in FP16 before
`F.rms_norm`. This implementation chooses semantic equivalence with the
materialized residual stream; the alternative is a different numerical
design, not a correction to this kernel.

## Backward

Let `dy` be the gradient of the normalized output and `ds_incoming` the direct
gradient of the returned residual sum. RMSNorm first computes:

```text
x_hat = s * rstd
g = weight * dy
ds_norm = rstd * (g - x_hat * mean(g * x_hat))
```

The complete gradient is:

```text
ds = ds_norm + ds_incoming
dx = ds
dresidual = ds
dweight = sum_rows(dy * x_hat)
```

The equality of both input gradients follows directly from:

```text
ds/dx = 1
ds/dresidual = 1
```

The implementation does not duplicate RMSNorm's synchronization protocol.
It extends the audited stage-1 kernel with a compile-time optional incoming
gradient load. RMSNorm compiles that path out; residual RMSNorm adds
`ds_incoming` before the same final `dx` store. The single-buffer lock
reduction and stage-2 `dweight` reduction are shared unchanged.

When the normalized output is unused, autograd receives no `dy`, bypasses the
lock reduction, and returns the direct residual-sum gradient unchanged. This
path remains valid under PyTorch deterministic mode. When `dy` is present,
the usual non-deterministic lock-reduction policy applies.

## Validation

The real-GPU suite covers:

- both forward outputs against `x + residual` and
  `torch.nn.functional.rms_norm`;
- bitwise equality of the returned FP16 sum with native `x + residual`;
- combined gradients from both returned tensors;
- exact equality of `dx` and `dresidual`;
- the residual-sum-only path under deterministic mode;
- independent row strides for `x`, `residual`, `ds_incoming`, and `dy`;
- width 1000 masked tails;
- 1,025 rows serialized through one lock for ten repeated launches;
- direct verification of `[lock, count] == [0, 1]`.

Compute Sanitizer 2026.2 `memcheck`, `initcheck`, `racecheck`, and `synccheck`
all report zero errors or warnings for the fused forward/backward workload.
The longer RMSNorm lock study remains the primary repeated validation of the
shared global-memory protocol.

## Benchmark methodology

The registered cost model counts the dominant tensor traffic:

```text
read x + read residual + write s + write y
= 4 * rows * columns * element_size
```

The benchmark records two comparisons:

1. native composition: `torch.add` followed by
   `torch.nn.functional.rms_norm`;
2. deliberately naive composition: separate add, FP32 pointwise operations,
   reduction, reciprocal square root, scaling, and cast.

PyTorch 2.12 has no `aten.rms_norm.out`, so both PyTorch baselines allocate.
The Triton timed region reuses preallocated `s`, `y`, and `rstd` buffers. This
allocation difference is recorded in row metadata and must be considered
when interpreting small-width speedups.

Run RMSNorm and residual RMSNorm in one clean GPU session:

```bash
uv run --frozen tklab-bench \
  --kernel rms_norm_forward \
  --kernel residual_rms_norm_forward
```

The CLI refuses pre-existing GPU utilization above 10%. No performance number
is claimed until that clean run produces committed JSON and plots.
