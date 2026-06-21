# Rotate-half RoPE

This implementation uses an explicit rotate-half convention. For an input
row split into equal halves:

```text
x = [x1, x2]
rotate_half(x) = [-x2, x1]
y = x * cos + rotate_half(x) * sin
```

The public contract accepts:

```text
x:   (rows, 2 * half_dim)
cos: (rows, half_dim)
sin: (rows, half_dim)
```

Half-dimension tables enforce that each pair uses the same angle. This makes
the orthogonal rotation and its inverse true for every accepted input instead
of relying on callers to duplicate full-width tables correctly.

## Forward

The kernel computes:

```text
y1 = x1 * cos - x2 * sin
y2 = x2 * cos + x1 * sin
```

Inputs and tables may use independent row strides. Final dimensions must be
contiguous, the feature dimension must be even, and masked half-width tails
are supported.

## Backward

The input gradient is the transpose of the forward rotation:

```text
dx1 = dy1 * cos + dy2 * sin
dx2 = dy2 * cos - dy1 * sin
```

Equivalently:

```text
dx = RoPE(dy, cos, -sin)
```

The custom autograd function also returns gradients for the angle tables:

```text
dcos = dy1 * x1 + dy2 * x2
dsin = -dy1 * x2 + dy2 * x1
```

All gradients are elementwise and deterministic. In the common case where
the angle tables are constants, compile-time branches remove the `x` loads
and the `dcos`/`dsin` stores; backward reduces to the inverse rotation.

## Validation

The real-GPU suite covers:

- forward and all three gradients against the explicit PyTorch convention;
- the inverse-rotation identity `sin -> -sin`;
- row-norm preservation for valid sine/cosine pairs;
- independent row strides and a width-1000 masked tail;
- deterministic-mode execution.

Compute Sanitizer 2026.2 reports zero errors or warnings under all four
focused tools.

## Benchmark methodology

The half-width cosine and sine tables together contain one full input width.
The effective traffic model is therefore:

```text
read x + read cos/sin + write output
= 3 * rows * columns * element_size
```

Each output element performs two multiplies and one add/subtract. In FP16,
that is approximately `3 FLOPs / 6 bytes = 0.5 FLOP/byte`, far below this
session's FP16/FP32-accumulate ridge point of 133.37 FLOP/byte. The clean run
reached 246.995 GB/s, or 98.74% of the measured bandwidth ceiling, confirming
the memory-bound classification.

The production-reference baseline uses direct half-wise pairwise formulas.
The deliberately naive baseline materializes full-width cosine/sine tables
and the rotate-half tensor. The accepted complete-registry run is stored in
`results/nvidia_geforce_rtx_4060/rope_forward.json`.

The custom autograd function implements first-order differentiation only.
