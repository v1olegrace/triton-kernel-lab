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

The PyTorch baseline uses the same split, pairwise formulas, and `torch.cat`.
Official performance numbers are deferred to the single clean benchmark
session for the complete registry.

The custom autograd function implements first-order differentiation only.
