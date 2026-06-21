# SwiGLU

The SwiGLU kernel implements the common gated transformer activation:

```text
output = value * SiLU(gate)
       = value * gate * sigmoid(gate)
```

Both inputs are 2D tensors with matching shape, dtype, and device. Their row
strides may differ, while the final dimension must be contiguous.

## Stable nonlinear math

The sigmoid is evaluated in FP32 without an overflowing exponential branch.
For `z = gate`:

```text
e = exp(-abs(z))
sigmoid(z) = where(z >= 0, 1 / (1 + e), e / (1 + e))
```

This is algebraically identical to the usual two-branch stable sigmoid while
using only `exp(-abs(z))`. Inputs such as `z = -100` and `z = 100` therefore
remain finite in forward and backward.

## Backward

Let `s = sigmoid(gate)`. Then:

```text
dvalue = dy * gate * s
dgate = dy * value * s * (1 + gate * (1 - s))
```

Backward is entirely elementwise, uses no atomics, and remains valid under
PyTorch deterministic mode.

## Validation

The real-GPU suite covers:

- independent row strides for value, gate, and upstream gradient;
- a width-1000 masked tail;
- forward and both gradients against `value * F.silu(gate)`;
- extreme gates from `-100` through `100`;
- deterministic-mode execution.

Compute Sanitizer 2026.2 reports zero errors or warnings under `memcheck`,
`initcheck`, `racecheck`, and `synccheck`.

## Benchmark methodology

The effective traffic model counts two reads and one output write:

```text
3 * rows * columns * element_size
```

The native baseline is `value * torch.nn.functional.silu(gate)`. A second
baseline explicitly composes the stable FP32 sigmoid and pointwise products.
Both PyTorch baselines allocate; the Triton timed region reuses a preallocated
output.

Dirty diagnostic measurements are intentionally not committed. Official
numbers will be produced with the complete registry in one clean GPU session.

The custom autograd function implements first-order differentiation only.
