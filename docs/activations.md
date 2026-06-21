# Elementwise activations

`src/tklab/kernels/activations.py` provides the pointwise activations that round
out the lab's elementwise coverage alongside `vector_add` and the gated
`swiglu`. They share one masked forward kernel and one deterministic,
elementwise backward kernel; a compile-time `ACTIVATION` selector specializes
each function instead of duplicating the launcher.

| Kernel | Definition | Backward derivative | Reference |
|---|---|---|---|
| `relu_forward` | `max(x, 0)` | `1[x > 0]` | `torch.relu` |
| `gelu_forward` | `0.5 x (1 + erf(x / √2))` | `Φ(x) + x·φ(x)` | `F.gelu` (exact) |
| `silu_forward` | `x · σ(x)` | `σ(x)(1 + x(1 − σ(x)))` | `F.silu` |
| `tanh_forward` | `tanh(x)` | `1 − tanh²(x)` | `torch.tanh` |

`Φ` is the standard-normal CDF and `φ` its density.

## Numerical policy

All nonlinear intermediates are computed in **FP32** regardless of the storage
dtype, matching the rest of the lab. GELU uses the exact error-function form
(`tl.math.erf`) so it agrees with `torch.nn.functional.gelu` under its default
`approximate="none"`. `tanh` is evaluated as `2·σ(2x) − 1` because this Triton
build does not expose `tl.math.tanh`; the identity is exact and reuses the
shared two-branch sigmoid based on `exp(-abs(x))`, which keeps both saturating
tails finite without evaluating an overflowing exponential.

Correctness is validated on GPU against float32 PyTorch references with row
strides and a masked non-power-of-two tail (`37 × 1000` with `[::2]` storage),
both forward and through `torch.autograd.grad`. A dedicated test drives
saturating inputs (`|x|` up to 100) through both sigmoid branches and asserts
finiteness of the output and the input gradient.

## Scope and limitations

- 2D tensors with a contiguous final dimension; row stride may be arbitrary.
- FP16 and FP32 inputs. The registered benchmark path uses FP16, like the other
  memory-bound kernels.
- First-order autograd only; the custom Triton backward is explicitly marked
  non-differentiable.
- Backward is deterministic and elementwise (no cross-program reduction), so —
  unlike the LayerNorm/RMSNorm parameter gradients — it imposes no lock protocol
  and accepts PyTorch's deterministic mode.
- `silu` here is the standalone activation; the gated `value · SiLU(gate)` fusion
  remains in [`swiglu.md`](swiglu.md).

## Cost model

Each activation reads one input element and writes one output element, so the
roofline numerator is `2 · rows · cols · element_size` (memory-bound).
