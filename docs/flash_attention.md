# Flash Attention forward and backward

The kernels implement dense scaled dot-product attention without
materializing the quadratic score or probability matrices, with a
recompute-based backward integrated through `torch.autograd`. Two registered
variants share one forward Triton kernel:

- `attention_noncausal`;
- `attention_causal`, using separate unmasked off-band and masked diagonal
  stages.

The current production contract is FP16, contiguous
`(batch, heads, sequence, head_dim)` tensors with `head_dim=64`. Benchmarks use
batch 1 and 16 heads.

## Online-softmax equivalence

For one query row, let the scaled score blocks be `S_1, S_2, ...`. After some
blocks, retain only:

```text
m   = maximum score seen so far
l   = sum(exp(score - m)) over scores seen so far
acc = sum(exp(score - m) * value) over scores seen so far
```

For a new score block `S_j`:

```text
new_m = max(m, row_max(S_j))
alpha = exp(m - new_m)
p     = exp(S_j - new_m)

acc = alpha * acc + p @ V_j
l   = alpha * l   + row_sum(p)
m   = new_m
```

Multiplying the old state by `alpha` changes its denominator from `exp(m)` to
`exp(new_m)`. Therefore the updated `acc` and `l` are exactly the numerator
and denominator that a full stable softmax would compute over every processed
block. The final result is `acc / l`. Streaming changes the evaluation order,
not the mathematical function; disagreement with full SDPA comes from finite
precision only.

The implementation uses base-2 exponentials:

```text
exp(x) = exp2(x * log2(e))
```

The scale `1 / sqrt(head_dim)` is applied before updating the row maximum.
`m`, `l`, score exponentials, and the output accumulator use FP32. The
probability tile is converted to FP16 only for the Tensor Core `P @ V`
operation.

## Causal staging

A causal query block processes:

1. all complete key blocks strictly below its diagonal, without a mask;
2. its diagonal band, with `query_index >= key_index`;
3. no key blocks above the diagonal.

Skipping the upper blocks avoids both unnecessary work and the undefined
`-inf - (-inf)` expression that appears when an entirely masked block is fed
to online softmax. The final partial query/key block uses the same causal mask
plus an independent `key_index < sequence_length` tail mask.

## Correctness

Both variants are compared with
`torch.nn.functional.scaled_dot_product_attention`. The aggregate requirement
is relative Frobenius error below `2e-2`; measured errors are substantially
smaller. Dedicated tests cover `N=128` and the adversarial `N=1000`.

The adversarial case simultaneously exercises:

- a partial query block;
- a partial key/value block;
- the partial causal diagonal block;
- absence of NaN or infinity in the output.

The `N=1000` causal and non-causal forward workloads pass Compute Sanitizer
`memcheck`, `initcheck`, and `synccheck` with zero errors, plus `racecheck`
with zero hazards or warnings. The backward has an independent partial-tile
workload at `N=129`; both masking modes and all three gradients pass the same
four tools. Raw summaries are committed as
`compute_sanitizer_attention_*.log` and
`compute_sanitizer_attention_backward_*.log`.

## Memory complexity

`benchmarks/flash_attention_memory.py` measures each implementation in a fresh
process. Flash is warmed and autotuned before resetting CUDA peak statistics,
so Triton's one-time benchmark buffer is excluded.

The materialized reference retains:

```python
scores = q @ k.transpose(-2, -1)
probabilities = softmax(scores)
output = probabilities @ v
```

For batch 1, 16 heads, dimension 64, and FP16:

| N | Flash increment | Materialized increment |
|---:|---:|---:|
| 512 | 1 MiB | 25.1 MiB |
| 4096 | 8 MiB | 1.02 GiB |
| 8192 | 16 MiB | 4.02 GiB |
| 16384 | 32 MiB | 16.0 GiB |
| 24576 | 48 MiB | OOM |

The measured log-log slopes are 1.00 for Flash and 1.87 for the materialized
path; the latter approaches the expected quadratic slope as fixed overhead is
amortized. PyTorch allocator rounding and the remaining linear Q/K/V/output
terms also keep a finite-range empirical slope below exactly 2.00.

This run used WSL2 over a Windows WDDM driver with 8,188 MiB of reported
dedicated GPU memory. WDDM GPU virtual addressing can back allocations through
local or system-memory segments, and this environment demonstrably allowed the
PyTorch allocation counter to exceed dedicated VRAM. The 16.0 GiB observation
at `N=16384` is therefore a real allocation-footprint measurement, not a claim
that 16 GiB remained resident in the RTX 4060's physical VRAM.

The two retained FP16 quadratic tensors alone cross the reported dedicated
capacity at approximately `N=11582`, before Q/K/V, output, and allocator
overhead. On a platform that does not oversubscribe device allocations, OOM
would occur around or below that point. The observed OOM at `N=24576` is
specific to this WDDM/WSL environment. This study proves the `O(N)` versus
`O(N^2)` footprint; it does not claim that paged execution beyond dedicated
VRAM has useful performance.

WDDM memory model reference:
<https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-virtual-memory-in-wddm-2-0>.

## Performance methodology

The benchmark sweeps `N=512` through `8192`. Triton timing uses preallocated
output; PyTorch SDPA has no public `out=` variant and therefore includes its
output allocation. This small asymmetry is documented rather than hidden.
Profiler inspection confirms that this PyTorch 2.12.1/CUDA 13 stack dispatches
the reference to `aten::_scaled_dot_product_flash_attention` and the
`pytorch_flash::flash_fwd_kernel`, rather than a materialized math fallback.

FLOP counts include `Q @ K^T` and `P @ V`. Causal counts use the exact lower
triangle `N * (N + 1) / 2`. SDPA is the primary baseline; theoretical
FP16-input/FP32-accumulate throughput is reported only as secondary context.

The clean RTX 4060 run, collected with the physical display off and the
Windows session left unlocked, produced:

| N | Non-causal vs SDPA | Non-causal TFLOP/s | Causal vs SDPA | Causal TFLOP/s |
|---:|---:|---:|---:|---:|
| 512 | 0.877x | 16.13 | 0.962x | 9.73 |
| 1024 | 0.911x | 22.20 | 0.952x | 16.65 |
| 2048 | 0.871x | 25.11 | 0.905x | 21.16 |
| 4096 | 0.780x | 23.14 | 0.883x | 23.72 |
| 8192 | 0.914x | 24.94 | 0.879x | 23.26 |

The custom kernel reaches 78-96% of PyTorch SDPA across the sweep. Its best
non-causal result is 25.11 TFLOP/s, 75.28% of the
maximum-observed-clock FP16-input/FP32-accumulate ceiling; the best causal
result is 23.72 TFLOP/s, 71.10% of that ceiling under the exact
lower-triangle FLOP count.

Against the empirical 31.35 TFLOP/s cuBLAS FP16-input/FP32-accumulate peak,
the 25.11 TFLOP/s non-causal result is 80.12%. This is the appropriate
accumulation regime: both attention matrix products accumulate in FP32. The
remaining gap includes online-softmax max, `exp2`, row-sum, and rescaling work
interleaved with Tensor Core operations. The measurements are consistent with
that non-MMA work reducing attainable HMMA duty cycle, but no attention NCU
stall analysis is claimed.

The autotuner favors 64x32 or 128x32 query/key tiles for most shapes, with
four warps and two stages. The causal `N=8192` winner uses 128x64. These
choices stay within the RTX 4060's per-block shared-memory limit and avoid
pretending that one tile is universally optimal.

## Specialization scope

`N_CTX` is a Triton `constexpr` and an autotune key. Every distinct sequence
length therefore selects or compiles a specialization. This is intentional
for the fixed sweep and makes tile provenance unambiguous, but it can create
compile-cache churn for production workloads with many arbitrary lengths. A
production interface would normally bucket sequence lengths or use a
runtime-length variant with a bounded specialization policy.

`HEAD_DIM == 64` is a compile-time invariant, not a general FlashAttention
claim. Supporting 128 or 256 changes register, shared-memory, tile, and
occupancy trade-offs materially on AD107 and requires its own autotune and
correctness study.

The kernel follows the FlashAttention-2 forward structure described by the
official Triton fused-attention tutorial and the FlashAttention papers:

- <https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html>
- <https://arxiv.org/abs/2205.14135>
- <https://tridao.me/publications/flash2/flash2.pdf>

## Backward

`attention_noncausal` and `attention_causal` are differentiable. The forward
optionally stores the per-query base-2 log-sum-exp
`softmax_lse = row_max + log2(row_sum)` (a `STORE_LSE` compile-time flag, so
the inference-only and benchmark launchers remain byte-for-byte identical).
The name is deliberately explicit: this value is the complete normalized
log-sum-exp, not the running row maximum. The public wrapper also selects this
statistics-free path whenever gradients are disabled or no Q/K/V input
requires gradients. The backward then recomputes probabilities
from `Q`, `K`, `V`, `softmax_lse`, and the saved output rather than retaining
the score matrix, preserving the `O(N)` activation footprint.

Three kernels run per backward:

1. a preprocessing pass computing `delta_i = sum_d O_id * dO_id`, equal to
   `sum_j P_ij (dO_i . V_j)`;
2. a key/value pass that, for each `K`/`V` block, streams every contributing
   query block and accumulates `dV = P^T dO` and `dK = scale * dS^T Q`;
3. a query pass that, for each query block, streams the contributing key blocks
   and accumulates `dQ = scale * dS K`.

Here `P = exp2(scale_log2 * QK^T - softmax_lse)`, `dP = dO V^T`, and
`dS = P * (dP - delta)`. The softmax scale multiplies `dQ` and `dK` only, never
`dV`. The recomputed `P` is masked exactly as the forward masks scores: the
`key_index < sequence_length` tail for both variants, plus the
`query_index >= key_index` triangle in the causal case, so padded keys and
upper-triangle positions contribute zero to every gradient.

Probabilities, `dP`, and the accumulators are FP32; matmul operands are cast to
FP16 for the Tensor Core path, matching the forward policy.
The custom backward supports first-order gradients only and is explicitly
marked non-differentiable for higher-order autograd.

### Recompute cost

The split avoids quadratic saved activations and gives every output block a
single writer, but it deliberately recomputes work. Both gradient kernels
rebuild `QK^T` and `dP = dO V^T`: the `dK/dV` kernel while owning one key/value
block, and the `dQ` kernel while owning one query block. For dense non-causal
attention, the leading matrix-multiply model is approximately:

```text
forward:  QK^T + PV                                      =  4 N^2 d FLOPs
backward: 2(QK^T) + 2(dO V^T) + P^T dO + dS^T Q + dS K = 14 N^2 d FLOPs
```

Thus this implementation's backward is about `3.5x` the forward's leading
matmul FLOPs, before elementwise softmax work. This is an intentional
memory/ownership trade-off, not an unexplained throughput regression. A more
aggressive implementation could fuse or share recomputed intermediates, but
would require a different ownership and synchronization design.

### Backward correctness

`dQ`, `dK`, and `dV` are validated on GPU against `torch.autograd.grad` of
`scaled_dot_product_attention` for both masking modes at `N=128` and the
adversarial `N=1000`, under a relative Frobenius bound of `3e-2`. The looser
bound versus the forward reflects the additional FP16 reductions over the
sequence in `dK`/`dV`. Larger head dimensions, GQA/MQA, dropout, and attention
bias remain out of scope, identical to the forward contract.
