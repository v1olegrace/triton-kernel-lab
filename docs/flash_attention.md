# Flash Attention forward

The Phase 6A kernels implement dense scaled dot-product attention without
materializing the quadratic score or probability matrices. Two registered
variants share one Triton kernel:

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

The same `N=1000` causal and non-causal workloads pass Compute Sanitizer
`memcheck`, `initcheck`, and `synccheck` with zero errors, plus `racecheck`
with zero hazards or warnings. Raw summaries are committed as
`compute_sanitizer_attention_*.log`.

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
amortized. WSL/WDDM allowed allocation beyond the RTX 4060's 8 GiB dedicated
memory through oversubscription, so the observed OOM occurred at `N=24576`,
not exactly when the dedicated-memory figure was crossed.

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

The clean RTX 4060 run, collected after the GPU remained idle with the Windows
screen locked, produced:

| N | Non-causal vs SDPA | Non-causal TFLOP/s | Causal vs SDPA | Causal TFLOP/s |
|---:|---:|---:|---:|---:|
| 512 | 0.863x | 15.61 | 0.945x | 9.55 |
| 1024 | 0.915x | 22.31 | 0.966x | 16.93 |
| 2048 | 0.875x | 25.31 | 0.913x | 21.40 |
| 4096 | 0.889x | 26.51 | 0.866x | 23.40 |
| 8192 | 0.865x | 26.43 | 0.867x | 25.02 |

The custom kernel reaches 86-97% of PyTorch SDPA across the sweep. Its best
non-causal result is 26.51 TFLOP/s, 82.67% of the clock-scaled
FP16-input/FP32-accumulate ceiling; the best causal result is 25.02 TFLOP/s,
78.00% of that ceiling under the exact lower-triangle FLOP count.

The autotuner favors 64x32 or 128x32 query/key tiles for most shapes, with
four warps and two stages. The causal `N=8192` winner uses 128x64. These
choices stay within the RTX 4060's per-block shared-memory limit and avoid
pretending that one tile is universally optimal.

The kernel follows the FlashAttention-2 forward structure described by the
official Triton fused-attention tutorial and the FlashAttention papers:

- <https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html>
- <https://arxiv.org/abs/2205.14135>
- <https://tridao.me/publications/flash2/flash2.pdf>

Backward is intentionally excluded from Phase 6A. It requires recomputation
of probabilities and separate gradient traversals for `dQ`, `dK`, and `dV`.
