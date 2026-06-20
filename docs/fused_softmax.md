# Fused softmax

The kernel computes a numerically stable row-wise softmax in one Triton
program per row. Masked tail lanes load negative infinity, and all exponent
and reduction work is performed in FP32 before storing in the input dtype.

## What the baselines prove

The benchmark reports two distinct comparisons:

- `speedup_vs_naive` compares against a composition of `max`, subtraction,
  `exp`, `sum`, and division. This baseline materializes intermediates and
  demonstrates the memory-traffic benefit of fusion.
- `speedup` compares against `torch.softmax`. Current PyTorch dispatches this
  operation to a fused vendor CUDA kernel, so this comparison measures kernel
  quality rather than the basic benefit of fusion.

`pct_peak` is the primary metric for large rows. Both Triton and the vendor
kernel approach the same DRAM roofline, so their speedup converges toward one.

## Occupancy curve

The speedup curve has the expected bell-like shape:

- Small rows are dominated by launch overhead and underutilize the SM with
  one program per row.
- Medium rows provide enough work per program while retaining useful
  occupancy.
- Large power-of-two blocks increase register pressure. Both implementations
  then converge on the same memory-bandwidth ceiling.

On the RTX 4060 run, the vendor-kernel speedup peaks around `N=4096` in FP16
and trends back toward one by `N=16384`.

## Scope

The final dimension must be contiguous. Arbitrary row strides are supported.
Rows wider than 65,536 columns are rejected because a single-pass row no
longer fits the intended on-chip working set; those shapes require an online
softmax design.
