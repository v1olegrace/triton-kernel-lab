# Benchmark results

Each subdirectory is named from `torch.cuda.get_device_name()` and contains:

- `peaks.json`: empirical bandwidth, cuBLAS sweeps, clocks, versions, and
  theoretical provenance;
- `<kernel>.json`: per-size timing and performance metrics;
- `layer_norm_backward.json`: gradient errors and two-stage backward timing;
- `layer_norm_lock_stress.json`: repeated high-contention lock validation;
- `rms_norm_lock_stress.json`: single-buffer lock/count validation;
- `layer_norm_backward_stages.png`: stage timing and incremental overhead;
- `compute_sanitizer_*.log`: focused real-GPU sanitizer summaries;
- `compute_sanitizer_attention_*.log`: Flash Attention tail/causal sanitizer summaries;
- `compute_sanitizer_rms_norm_*.log`: RMSNorm forward/backward sanitizer summaries;
- `compute_sanitizer_residual_rms_norm_*.log`: fused residual RMSNorm sanitizer summaries;
- `compute_sanitizer_elementwise_*.log`: SwiGLU and RoPE sanitizer summaries;
- `ncu_*.csv`: single-launch Nsight Compute counter exports, when available;
- `ncu_matmul_summary.json`: compact counter and launch-configuration summary;
- `attention_{noncausal,causal}.json`: Flash Attention timing vs PyTorch SDPA;
- `flash_attention_memory.json`: isolated-process peak-allocation study;
- `flash_attention_memory.png`: measured linear-vs-quadratic memory curve;
- `*_speedup.png`: Triton/PyTorch comparisons;
- `*_tflops.png`: compute throughput curves;
- cross-kernel comparison plots where applicable.

Results are machine-specific evidence, not universal performance guarantees.
Regenerate them with:

```bash
uv run tklab-bench --force-peaks
```

Do not edit generated JSON manually. The benchmark CLI writes files
atomically.
