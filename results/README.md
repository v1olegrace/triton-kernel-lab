# Benchmark results

Each subdirectory is named from `torch.cuda.get_device_name()` and contains:

- `peaks.json`: empirical bandwidth, cuBLAS sweeps, clocks, versions, and
  theoretical provenance;
- `<kernel>.json`: per-size timing and performance metrics;
- `layer_norm_backward.json`: gradient errors and two-stage backward timing;
- `layer_norm_lock_stress.json`: repeated high-contention lock validation;
- `layer_norm_backward_stages.png`: stage timing and incremental overhead;
- `compute_sanitizer_*.log`: focused real-GPU sanitizer summaries;
- `ncu_*.csv`: single-launch Nsight Compute counter exports, when available;
- `ncu_matmul_summary.json`: compact counter and launch-configuration summary;
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
