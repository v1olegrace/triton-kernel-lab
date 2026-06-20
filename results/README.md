# Benchmark results

Each subdirectory is named from `torch.cuda.get_device_name()` and contains:

- `peaks.json`: empirical bandwidth, cuBLAS sweeps, clocks, versions, and
  theoretical provenance;
- `<kernel>.json`: per-size timing and performance metrics;
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
