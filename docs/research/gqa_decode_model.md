# Contiguous GQA decode performance model

Specification version: **1.0**. Status: mathematical contract; documentation
only.

This document freezes the mathematical vocabulary for a future contiguous
grouped-query-attention (GQA) decode research program. It defines a workload
and the quantities that future implementations and measurements must report;
it does not describe an implemented GQA kernel or claim a performance result.

The model uses two independent annotations:

- epistemic labels describe why a statement is present;
- measurement-status labels describe the maturity of a numerical input.

## Epistemic and measurement labels

The epistemic labels are:

- `[D]`: derived from the mathematical model;
- `[E]`: existing measured repository evidence;
- `[H]`: hypothesis for a future experiment;
- `[I]`: interpretation of a derivation or observation;
- `[X]`: unknown or not measured.

The measurement-status labels are:

- `CALIBRATED`: measured by a versioned laboratory calibration with recorded
  environment and protocol;
- `DIAGNOSTIC`: observed in a non-reference, dirty, or incomplete protocol and
  not eligible for a repository performance claim;
- `NOMINAL`: a device specification or analytical assumption, not a laboratory
  measurement;
- `UNKNOWN`: no admissible measurement exists in the repository.

An `[H]` statement must not be promoted to `[E]` without a committed experiment
artifact. A `NOMINAL` value must not be presented as `CALIBRATED`.

## Canonical decode workload

The first research target is regular, contiguous GQA decode with one query
token:

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `Hq` | number of query heads |
| `Hkv` | number of key/value heads |
| `G = Hq / Hkv` | query heads per KV head |
| `Dh` | head dimension |
| `L` | KV context length |
| `q_len` | query length, fixed to 1 for decode |
| `sKV` | bytes per stored KV payload element |
| `sQ` | bytes per Q element |
| `sO` | bytes per output element |

The canonical Llama-3-8B-like geometry is:

```text
Hq  = 32
Hkv = 8
G   = 4
Dh  = 128
```

`B` and `L` remain symbolic. The model assumes `Hq` is divisible by `Hkv`.
Unless a section explicitly says otherwise, byte counts describe only the
attention core and exclude projection weights, allocator activity, kernel
launches, and KV-cache append traffic.

## Semantic GQA mapping

For regular grouped-query attention, each query head maps to exactly one KV
head:

```text
kv_head(q_head) = floor(q_head / G)
```

A query head does not attend to all `Hkv` heads. Consequently, `Hkv` must not
appear as an extra multiplier in the useful `QK^T` or `PV` FLOP count.

## Unique compulsory KV payload

The unique element counts are `[D]`:

```text
N_K = B Hkv L Dh
N_V = B Hkv L Dh
```

The unique stored KV payload is therefore `[D]`:

```text
Bytes_KV_unique = 2 sKV B Hkv L Dh
```

`Bytes_KV_unique` is a logical compulsory-payload quantity. It is not, by
itself, a measurement of algorithmic load requests, L2-to-SM traffic, or
DRAM-to-L2 traffic.

For a sub-byte representation, `sKV` may be an effective fractional payload
size, such as 0.5 byte for densely packed INT4. Packing, alignment, scales,
zero-points, and other metadata are separate costs and can make the stored
representation larger than this payload-only expression.

## Useful and executed work

This project counts one fused multiply-add as two FLOPs. Under that convention,
the useful decode matrix work is `[D]`:

```text
F_QK_useful = 2 B Hq L Dh
F_PV_useful = 2 B Hq L Dh

F_decode_useful = 4 B Hq L Dh
```

This first-order model excludes scale, softmax, address generation, reductions,
type conversion, and other auxiliary instructions.

Every future kernel must keep these two quantities distinct:

```text
F_useful
F_executed
```

`F_useful` describes the conventional mathematical workload. `F_executed`
describes operations actually issued by a concrete implementation and may be
larger because of padding, masking, redundant work, split-KV partials, or
auxiliary reductions. They must not be silently substituted for each other.
Hardware instruction classes that do not map cleanly to conventional FLOPs
must be reported separately rather than assigned an invented FLOP count.

## Operational intensity

### KV-payload-only ideal

The asymptotic intensity obtained from useful decode FLOPs and the unique KV
payload is `[D]`:

```text
I_KV_ideal = F_decode_useful / Bytes_KV_unique
           = 2 G / sKV
```

For `G = 4`:

| KV representation | Payload bytes/element | `I_KV_ideal` |
|---|---:|---:|
| FP16 or BF16 | 2 | 4 FLOP/B |
| FP8 payload only | 1 | 8 FLOP/B |

These are KV-payload-only intensities, not complete kernel intensities.

### More complete minimum-traffic model

The minimum model additionally counts one Q read, one output write, and the
stored quantization metadata `[D]`:

```text
Bytes_Q = sQ B Hq Dh
Bytes_O = sO B Hq Dh

Bytes_min = Bytes_KV_unique
          + Bytes_Q
          + Bytes_O
          + Bytes_quant_metadata

I_min = F_decode_useful / Bytes_min
```

Quantized KV storage must use:

```text
Bytes_quantized_KV = Bytes_payload
                   + Bytes_scales
                   + Bytes_zero_points
                   + Bytes_other_metadata
```

`Bytes_quant_metadata` may include per-tensor, per-head, per-channel, or
per-group scales and zero-points, plus layout and alignment overhead. Its value
must remain symbolic until the representation, grouping, scalar dtypes,
packing, and padding rules are fixed. A toy calculation may set it to zero only
when the omission is explicit in the label.

### Physical intensity

A measured physical intensity is a different quantity:

```text
I_physical = F_executed / Bytes_observed
```

`Bytes_observed` must name its boundary, such as DRAM-to-L2 or L2-to-SM, and
must include all traffic attributable to the timed kernel at that boundary.
`I_physical`, `I_min`, and `I_KV_ideal` answer different questions and must not
share an unlabeled `intensity` field.

## KV traffic amplification

One generic amplification factor cannot identify where GQA reuse is lost. The
model defines three quantities:

```text
A_alg  = algorithmic KV load bytes
         / Bytes_KV_unique

A_l2_to_sm = observed L2-to-SM KV-related bytes
             / Bytes_KV_unique

A_dram = observed DRAM-to-L2 KV-related bytes
         / Bytes_KV_unique
```

The conceptual path is:

```text
logical GQA reuse
        -> algorithmic load requests
        -> L1/shared/cache reuse
        -> L2 traffic
        -> DRAM traffic
```

A program-per-query-head decomposition may issue approximately `G` copies of
the logical KV loads. That does not imply `G` copies of DRAM traffic because
cache hits may serve the redundant requests. Conversely, a warm invocation can
have `A_dram < 1` relative to one invocation's unique payload when cache lines
were resident before timing.

No value is assigned to `A_alg`, `A_l2_to_sm`, or `A_dram` in this
specification. If profiler counters cannot isolate KV from Q, output, metadata,
and auxiliary traffic, the measured numerator must be labeled as whole-kernel
traffic and the KV-specific amplification remains `[X]`.

## Cache state is part of the experiment

Future results must distinguish at least:

```text
warm-cache
controlled cache-thrashing/cache-state
```

The second term must not be shortened to "perfect cache flush" unless the
measurement protocol demonstrates that property. The current benchmark
methodology states that `triton.testing.do_bench` performs L2 clearing between
measured repetitions; a future GQA experiment must still record and validate
the exact behavior it relies on.

For `B=1`, `Hkv=8`, `Dh=128`, `L=4096`, and FP16/BF16 KV, the payload occupies
16 MiB `[D]`. The local RTX 4060 reports a nominal 24 MiB L2 capacity
`[I][NOMINAL]`. KV payload alone reaches that nominal capacity at `[D]`:

```text
L = 24 MiB / (2 * 2 B Hkv Dh)
  = 6144 tokens
```

This is a reference point, not a prediction of an exact cache cliff. Q, output,
metadata, instructions, unrelated data, set mapping, replacement policy, and
concurrent traffic all reduce or reshape the effective cache available to KV.

## Reference numerical point

For:

```text
B   = 1
Hq  = 32
Hkv = 8
G   = 4
Dh  = 128
L   = 4096
```

the exact derived values are:

| Quantity | Value | Status |
|---|---:|---|
| `N_K` | 4,194,304 elements | `[D]` |
| `N_V` | 4,194,304 elements | `[D]` |
| `F_QK_useful` | 33,554,432 FLOPs | `[D]` |
| `F_PV_useful` | 33,554,432 FLOPs | `[D]` |
| `F_decode_useful` | 67,108,864 FLOPs | `[D]` |
| FP16/BF16 unique KV | 16,777,216 bytes = 16 MiB | `[D]` |
| FP8 unique KV payload | 8,388,608 bytes = 8 MiB | `[D]` |

With FP16 Q, KV, and output, and no quantization metadata:

```text
Bytes_min = 16,777,216 + 8,192 + 8,192
          = 16,793,600 bytes

I_min = 67,108,864 / 16,793,600
      = 3.9960975609... FLOP/B
      ~= 3.9961 FLOP/B
```

With FP8 KV payload, BF16 Q/output, and explicitly ignored quantization
metadata:

```text
Bytes_min_toy = 8,388,608 + 8,192 + 8,192
              = 8,404,992 bytes

I_min_toy = 67,108,864 / 8,404,992
          = 7.9844054580... FLOP/B
          ~= 7.9844 FLOP/B
```

The latter is a payload-only quantization toy point, not a complete FP8 storage
or dequantization model.

## Bandwidth lower bounds

The committed RTX 4060 calibration records 250.137417 GB/s
`[E][CALIBRATED]`. The rounded analytical reference below uses exactly
`BW = 250e9 B/s`; it is not a universal device constant.

Using only the unique KV payload gives `[D]`:

| KV representation | Analytical traffic lower bound |
|---|---:|
| FP16/BF16, 16 MiB | 67.108864 microseconds |
| FP8 payload only, 8 MiB | 33.554432 microseconds |

These values are **analytical traffic lower bounds**, not predicted kernel
latencies or expected measurements. They exclude Q/output traffic,
quantization metadata, reductions, launch latency, synchronization, redundant
loads, and insufficient parallelism.

The first-order roofline lower bound is:

```text
T >= max(F / P, Bytes / BW)
```

`F`, `P`, `Bytes`, and `BW` must refer to compatible numerical and traffic
policies. Launch latency, occupancy, insufficient parallelism, cache behavior,
synchronization, reductions, instruction throughput, and other non-overlapped
costs can raise measured latency above this bound.

The repository currently contains no admissible PCIe bandwidth calibration for
12.45 GiB/s and no admissible CPU-memory calibration for approximately
40 GiB/s. Both values are `[X][UNKNOWN]` here. They must not become project
constants until a reproducible benchmark records their direction, allocation
mode, transfer size, warmup, timing semantics, and environment.

## Prefill convention

Decode formulas above must not be reused as prefill formulas. For query and KV
sequence length `S`, dense non-causal attention has `[D]`:

```text
F_prefill_noncausal = 4 B Hq Dh S^2
```

Causal useful work covers the exact lower triangle `[D]`:

```text
P_causal = S(S + 1) / 2

F_prefill_causal_useful = 4 B Hq Dh P_causal
                          = 2 B Hq Dh S(S + 1)
```

Every prefill result must state whether it reports non-causal useful FLOPs,
causal useful FLOPs, or executed FLOPs. The existing FlashAttention benchmark
already follows the full-square and exact-lower-triangle conventions.

## Llama linear-projection convention

For future block-level accounting, define:

```text
d    = Hq Dh
d_kv = Hkv Dh = d / G
M    = B S
```

The GQA projection FLOPs are `[D]`:

```text
F_Q = 2 M d^2
F_K = 2 M d d_kv
F_V = 2 M d d_kv
F_O = 2 M d^2
```

SwiGLU has gate, up, and down projections: three GEMMs. Therefore `[D]`:

```text
F_MLP = 6 M d d_ff

F_linear = 4 M d^2 (1 + 1/G)
         + 6 M d d_ff
```

Attention work, normalization, RoPE, residual operations, nonlinear scalar
instructions, and any materialized intermediate traffic remain separate. No
speedup follows from these FLOP counts alone.

## Falsifiable hypotheses

- **H01 `[H]`:** Long-context contiguous GQA decode enters a memory-side
  regime, but the binding resource and transition point must be measured rather
  than inferred solely from `I_KV_ideal`.
- **H02 `[H]`:** At low batch or insufficient parallelism, split-KV reduces
  under-utilization at the cost of partial-state traffic and a final exact LSE
  combine.
- **H03 `[H]`:** Grouping the `G` query heads that share one KV head reduces
  algorithmic redundant KV traffic relative to a program-per-query-head
  decomposition.
- **H04 `[H]`:** A reduction in `A_alg` does not translate one-to-one into a
  reduction in `A_dram` because cache reuse may already service redundant
  requests.
- **H05 `[H]`:** The observed performance regime changes as the KV working set
  crosses the effective cache hierarchy.

None of these hypotheses is supported evidence until a controlled experiment
and its raw artifacts are committed.

## Repository compatibility and sources

- The current attention cost model counts both matrix multiplications and uses
  `S^2` for non-causal attention and `S(S+1)/2` for causal attention:
  [`src/tklab/kernels/flash_attention.py`](../../src/tklab/kernels/flash_attention.py).
- Timing, L2-clearing, byte-model, roofline, and provenance conventions are
  defined in [`docs/benchmarking.md`](../benchmarking.md).
- The calibrated bandwidth and its session provenance are stored in
  [`results/nvidia_geforce_rtx_4060/peaks.json`](../../results/nvidia_geforce_rtx_4060/peaks.json).
- Existing FlashAttention supports equal Q/K/V head counts and `Dh=64`; it is
  not evidence for the future GQA decode workload defined here. Its scope is
  recorded in [`docs/flash_attention.md`](../flash_attention.md).

This specification intentionally adds no kernel, benchmark behavior, result
artifact, or README performance claim.
