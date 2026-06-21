"""Unified roofline: every kernel against the measured RTX 4060 ceilings.

The figure plots one point per kernel on a log-log roofline. Throughput and
percent-of-ceiling are read from the committed benchmark JSON (empirical);
operational intensity is analytic.

Byte counts in each intensity reuse the *exact* per-kernel cost model that the
benchmarks themselves use (``KernelSpec.bytes_moved`` / ``KernelSpec.flops`` in
``src/tklab/kernels``); the FLOP counts are the conventional analytic estimates
documented inline. Intensity only sets a point's x-position, so these estimates
are approximate by construction -- the vertical distance to a ceiling (the part
that carries the "optimized for its regime" claim) is the measured throughput
ratio and is independent of the FLOP count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results" / "nvidia_geforce_rtx_4060"

_MEM_COLOR = "#2b6cb0"
_CMP_COLOR = "#c53030"


@dataclass(frozen=True)
class Kernel:
    """One roofline point sourced from a committed benchmark JSON."""

    name: str  # label on the plot
    json_name: str  # file under RESULTS
    metric: str  # "gbps" (memory) or "tflops" (compute)
    regime: str  # "mem" or "cmp"
    roof: str  # which ceiling the % is measured against
    intensity: float  # analytic FLOP/byte (x-position)
    note: str  # how the intensity was derived
    select: dict[str, object] | None = None  # row filter; None => best metric
    label_dx: float = 6.0  # label offset (points) for inline labels
    label_dy: float = 5.0
    # Leader-line label target in DATA coords (x, y). When set, the label is
    # placed in the open space below the diagonal and joined to the marker by a
    # thin line, so the dense memory-bound cluster stays legible without moving
    # any point off its true intensity.
    leader: tuple[float, float] | None = None


# Intensity derivations (s = dtype bytes; per-element FLOP / per-element bytes;
# row/column factors cancel for the row-wise kernels):
#   vector_add  fp32 : add=1 FLOP/elem ; 3 reads+write x s  -> 1/(3*4)   = 0.083
#   softmax     fp32 : max,sub,exp,sum,div ~5 ; 2*s          -> 5/(2*4)   = 0.625
#   layer_norm  fp16 : mean,var,norm,scale,shift ~5 ; 2*s    -> 5/(2*2)   = 1.25
#   rms_norm    fp16 : sq,sum,rstd-mul,weight-mul ~4 ; 2*s    -> 4/(2*2)   = 1.0
#   residual    fp16 : add,sq,sum,norm,scale ~5 ; 4*s         -> 5/(4*2)   = 0.625
#   swiglu      fp16 : sigmoid~4,silu-mul,out-mul ~6 ; 3*s     -> 6/(3*2)   = 1.0
#   rope        fp16 : x*cos, rot*sin, add ~3 ; 3*s            -> 3/(3*2)   = 0.5
#   matmul      fp16 : 2n^3 ; 3n^2*s  -> n/(3*s); n=2048       -> 2048/6    = 682.7
#   flash caus. fp16 : ~N/(2s) (Q,K,V,O traffic); N=4096       -> 4096/4    = 1024
KERNELS: tuple[Kernel, ...] = (
    Kernel(
        "vector_add",
        "vector_add.json",
        "gbps",
        "mem",
        "bw",
        1 / 12,
        "add / 3 r-w x4B",
        select={"dtype": "float32"},
        label_dy=-12,
    ),
    Kernel(
        "rope",
        "rope_forward.json",
        "gbps",
        "mem",
        "bw",
        0.5,
        "3 flop / 3 r-w x2B",
        leader=(2.2, 1.35e11),
    ),
    Kernel(
        "softmax",
        "fused_softmax.json",
        "gbps",
        "mem",
        "bw",
        0.625,
        "5 flop / 2 r-w x4B",
        select={"dtype": "float32"},
        leader=(3.6, 9.2e10),
    ),
    Kernel(
        "residual_rms_norm",
        "residual_rms_norm_forward.json",
        "gbps",
        "mem",
        "bw",
        0.625,
        "5 flop / 4 r-w x2B",
        leader=(6.2, 6.3e10),
    ),
    Kernel(
        "rms_norm",
        "rms_norm_forward.json",
        "gbps",
        "mem",
        "bw",
        1.0,
        "4 flop / 2 r-w x2B",
        leader=(10.5, 4.4e10),
    ),
    Kernel(
        "swiglu",
        "swiglu_forward.json",
        "gbps",
        "mem",
        "bw",
        1.0,
        "6 flop / 3 r-w x2B",
        leader=(18.0, 3.1e10),
    ),
    Kernel(
        "layer_norm",
        "layer_norm_forward.json",
        "gbps",
        "mem",
        "bw",
        1.25,
        "5 flop / 2 r-w x2B",
        leader=(31.0, 2.2e10),
    ),
    Kernel(
        "matmul fp16-acc",
        "matmul_fp16acc.json",
        "tflops",
        "cmp",
        "fp16",
        2 * 2048 / 6,
        "2n^3 / 3n^2 x2B @ n=2048",
        select={"size": 2048},
        leader=(1650.0, 5.6e13),
    ),
    Kernel(
        "matmul fp32-acc",
        "matmul_fp32acc.json",
        "tflops",
        "cmp",
        "fp32",
        2 * 2048 / 6,
        "2n^3 / 3n^2 x2B @ n=2048",
        select={"size": 2048},
        leader=(1650.0, 2.55e13),
    ),
    Kernel(
        "flash_attn (causal)",
        "attention_causal.json",
        "tflops",
        "cmp",
        "fp32",
        1024.0,
        "~N/2s @ N=4096",
        leader=(2600.0, 1.5e13),
    ),
)


def _load_rows(json_name: str) -> list[dict[str, Any]]:
    payload = json.loads((RESULTS / json_name).read_text())
    return cast("list[dict[str, Any]]", payload["rows"])


def _select_row(
    rows: list[dict[str, Any]], metric: str, select: dict[str, object] | None
) -> dict[str, Any]:
    """Return the chosen row: best ``metric`` among ``select`` matches."""
    candidates = rows
    if select is not None:
        candidates = [r for r in rows if all(r.get(k) == v for k, v in select.items())]
        if not candidates:
            raise ValueError(f"no row matched {select}")
    return max(candidates, key=lambda r: r[metric])


def main() -> None:
    """Render the unified roofline PNG from the committed RTX 4060 JSONs."""
    peaks = json.loads((RESULTS / "peaks.json").read_text())
    bw = peaks["peak_bw_gbps"] * 1e9  # B/s
    roofs = {
        "fp32": peaks["theoretical_tflops_fp16_fp32acc_at_measured_clock"] * 1e12,
        "fp16": peaks["theoretical_tflops_fp16_fp16acc_at_measured_clock"] * 1e12,
    }

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    intensities = np.logspace(-2, 4, 400)

    # Two memory-then-compute ceilings; the diagonal (I*BW) is shared.
    ax.loglog(
        intensities,
        np.minimum(intensities * bw, roofs["fp16"]),
        "-",
        color="0.35",
        lw=1.4,
        label="fp16-acc roof",
    )
    ax.loglog(
        intensities,
        np.minimum(intensities * bw, roofs["fp32"]),
        "--",
        color="0.55",
        lw=1.2,
        label="fp32-acc roof",
    )
    # Roof values sit just past each ridge point so they clear the kernel
    # markers clustered at the right edge.
    for key, txt, xpos in (("fp16", "fp16-acc", 320.0), ("fp32", "fp32-acc", 150.0)):
        ax.text(
            xpos,
            roofs[key],
            f"{roofs[key] / 1e12:.1f} TFLOP/s ({txt})",
            va="bottom",
            ha="left",
            fontsize=8,
            color="0.4",
        )
    ax.text(
        0.012,
        0.012 * bw,
        f"{bw / 1e9:.0f} GB/s ",
        rotation=34,
        va="bottom",
        ha="left",
        fontsize=8,
        color="0.4",
    )

    reported: list[tuple[str, str, float, float]] = []
    for k in KERNELS:
        row = _select_row(_load_rows(k.json_name), k.metric, k.select)
        if k.regime == "mem":
            perf = k.intensity * row["gbps"] * 1e9  # FLOP/s = (FLOP/B) * (B/s)
            pct = row["pct_peak"]
            throughput_label = f"{row['gbps']:.0f} GB/s"
        else:
            perf = row["tflops"] * 1e12
            pct = row["pct_theoretical"]
            throughput_label = f"{row['tflops']:.1f} TFLOP/s"
        color = _MEM_COLOR if k.regime == "mem" else _CMP_COLOR
        ax.loglog(k.intensity, perf, "o", color=color, ms=7, zorder=5)
        if k.leader is not None:
            ax.annotate(
                f"{k.name}  {pct:.0f}%",
                (k.intensity, perf),
                xytext=k.leader,
                textcoords="data",
                fontsize=7.5,
                color=color,
                va="center",
                ha="left",
                arrowprops={
                    "arrowstyle": "-",
                    "color": color,
                    "lw": 0.6,
                    "shrinkA": 3,
                    "shrinkB": 2,
                },
            )
        else:
            ax.annotate(
                f"{k.name}\n{pct:.0f}%",
                (k.intensity, perf),
                textcoords="offset points",
                xytext=(k.label_dx, k.label_dy),
                fontsize=7.5,
                color=color,
                ha="right" if k.label_dx < 0 else "left",
            )
        reported.append((k.name, throughput_label, pct, k.intensity))

    ax.set_xlabel("Operational intensity (FLOP/byte, analytic)")
    ax.set_ylabel("Performance (FLOP/s)")
    ax.set_title("triton-kernel-lab — roofline (RTX 4060, SM89)")
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(1e10, 1e14)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    out = RESULTS / "roofline.png"
    # Suppress the version-stamped "Software" tEXt chunk so the PNG is a
    # deterministic function of the committed JSONs (same data -> same bytes),
    # independent of the matplotlib build that rendered it.
    fig.savefig(out, dpi=150, metadata={"Software": None})
    print(f"wrote {out}")
    print(f"\n{'kernel':22s} {'throughput':14s} {'% ceiling':>9s} {'intensity':>10s}")
    for name, tput, pct, inten in reported:
        print(f"{name:22s} {tput:14s} {pct:8.1f}% {inten:10.3f}")


if __name__ == "__main__":
    main()
