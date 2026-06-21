"""Empirical bandwidth and compute-roofline calibration.

Bandwidth is measured with large device-to-device copies. Compute baselines
use allocation-free ``torch.mm`` calls, a steady-state warmup, and concurrent
SM-clock sampling. Theoretical values are explicitly separated from measured
cuBLAS throughput.
"""

from __future__ import annotations

import math
import re
import statistics
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Generic, TypedDict, TypeVar, cast

import torch
import triton

from tklab.harness.jsonio import JsonObject, read_json_object, write_json_atomic

PEAKS_SCHEMA_VERSION = 7
DEFAULT_BW_ELEMENT_COUNTS = tuple(2**exponent for exponent in range(22, 26))
DEFAULT_TFLOPS_SIZES = (512, 1024, 2048, 4096, 8192)
_NVIDIA_RTX4060_SPEC_URL = (
    "https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/"
)
_NVIDIA_ADA_WHITEPAPER_URL = (
    "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf"
)


@dataclass(frozen=True, slots=True)
class GpuTheoreticalProfile:
    """Hardware facts required to derive clock-scaled Tensor Core ceilings."""

    model_substring: str
    rated_boost_mhz: int
    streaming_multiprocessors: int
    tensor_cores_per_sm: int
    fp16_fp32acc_flops_per_clock_per_tensor_core: int
    product_spec_url: str
    architecture_whitepaper_url: str

    def fp16_fp32acc_tflops(self, clock_mhz: int) -> float:
        """Return dense FP16-input/FP32-accumulate throughput at ``clock_mhz``."""
        return (
            self.streaming_multiprocessors
            * self.tensor_cores_per_sm
            * self.fp16_fp32acc_flops_per_clock_per_tensor_core
            * clock_mhz
            / 1e6
        )

    def fp16_fp16acc_tflops(self, clock_mhz: int) -> float:
        """Return dense FP16-input/FP16-accumulate throughput at ``clock_mhz``."""
        return 2.0 * self.fp16_fp32acc_tflops(clock_mhz)


RTX4060_PROFILE = GpuTheoreticalProfile(
    model_substring="RTX 4060",
    rated_boost_mhz=2460,
    streaming_multiprocessors=24,
    tensor_cores_per_sm=4,
    fp16_fp32acc_flops_per_clock_per_tensor_core=128,
    product_spec_url=_NVIDIA_RTX4060_SPEC_URL,
    architecture_whitepaper_url=_NVIDIA_ADA_WHITEPAPER_URL,
)
_GPU_PROFILES = (RTX4060_PROFILE,)
_ResultT = TypeVar("_ResultT")


class BandwidthSample(TypedDict):
    """One device-copy bandwidth observation."""

    elements: int
    working_set_bytes: int
    ms: float
    gbps: float


class TflopsSample(TypedDict):
    """One square GEMM throughput and clock observation."""

    n: int
    ms: float
    tflops: float
    sm_clock_mhz: int
    sm_clock_min_mhz: int
    sm_clock_max_mhz: int
    sm_clock_samples_mhz: list[int]


class TheoreticalProvenance(TypedDict):
    """Serialized provenance for theoretical compute ceilings."""

    model: str
    rated_boost_clock_mhz: int
    streaming_multiprocessors: int
    tensor_cores_per_sm: int
    fp16_fp32acc_flops_per_clock_per_tensor_core: int
    fp16_fp32acc_rated_tflops: float
    fp16_fp16acc_rated_tflops: float
    ceiling_clock_policy: str
    product_spec_url: str
    architecture_whitepaper_url: str


class PeakResults(TypedDict):
    """Versioned roofline-calibration payload."""

    schema_version: int
    gpu_name: str
    gpu_slug: str
    compute_capability: str
    torch_version: str
    triton_version: str
    calibration_started_at_utc: str
    calibration_preflight_gpu_utilization_pct: list[int]
    calibration_preflight_gpu_utilization_limit_pct: int
    peak_bw_gbps: float
    bandwidth_elements: int
    bandwidth_samples: list[BandwidthSample]
    cublas_tflops_fp16_fp32acc: float
    cublas_tflops_fp16_reduced_precision_allowed: float
    torch_fp16acc_available: bool
    theoretical_tflops_fp16_fp32acc_rated: float
    theoretical_tflops_fp16_fp32acc_at_measured_clock: float
    theoretical_tflops_fp16_fp16acc_rated: float
    theoretical_tflops_fp16_fp16acc_at_measured_clock: float
    theoretical_ceiling_sm_clock_mhz: int
    tflops_samples: list[TflopsSample]
    tflops_samples_reduced_precision_allowed: list[TflopsSample]
    best_tflops_size: int
    best_tflops_sm_clock_mhz: int
    best_reduced_precision_size: int
    best_reduced_precision_sm_clock_mhz: int
    allow_fp16_reduced_precision_reduction: bool
    theoretical_provenance: TheoreticalProvenance


def torch_fp16acc_available() -> bool:
    """Report whether this torch build exposes FP16 Tensor Core accumulation.

    ``torch.backends.cuda.matmul.allow_fp16_accumulation`` (torch >= 2.7) is the
    knob that selects the FP16-accumulate GEMM path. This is a version-determined
    capability probe, not a throughput measurement; the achieved FP16-accumulate
    rate is characterized separately by the matmul benchmark.
    """
    return hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation")


def gpu_slug(device: int = 0) -> str:
    """Return a filesystem-safe slug for a CUDA device name.

    Args:
        device: CUDA device index.

    Returns:
        Lowercase alphanumeric slug.

    Raises:
        RuntimeError: If CUDA or ``device`` is unavailable.
    """
    _validate_cuda_device(device)
    name = torch.cuda.get_device_name(device).lower()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def gpu_utilization_pct(device: int = 0) -> int:
    """Return the current whole-device utilization percentage.

    Args:
        device: CUDA device index.

    Returns:
        Integer utilization percentage reported by ``nvidia-smi``.

    Raises:
        RuntimeError: If ``nvidia-smi`` cannot provide the value.
    """
    return _nvidia_smi_integer(
        "utilization.gpu",
        device=device,
        error_message="unable to query GPU utilization with nvidia-smi",
    )


def measured_peak_bw_gbps(
    device: int = 0,
    *,
    element_counts: tuple[int, ...] = DEFAULT_BW_ELEMENT_COUNTS,
    warmup_ms: int = 100,
    repetition_ms: int = 500,
) -> float:
    """Measure the best sustained device-copy bandwidth.

    Args:
        device: CUDA device index.
        element_counts: FP32 tensor sizes to sweep.
        warmup_ms: Triton benchmark warmup duration.
        repetition_ms: Triton benchmark measurement duration.

    Returns:
        Maximum observed effective GB/s, counting one read and one write.
    """
    samples = _measure_bandwidth_samples(
        device,
        element_counts=element_counts,
        warmup_ms=warmup_ms,
        repetition_ms=repetition_ms,
    )
    return max(sample["gbps"] for sample in samples)


def theoretical_tflops_fp16_fp32acc(
    sm_clock_mhz: int,
    *,
    profile: GpuTheoreticalProfile = RTX4060_PROFILE,
) -> float:
    """Return a theoretical dense FP16/FP32-accumulate ceiling."""
    return profile.fp16_fp32acc_tflops(sm_clock_mhz)


def theoretical_tflops_fp16_fp16acc(
    sm_clock_mhz: int,
    *,
    profile: GpuTheoreticalProfile = RTX4060_PROFILE,
) -> float:
    """Return a theoretical dense FP16/FP16-accumulate ceiling."""
    return profile.fp16_fp16acc_tflops(sm_clock_mhz)


def measure_peaks(
    device: int = 0,
    *,
    calibration_started_at_utc: str,
    calibration_preflight_gpu_utilization_pct: tuple[int, ...],
    calibration_preflight_gpu_utilization_limit_pct: int,
) -> PeakResults:
    """Calibrate empirical and theoretical roofline baselines.

    Args:
        device: CUDA device index.
        calibration_started_at_utc: ISO-8601 start time for this clean run.
        calibration_preflight_gpu_utilization_pct: Utilization samples
            collected before calibration.
        calibration_preflight_gpu_utilization_limit_pct: Maximum permitted
            preflight utilization.

    Returns:
        Versioned peak-calibration payload.

    Raises:
        ValueError: If no theoretical profile matches the selected GPU.
        RuntimeError: If CUDA, ``nvidia-smi``, or the selected device is
            unavailable.
    """
    _validate_cuda_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    profile = _profile_for(gpu_name)
    major, minor = torch.cuda.get_device_capability(device)

    bandwidth_samples = _measure_bandwidth_samples(
        device,
        element_counts=DEFAULT_BW_ELEMENT_COUNTS,
        warmup_ms=100,
        repetition_ms=500,
    )
    best_bandwidth = max(bandwidth_samples, key=lambda sample: sample["gbps"])
    precise_samples = _measure_tflops_samples(
        device,
        allow_reduced_precision=False,
    )
    reduced_precision_samples = _measure_tflops_samples(
        device,
        allow_reduced_precision=True,
    )
    best_precise = max(precise_samples, key=lambda sample: sample["tflops"])
    best_reduced = max(
        reduced_precision_samples,
        key=lambda sample: sample["tflops"],
    )
    theoretical_ceiling_clock_mhz = _maximum_observed_sm_clock_mhz(
        precise_samples,
        reduced_precision_samples,
    )

    return {
        "schema_version": PEAKS_SCHEMA_VERSION,
        "gpu_name": gpu_name,
        "gpu_slug": gpu_slug(device),
        "compute_capability": f"{major}.{minor}",
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "calibration_started_at_utc": calibration_started_at_utc,
        "calibration_preflight_gpu_utilization_pct": list(
            calibration_preflight_gpu_utilization_pct
        ),
        "calibration_preflight_gpu_utilization_limit_pct": (
            calibration_preflight_gpu_utilization_limit_pct
        ),
        "peak_bw_gbps": best_bandwidth["gbps"],
        "bandwidth_elements": best_bandwidth["elements"],
        "bandwidth_samples": bandwidth_samples,
        "cublas_tflops_fp16_fp32acc": best_precise["tflops"],
        "cublas_tflops_fp16_reduced_precision_allowed": best_reduced["tflops"],
        "torch_fp16acc_available": torch_fp16acc_available(),
        "theoretical_tflops_fp16_fp32acc_rated": profile.fp16_fp32acc_tflops(
            profile.rated_boost_mhz
        ),
        "theoretical_tflops_fp16_fp32acc_at_measured_clock": (
            profile.fp16_fp32acc_tflops(theoretical_ceiling_clock_mhz)
        ),
        "theoretical_tflops_fp16_fp16acc_rated": profile.fp16_fp16acc_tflops(
            profile.rated_boost_mhz
        ),
        "theoretical_tflops_fp16_fp16acc_at_measured_clock": (
            profile.fp16_fp16acc_tflops(theoretical_ceiling_clock_mhz)
        ),
        "theoretical_ceiling_sm_clock_mhz": theoretical_ceiling_clock_mhz,
        "tflops_samples": precise_samples,
        "tflops_samples_reduced_precision_allowed": reduced_precision_samples,
        "best_tflops_size": best_precise["n"],
        "best_tflops_sm_clock_mhz": best_precise["sm_clock_mhz"],
        "best_reduced_precision_size": best_reduced["n"],
        "best_reduced_precision_sm_clock_mhz": best_reduced["sm_clock_mhz"],
        "allow_fp16_reduced_precision_reduction": False,
        "theoretical_provenance": {
            "model": gpu_name,
            "rated_boost_clock_mhz": profile.rated_boost_mhz,
            "streaming_multiprocessors": profile.streaming_multiprocessors,
            "tensor_cores_per_sm": profile.tensor_cores_per_sm,
            "fp16_fp32acc_flops_per_clock_per_tensor_core": (
                profile.fp16_fp32acc_flops_per_clock_per_tensor_core
            ),
            "fp16_fp32acc_rated_tflops": profile.fp16_fp32acc_tflops(profile.rated_boost_mhz),
            "fp16_fp16acc_rated_tflops": profile.fp16_fp16acc_tflops(profile.rated_boost_mhz),
            "ceiling_clock_policy": (
                "maximum sm_clock_max_mhz observed across both compute calibration sweeps"
            ),
            "product_spec_url": profile.product_spec_url,
            "architecture_whitepaper_url": profile.architecture_whitepaper_url,
        },
    }


def load_or_measure_peaks(
    results_root: Path,
    *,
    device: int = 0,
    force: bool = False,
    calibration_started_at_utc: str,
    calibration_preflight_gpu_utilization_pct: tuple[int, ...],
    calibration_preflight_gpu_utilization_limit_pct: int,
) -> tuple[PeakResults, Path]:
    """Load a compatible peak cache or calibrate and persist a new one.

    Args:
        results_root: Root directory for per-GPU results.
        device: CUDA device index.
        force: Ignore an existing cache and recalibrate.
        calibration_started_at_utc: ISO-8601 start time for the current run.
        calibration_preflight_gpu_utilization_pct: Utilization samples for the
            current run, persisted if calibration is performed.
        calibration_preflight_gpu_utilization_limit_pct: Maximum permitted
            preflight utilization.

    Returns:
        Peak payload and its JSON path.

    Raises:
        ValueError: If an existing payload uses an incompatible schema.
    """
    output_dir = results_root / gpu_slug(device)
    output_path = output_dir / "peaks.json"
    if output_path.exists() and not force:
        stored = read_json_object(output_path)
        major, minor = torch.cuda.get_device_capability(device)
        _validate_stored_peaks(
            stored,
            path=output_path,
            expected_gpu_name=torch.cuda.get_device_name(device),
            expected_compute_capability=f"{major}.{minor}",
            expected_torch_version=torch.__version__,
            expected_triton_version=triton.__version__,
        )
        return cast(PeakResults, stored), output_path

    peaks = measure_peaks(
        device,
        calibration_started_at_utc=calibration_started_at_utc,
        calibration_preflight_gpu_utilization_pct=(calibration_preflight_gpu_utilization_pct),
        calibration_preflight_gpu_utilization_limit_pct=(
            calibration_preflight_gpu_utilization_limit_pct
        ),
    )
    write_json_atomic(output_path, cast(JsonObject, peaks))
    return peaks, output_path


def _measure_bandwidth_samples(
    device: int,
    *,
    element_counts: tuple[int, ...],
    warmup_ms: int,
    repetition_ms: int,
) -> list[BandwidthSample]:
    """Measure effective bandwidth across several working-set sizes."""
    if not element_counts or any(elements <= 0 for elements in element_counts):
        raise ValueError("element_counts must contain positive integers")
    torch_device = torch.device("cuda", device)
    samples: list[BandwidthSample] = []

    for elements in element_counts:
        source = torch.randn(elements, device=torch_device, dtype=torch.float32)
        destination = torch.empty_like(source)
        milliseconds = float(
            triton.testing.do_bench(
                partial(destination.copy_, source),
                warmup=warmup_ms,
                rep=repetition_ms,
                return_mode="median",
            )
        )
        bytes_read_written = 2 * source.numel() * source.element_size()
        samples.append(
            {
                "elements": elements,
                "working_set_bytes": bytes_read_written,
                "ms": milliseconds,
                "gbps": bytes_read_written / (milliseconds * 1e-3) / 1e9,
            }
        )
    return samples


def _maximum_observed_sm_clock_mhz(
    *sample_groups: list[TflopsSample],
) -> int:
    """Return the highest sampled SM clock across compute calibration sweeps.

    Raises:
        ValueError: If no compute samples are supplied.
    """
    clocks = [sample["sm_clock_max_mhz"] for group in sample_groups for sample in group]
    if not clocks:
        raise ValueError("at least one compute clock sample is required")
    return max(clocks)


def _measure_tflops_samples(
    device: int,
    *,
    sizes: tuple[int, ...] = DEFAULT_TFLOPS_SIZES,
    repetition_ms: int = 200,
    allow_reduced_precision: bool,
) -> list[TflopsSample]:
    """Measure allocation-free cuBLAS GEMM throughput and concurrent clocks."""
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("sizes must contain positive integers")

    torch_device = torch.device("cuda", device)
    samples: list[TflopsSample] = []
    previous_setting = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = allow_reduced_precision
    try:
        for size in sizes:
            left = torch.randn(size, size, device=torch_device, dtype=torch.float16)
            right = torch.randn_like(left)
            output = torch.empty_like(left)
            matmul = partial(torch.mm, left, right, out=output)
            _warm_to_steady_clock(matmul)

            benchmark = partial(
                triton.testing.do_bench,
                matmul,
                warmup=50,
                rep=repetition_ms,
                return_mode="median",
            )
            clock_samples = _measure_with_clock_monitor(
                benchmark,
                device=device,
            )
            milliseconds = float(clock_samples.result)
            clocks = clock_samples.clocks_mhz or [_sm_clock_mhz(device)]
            median_clock = int(statistics.median(clocks))
            samples.append(
                {
                    "n": size,
                    "ms": milliseconds,
                    "tflops": 2 * size**3 / (milliseconds * 1e-3) / 1e12,
                    "sm_clock_mhz": median_clock,
                    "sm_clock_min_mhz": min(clocks),
                    "sm_clock_max_mhz": max(clocks),
                    "sm_clock_samples_mhz": clocks,
                }
            )
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous_setting
    return samples


@dataclass(frozen=True, slots=True)
class _MonitoredResult(Generic[_ResultT]):
    """Result returned by a callable and clocks sampled while it ran."""

    result: _ResultT
    clocks_mhz: list[int]


def _measure_with_clock_monitor(
    fn: Callable[[], _ResultT],
    *,
    device: int,
    interval_seconds: float = 0.05,
) -> _MonitoredResult[_ResultT]:
    """Execute ``fn`` while sampling SM clocks on a background thread."""
    stop = threading.Event()
    clocks: list[int] = []
    errors: list[BaseException] = []

    def monitor() -> None:
        """Sample clocks until the benchmark signals completion."""
        try:
            while not stop.is_set():
                clocks.append(_sm_clock_mhz(device))
                stop.wait(interval_seconds)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=monitor, daemon=True, name="sm-clock-monitor")
    thread.start()
    try:
        result = fn()
    finally:
        stop.set()
        thread.join()
    if errors:
        raise RuntimeError("failed to monitor SM clocks") from errors[0]
    return _MonitoredResult(result=result, clocks_mhz=clocks)


def _warm_to_steady_clock(fn: Callable[[], object], seconds: float = 3.0) -> None:
    """Run a GPU workload until boost clocks reach a steady operating state."""
    if seconds <= 0:
        raise ValueError("warmup duration must be positive")
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        fn()
    torch.cuda.synchronize()


def _sm_clock_mhz(device: int = 0) -> int:
    """Read the current SM clock with ``nvidia-smi``.

    Raises:
        RuntimeError: If ``nvidia-smi`` is missing, fails, or returns an
            unparsable value.
    """
    return _nvidia_smi_integer(
        "clocks.sm",
        device=device,
        error_message="unable to query SM clock with nvidia-smi",
    )


def _nvidia_smi_integer(
    field: str,
    *,
    device: int,
    error_message: str,
) -> int:
    """Query one integer-valued ``nvidia-smi`` field."""
    command = [
        "nvidia-smi",
        f"--query-gpu={field}",
        "--format=csv,noheader,nounits",
        "-i",
        str(device),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        first_line = result.stdout.strip().splitlines()[0]
        return int(first_line)
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        IndexError,
        ValueError,
    ) as error:
        raise RuntimeError(error_message) from error


def _profile_for(gpu_name: str) -> GpuTheoreticalProfile:
    """Return the theoretical profile matching ``gpu_name``.

    Raises:
        ValueError: If the GPU has no audited profile.
    """
    for profile in _GPU_PROFILES:
        if profile.model_substring in gpu_name:
            return profile
    raise ValueError(
        f"no theoretical compute profile for {gpu_name!r}; "
        "add an audited GpuTheoreticalProfile before benchmarking compute kernels"
    )


def _validate_cuda_device(device: int) -> None:
    """Validate that a CUDA device index can be selected."""
    if device < 0:
        raise ValueError("device index must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {device} is unavailable; found {torch.cuda.device_count()} device(s)"
        )


def _validate_stored_peaks(
    payload: JsonObject,
    *,
    path: Path,
    expected_gpu_name: str,
    expected_compute_capability: str,
    expected_torch_version: str,
    expected_triton_version: str,
) -> None:
    """Validate cached peak provenance against the active software and GPU."""
    if payload.get("schema_version") != PEAKS_SCHEMA_VERSION:
        raise ValueError(f"{path}: incompatible peaks schema; rerun with --force-peaks")
    required_fields = {
        "gpu_name",
        "peak_bw_gbps",
        "cublas_tflops_fp16_fp32acc",
        "theoretical_tflops_fp16_fp32acc_at_measured_clock",
        "theoretical_tflops_fp16_fp16acc_at_measured_clock",
        "theoretical_ceiling_sm_clock_mhz",
        "theoretical_provenance",
        "calibration_started_at_utc",
        "calibration_preflight_gpu_utilization_pct",
        "calibration_preflight_gpu_utilization_limit_pct",
    }
    missing = sorted(required_fields.difference(payload))
    if missing:
        raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")

    utilization_samples = payload["calibration_preflight_gpu_utilization_pct"]
    utilization_limit = payload["calibration_preflight_gpu_utilization_limit_pct"]
    if (
        not isinstance(utilization_samples, list)
        or not utilization_samples
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 100
            for value in utilization_samples
        )
    ):
        raise ValueError(f"{path}: invalid calibration GPU-utilization samples")
    validated_utilization_samples = cast(list[int], utilization_samples)
    if (
        not isinstance(utilization_limit, int)
        or isinstance(utilization_limit, bool)
        or utilization_limit < 0
        or utilization_limit > 100
        or max(validated_utilization_samples) > utilization_limit
    ):
        raise ValueError(f"{path}: invalid calibration GPU-utilization limit")

    positive_numeric_fields = {
        "peak_bw_gbps",
        "cublas_tflops_fp16_fp32acc",
        "theoretical_tflops_fp16_fp32acc_at_measured_clock",
        "theoretical_tflops_fp16_fp16acc_at_measured_clock",
        "theoretical_ceiling_sm_clock_mhz",
    }
    invalid_numeric: list[str] = []
    for field in sorted(positive_numeric_fields):
        value = payload[field]
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            invalid_numeric.append(field)
    if invalid_numeric:
        raise ValueError(
            f"{path}: cached peaks contain invalid positive numeric fields: "
            f"{', '.join(invalid_numeric)}"
        )

    provenance = payload["theoretical_provenance"]
    if not isinstance(provenance, dict):
        raise ValueError(f"{path}: theoretical_provenance must be an object")
    required_provenance_fields = {
        "model",
        "rated_boost_clock_mhz",
        "streaming_multiprocessors",
        "tensor_cores_per_sm",
        "fp16_fp32acc_flops_per_clock_per_tensor_core",
        "fp16_fp32acc_rated_tflops",
        "fp16_fp16acc_rated_tflops",
        "ceiling_clock_policy",
        "product_spec_url",
        "architecture_whitepaper_url",
    }
    missing_provenance = sorted(required_provenance_fields.difference(provenance))
    if missing_provenance:
        raise ValueError(
            f"{path}: theoretical_provenance is missing fields: {', '.join(missing_provenance)}"
        )

    expected = {
        "gpu_name": expected_gpu_name,
        "compute_capability": expected_compute_capability,
        "torch_version": expected_torch_version,
        "triton_version": expected_triton_version,
    }
    mismatches = [
        f"{field}={payload.get(field)!r} (expected {value!r})"
        for field, value in expected.items()
        if payload.get(field) != value
    ]
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            f"{path}: cached peaks do not match the active environment: {details}; "
            "rerun with --force-peaks"
        )
