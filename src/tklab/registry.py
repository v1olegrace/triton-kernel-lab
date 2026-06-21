"""Kernel registry and shared contracts.

The registry is the architectural center of the project. A :class:`KernelSpec`
contains every operation required by correctness tests and benchmarks, which
keeps kernel modules declarative and prevents test/benchmark drift.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import torch

TensorArgs: TypeAlias = tuple[torch.Tensor, ...]
BenchmarkScalar: TypeAlias = str | int | float | bool
InputFn: TypeAlias = Callable[[int, torch.device, torch.dtype], TensorArgs]
AdversarialFn: TypeAlias = Callable[[torch.device], TensorArgs]
TensorFn: TypeAlias = Callable[..., torch.Tensor]
LaunchFn: TypeAlias = Callable[[TensorArgs, torch.Tensor], None]
OutputFn: TypeAlias = Callable[[TensorArgs], torch.Tensor]
CostFn: TypeAlias = Callable[[int, torch.dtype], int]
AssertFn: TypeAlias = Callable[[torch.Tensor, torch.Tensor], None]
BenchmarkMetadataFn: TypeAlias = Callable[
    [int, torch.dtype],
    Mapping[str, BenchmarkScalar],
]
BenchmarkCall: TypeAlias = Callable[[], object]
BenchmarkCallFactory: TypeAlias = Callable[[TensorArgs, torch.Tensor], BenchmarkCall]
Bound: TypeAlias = Literal["memory", "compute"]
ComputeMode: TypeAlias = Literal[
    "fp16_fp32acc",
    "fp16_fp16acc",
    "attention_fp16_fp32acc",
]


@dataclass(frozen=True, slots=True)
class KernelSpec:
    """Describe how to execute, validate, and benchmark one kernel.

    Args:
        name: Stable identifier used in tests and result filenames.
        description: Human-readable summary of the operation.
        triton_fn: User-facing wrapper that allocates and returns the output.
        launch_fn: Allocation-free launcher used by the benchmark harness.
        ref_fn: High-precision PyTorch reference used for correctness.
        make_inputs: Factory for benchmark and correctness inputs.
        make_output: Allocation function for the launcher's output buffer.
        sizes: Benchmark sizes. Their meaning is kernel-specific.
        bound: Whether the operation is memory- or compute-bound.
        bytes_moved: Effective byte-cost model for memory-bound kernels.
        flops: FLOP-cost model for compute-bound kernels.
        dtypes: Input dtypes supported by this specification.
        correctness_sizes: Optional reduced set used by expensive GPU tests.
        make_adversarial: Factory for a tail/stride edge case.
        naive_fn: Optional pedagogical baseline, such as unfused softmax.
        reference_launch_fn: Optional allocation-free PyTorch baseline.
        assert_fn: Optional kernel-specific numerical assertion.
        benchmark_metadata: Optional metadata captured after autotuning.
        benchmark_call_factory: Optional factory that prepares auxiliary
            buffers outside the timed region and returns the Triton call.
        reference_call_factory: Optional equivalent for the PyTorch baseline.
        naive_call_factory: Optional equivalent for a multi-output or
            allocation-controlled naive baseline.
        compute_mode: Accumulation mode for compute-bound kernels.
        supports_interpreter: Whether ``TRITON_INTERPRET=1`` is supported.
        rtol: Optional relative tolerance overriding the dtype default.
        atol: Optional absolute tolerance overriding the dtype default.

    Raises:
        ValueError: If required fields are missing or mutually inconsistent.
    """

    name: str
    description: str
    triton_fn: TensorFn
    launch_fn: LaunchFn
    ref_fn: TensorFn
    make_inputs: InputFn
    make_output: OutputFn
    sizes: Sequence[int]
    bound: Bound
    bytes_moved: CostFn | None = None
    flops: CostFn | None = None
    dtypes: Sequence[torch.dtype] = field(default_factory=lambda: (torch.float16,))
    correctness_sizes: Sequence[int] | None = None
    make_adversarial: AdversarialFn | None = None
    naive_fn: TensorFn | None = None
    reference_launch_fn: LaunchFn | None = None
    assert_fn: AssertFn | None = None
    benchmark_metadata: BenchmarkMetadataFn | None = None
    benchmark_call_factory: BenchmarkCallFactory | None = None
    reference_call_factory: BenchmarkCallFactory | None = None
    naive_call_factory: BenchmarkCallFactory | None = None
    compute_mode: ComputeMode | None = None
    supports_interpreter: bool = True
    rtol: float | None = None
    atol: float | None = None

    def __post_init__(self) -> None:
        """Validate specification invariants after construction."""
        object.__setattr__(self, "sizes", tuple(self.sizes))
        object.__setattr__(
            self,
            "correctness_sizes",
            None if self.correctness_sizes is None else tuple(self.correctness_sizes),
        )
        object.__setattr__(self, "dtypes", tuple(self.dtypes))
        if not self.name or not self.name.isidentifier():
            raise ValueError("kernel name must be a non-empty Python identifier")
        if not self.description.strip():
            raise ValueError(f"{self.name}: description must not be empty")
        self._validate_sizes(self.sizes, field_name="sizes")
        if self.correctness_sizes is not None:
            self._validate_sizes(self.correctness_sizes, field_name="correctness_sizes")
        if not self.dtypes:
            raise ValueError(f"{self.name}: at least one dtype is required")
        if any(not isinstance(dtype, torch.dtype) for dtype in self.dtypes):
            raise ValueError(f"{self.name}: dtypes must contain torch.dtype values")
        if len(set(self.dtypes)) != len(self.dtypes):
            raise ValueError(f"{self.name}: dtypes must not contain duplicates")
        if self.bound == "memory":
            if self.bytes_moved is None:
                raise ValueError(f"{self.name}: memory-bound kernels require bytes_moved")
            if self.flops is not None or self.compute_mode is not None:
                raise ValueError(
                    f"{self.name}: memory-bound kernels cannot define compute metadata"
                )
        elif self.bound == "compute":
            if self.flops is None or self.compute_mode is None:
                raise ValueError(
                    f"{self.name}: compute-bound kernels require flops and compute_mode"
                )
            if self.compute_mode not in {
                "fp16_fp32acc",
                "fp16_fp16acc",
                "attention_fp16_fp32acc",
            }:
                raise ValueError(f"{self.name}: unsupported compute mode {self.compute_mode!r}")
        else:
            raise ValueError(f"{self.name}: unsupported bound {self.bound!r}")

    def validation_sizes(self) -> Sequence[int]:
        """Return the sizes used by correctness tests."""
        return self.correctness_sizes or self.sizes

    @staticmethod
    def _validate_sizes(sizes: Sequence[int], *, field_name: str) -> None:
        """Validate a benchmark or correctness size sequence.

        Args:
            sizes: Sequence to validate.
            field_name: Field name included in error messages.

        Raises:
            ValueError: If the sequence is empty, non-positive, or duplicated.
        """
        if not sizes or any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in sizes
        ):
            raise ValueError(f"{field_name} must contain positive integers")
        if len(set(sizes)) != len(sizes):
            raise ValueError(f"{field_name} must not contain duplicates")


REGISTRY: dict[str, KernelSpec] = {}


def register(spec: KernelSpec) -> KernelSpec:
    """Register a kernel specification.

    Args:
        spec: Validated specification to add.

    Returns:
        The same specification, enabling module-level assignment.

    Raises:
        ValueError: If another kernel already uses ``spec.name``.
    """
    if spec.name in REGISTRY:
        raise ValueError(f"duplicate kernel: {spec.name}")
    REGISTRY[spec.name] = spec
    return spec
