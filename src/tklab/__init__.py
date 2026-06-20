"""Public package API for Triton Kernel Lab."""

from importlib.metadata import PackageNotFoundError, version

from tklab.registry import REGISTRY, KernelSpec, register

try:
    __version__ = version("tklab")
except PackageNotFoundError:
    __version__ = "0.0.0+uninstalled"

__all__ = ["REGISTRY", "KernelSpec", "__version__", "register"]
