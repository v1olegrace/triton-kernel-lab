"""Kernel package.

Importing this module registers every built-in kernel specification.
"""

from tklab.kernels.fused_softmax import SOFTMAX
from tklab.kernels.layer_norm import LAYER_NORM
from tklab.kernels.matmul import MATMUL_FP16ACC, MATMUL_FP32ACC
from tklab.kernels.vector_add import VECTOR_ADD

__all__ = ["LAYER_NORM", "MATMUL_FP16ACC", "MATMUL_FP32ACC", "SOFTMAX", "VECTOR_ADD"]
