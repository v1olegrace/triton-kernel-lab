"""Kernel package.

Importing this module registers every built-in kernel specification.
"""

from tklab.kernels.activations import GELU, RELU, SILU, TANH
from tklab.kernels.flash_attention import ATTENTION_CAUSAL, ATTENTION_NONCAUSAL
from tklab.kernels.fused_softmax import SOFTMAX
from tklab.kernels.layer_norm import LAYER_NORM
from tklab.kernels.matmul import MATMUL_FP16ACC, MATMUL_FP32ACC
from tklab.kernels.residual_rms_norm import RESIDUAL_RMS_NORM
from tklab.kernels.rms_norm import RMS_NORM
from tklab.kernels.rope import ROPE
from tklab.kernels.swiglu import SWIGLU
from tklab.kernels.vector_add import VECTOR_ADD

__all__ = [
    "ATTENTION_CAUSAL",
    "ATTENTION_NONCAUSAL",
    "GELU",
    "LAYER_NORM",
    "MATMUL_FP16ACC",
    "MATMUL_FP32ACC",
    "RELU",
    "RESIDUAL_RMS_NORM",
    "RMS_NORM",
    "ROPE",
    "SILU",
    "SOFTMAX",
    "SWIGLU",
    "TANH",
    "VECTOR_ADD",
]
