# SPDX-License-Identifier: Apache-2.0
"""
vllm-mlx: Apple Silicon MLX backend for vLLM

This package provides native Apple Silicon GPU acceleration for vLLM
using Apple's MLX framework, mlx-lm for LLMs, and mlx-vlm for
vision-language models.
"""

__version__ = "0.1.0"

from vllm_mlx.platform import MLXPlatform
from vllm_mlx.worker import MLXWorker
from vllm_mlx.model_runner import MLXModelRunner
from vllm_mlx.attention import MLXAttentionBackend

__all__ = [
    "MLXPlatform",
    "MLXWorker",
    "MLXModelRunner",
    "MLXAttentionBackend",
    "__version__",
]
