"""Reproducibility utilities for SynthQuanta training.

Sets seeds across Python, NumPy, and PyTorch so that runs with the same
seed produce the same initial model weights and (where the framework
guarantees it) the same gradient updates.

Important limitation: PyTorch does not guarantee bit-for-bit reproducibility
across different hardware or CUDA versions.  We record all relevant context
so experiments can be re-run under the same conditions.
"""
from __future__ import annotations

import platform
import random
import sys

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_hardware_info() -> dict:
    """Capture the hardware and software context for experiment records."""
    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else None
    gpu_count = torch.cuda.device_count() if cuda else 0

    return {
        "device": "cuda" if cuda else "cpu",
        "cuda_available": cuda,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": None,  # avoid psutil dependency
    }
