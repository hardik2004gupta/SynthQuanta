"""Detect the best available PyTorch INT8 quantization backend for this platform."""
from __future__ import annotations

import platform

_KNOWN_BACKENDS = ("fbgemm", "qnnpack", "onednn", "x86")


def detect_quantized_backend() -> str:
    """Return the best available PyTorch quantization backend.

    Priority: use PyTorch's own default engine for this build, which is always
    the best choice since it's compiled in.  Falls back to platform heuristics
    only when PyTorch reports an unrecognized default.

    fbgemm / onednn / x86: x86 (Linux / Windows)
    qnnpack:               ARM (Apple Silicon, mobile)
    """
    import torch

    # Ask PyTorch what its default engine is — this is always the safest choice.
    current = torch.backends.quantized.engine
    if current and current in _KNOWN_BACKENDS:
        return current

    # Fallback: platform heuristics
    machine = platform.machine().lower()
    system = platform.system()

    if machine in ("arm64", "aarch64"):
        return "qnnpack"
    if system == "Darwin" and machine == "arm64":
        return "qnnpack"

    return "fbgemm"


def available_backends() -> list[str]:
    """Return all INT8 quantization backends supported by this PyTorch build.

    The current (default) backend is always listed; additional backends are
    probed by attempting to set them without affecting the running state.
    """
    import torch

    supported: list[str] = []

    # The current engine is definitely available
    current = torch.backends.quantized.engine
    if current:
        supported.append(current)

    # Probe additional backends without permanently changing the engine
    for b in _KNOWN_BACKENDS:
        if b == current:
            continue
        try:
            orig = torch.backends.quantized.engine
            torch.backends.quantized.engine = b
            torch.backends.quantized.engine = orig
            supported.append(b)
        except RuntimeError:
            pass

    return supported
