"""QLoRA capability detection for SynthQuanta.

True QLoRA requires:
  1. bitsandbytes library (4-bit NF4/FP4 base weight quantization)
  2. CUDA-capable GPU (bitsandbytes CPU support is limited/experimental)

If either condition is not met, this module raises QLoRAError with a
useful diagnostic.  The system NEVER silently falls back to LoRA when
QLoRA is requested (§28 of the implementation spec, §9 of CLAUDE.md).
"""
from __future__ import annotations


class QLoRAError(Exception):
    """Raised when QLoRA cannot run in the current environment."""


def check_qlora_capability() -> None:
    """Verify QLoRA can run.  Raises QLoRAError with diagnostic if it cannot.

    Call this before attempting any QLoRA training.  Do NOT catch and
    suppress the error — let it propagate so the job is marked FAILED.
    """
    import torch  # noqa: PLC0415 — local import to defer module load

    # Check 1: CUDA availability
    if not torch.cuda.is_available():
        device_info = _describe_device()
        raise QLoRAError(
            "QLoRA requires a CUDA-capable GPU.  "
            f"Current device: {device_info}.  "
            "To train on this machine, use method='lora' or method='full' instead."
        )

    # Check 2: bitsandbytes availability
    try:
        import bitsandbytes  # noqa: F401, PLC0415
    except ImportError:
        raise QLoRAError(
            "QLoRA requires the 'bitsandbytes' library, which is not installed.  "
            "Install it with: pip install bitsandbytes  "
            "(Note: bitsandbytes requires a CUDA GPU.)"
        )

    # Check 3: bitsandbytes CUDA support actually works
    try:
        import bitsandbytes as bnb  # noqa: PLC0415
        _bnb_version = getattr(bnb, "__version__", "unknown")
    except Exception as exc:
        raise QLoRAError(
            f"bitsandbytes import succeeded but initialization failed: {exc}.  "
            "Ensure the bitsandbytes version is compatible with your CUDA version."
        )


def qlora_diagnostic() -> dict:
    """Return a diagnostic dict for reporting QLoRA environment state."""
    import torch  # noqa: PLC0415

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None

    try:
        import bitsandbytes as bnb  # noqa: PLC0415
        bnb_version = bnb.__version__
        bnb_available = True
    except ImportError:
        bnb_version = None
        bnb_available = False

    return {
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "bitsandbytes_available": bnb_available,
        "bitsandbytes_version": bnb_version,
        "qlora_supported": cuda_available and bnb_available,
    }


def _describe_device() -> str:
    import torch  # noqa: PLC0415
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    return "CPU only"
