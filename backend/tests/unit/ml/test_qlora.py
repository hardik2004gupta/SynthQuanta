"""Unit tests for QLoRA capability detection.

On this machine (CPU only, no bitsandbytes), check_qlora_capability()
must raise QLoRAError — never silently succeed or fall back.
"""
import pytest

from app.ml.adapters.qlora import QLoRAError, check_qlora_capability, qlora_diagnostic


def test_check_qlora_raises_on_cpu_only():
    """QLoRA must fail with a clear diagnostic on CPU-only machines."""
    with pytest.raises(QLoRAError) as exc_info:
        check_qlora_capability()
    msg = str(exc_info.value)
    # Must explain WHY it failed and what to use instead
    assert "CUDA" in msg or "bitsandbytes" in msg


def test_qlora_error_is_not_caught():
    """QLoRAError must propagate; verify it is not a subclass of ValueError or similar."""
    assert issubclass(QLoRAError, Exception)
    assert not issubclass(QLoRAError, ValueError)


def test_qlora_diagnostic_returns_dict():
    diag = qlora_diagnostic()
    assert isinstance(diag, dict)
    assert "cuda_available" in diag
    assert "bitsandbytes_available" in diag
    assert "qlora_supported" in diag


def test_qlora_diagnostic_false_on_cpu():
    diag = qlora_diagnostic()
    # On a CPU-only machine, qlora_supported must be False
    assert diag["qlora_supported"] is False
    assert diag["cuda_available"] is False
