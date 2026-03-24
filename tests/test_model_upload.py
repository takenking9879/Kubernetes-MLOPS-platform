"""Unit tests for model architecture upload validation (AST-based).

Tests the _validate_architecture() function in model_architectures.py
without requiring S3, network access, or a running FastAPI server.
"""
import sys
from pathlib import Path

import pytest

# Add backend to path so imports work from tests/
_BACKEND = Path(__file__).resolve().parent.parent / "app" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from routers.model_architectures import _validate_architecture  # noqa: E402
from fastapi import HTTPException  # noqa: E402

# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_MODULE = """\
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_dim: int = 14, num_classes: int = 6):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
"""

VALID_DOTTED_BASE = """\
import torch.nn as nn

class AnotherModel(torch.nn.Module):
    def forward(self, x):
        return x
"""

MISSING_FORWARD = """\
import torch.nn as nn

class MyModel(nn.Module):
    def fit(self, x):
        return x
"""

NOT_A_MODULE = """\
class MyModel:
    def forward(self, x):
        return x
"""

SYNTAX_ERROR = "class MyModel(nn.Module\n"

TWO_CLASSES_ONE_VALID = """\
import torch.nn as nn

class Helper:
    pass

class RealModel(nn.Module):
    def forward(self, x):
        return x
"""


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_valid_module_passes():
    """A well-formed nn.Module subclass with forward() should not raise."""
    _validate_architecture(VALID_MODULE, "model.py")  # must not raise


def test_valid_dotted_base_passes():
    """torch.nn.Module (dotted) base is also accepted."""
    _validate_architecture(VALID_DOTTED_BASE, "model.py")


def test_missing_forward_raises_422():
    """A class inheriting nn.Module but missing forward() should raise 422."""
    with pytest.raises(HTTPException) as exc_info:
        _validate_architecture(MISSING_FORWARD, "bad.py")
    assert exc_info.value.status_code == 422
    assert "forward()" in exc_info.value.detail


def test_not_a_module_raises_422():
    """A plain Python class (not inheriting nn.Module) should raise 422."""
    with pytest.raises(HTTPException) as exc_info:
        _validate_architecture(NOT_A_MODULE, "bad.py")
    assert exc_info.value.status_code == 422
    assert "nn.Module" in exc_info.value.detail


def test_syntax_error_raises_422():
    """Invalid Python syntax should raise 422 with 'syntax error' in detail."""
    with pytest.raises(HTTPException) as exc_info:
        _validate_architecture(SYNTAX_ERROR, "broken.py")
    assert exc_info.value.status_code == 422
    assert "syntax" in exc_info.value.detail.lower()


def test_two_classes_one_valid_passes():
    """If at least one nn.Module subclass with forward() exists, the file is valid."""
    _validate_architecture(TWO_CLASSES_ONE_VALID, "mixed.py")


def test_empty_file_raises_422():
    """An empty file has no nn.Module subclass — should raise 422."""
    with pytest.raises(HTTPException) as exc_info:
        _validate_architecture("", "empty.py")
    assert exc_info.value.status_code == 422
