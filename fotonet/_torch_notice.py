"""Helpers for guiding users to install PyTorch outside package dependencies."""

from __future__ import annotations

import importlib.util
import sys


TORCH_INSTALL_MESSAGE = "please download torch for your system at https://pytorch.org"


def is_torch_available():
    return importlib.util.find_spec("torch") is not None


def maybe_print_torch_notice(torch_available=None):
    if torch_available is None:
        torch_available = is_torch_available()
    if not torch_available:
        print(TORCH_INSTALL_MESSAGE, file=sys.stderr)
