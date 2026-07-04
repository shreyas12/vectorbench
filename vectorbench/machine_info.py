"""Collect machine and dependency-version info for reproducibility and report disclaimers."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import faiss
import psutil

from . import __version__


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def collect_machine_info() -> dict:
    """CPU, RAM, and interpreter/library versions. Laptop-grade; background load uncontrolled."""
    return {
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "python": sys.version.split()[0],
        "faiss": faiss.__version__,
        "numpy": _pkg_version("numpy"),
        "torch": _pkg_version("torch"),
        "sentence_transformers": _pkg_version("sentence-transformers"),
        "vectorbench": __version__,
    }
