"""Shared-Shape Temporal GAM classifiers for binary longitudinal data."""

from importlib.metadata import PackageNotFoundError, version

from .classifier import SSTGAM

try:
    __version__ = version("sstgam")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["SSTGAM", "__version__"]
