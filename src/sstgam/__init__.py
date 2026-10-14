"""Shared-Shape Temporal GAMs for longitudinal data: a binary classifier and a regressor."""

from importlib.metadata import PackageNotFoundError, version

from .classifier import SSTGAM, Explanation, SSTGAMRegressor
from .preprocessing import bin_timestamps

try:
    __version__ = version("sstgam")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["SSTGAM", "Explanation", "SSTGAMRegressor", "__version__", "bin_timestamps"]
