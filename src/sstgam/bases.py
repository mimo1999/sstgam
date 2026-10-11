"""
Basis-function and roughness-penalty primitives for the Shared-Shape Temporal
GAM.

Every shape function in this model is a linear combination of fixed basis
functions placed on a regular grid in a *normalised* coordinate. Two basis
kinds are available, both parameterised the same way -- ``make_*_params(lo,
hi, n)`` returns a ``(param_a, param_b)`` pair and ``*_np(z, param_a,
param_b)`` evaluates the resulting ``(..., n)`` design matrix:

- ``"rbf"``: fixed-bandwidth Gaussian bumps on evenly spaced centers.
- ``"bspline"``: a clamped, evenly knotted B-spline basis.

A second-difference penalty on *adjacent* coefficients yields the "penalise
wiggliness" smoothing a penalised spline gives for either basis, so the shape
functions stay interpretable and stable when observations are sparse.

This module is pure numerics and can be unit-tested in isolation.
"""

import numpy as np
from scipy.interpolate import BSpline

_BSPLINE_DEFAULT_DEGREE = 3


def make_rbf_centers(lo: float, hi: float, n: int) -> tuple[np.ndarray, float]:
    """Return ``n`` equally-spaced RBF centers on ``[lo, hi]`` and a bandwidth.

    The bandwidth is set to the center spacing, which gives ~50% neighbour
    overlap — smooth reconstruction without leaving gaps between bumps.
    """
    n = max(int(n), 1)
    centers = np.linspace(lo, hi, n).astype(np.float32)
    spacing = (hi - lo) / (n - 1) if n > 1 else (hi - lo)
    bandwidth = float(spacing) if spacing > 1e-8 else 1.0
    return centers, bandwidth


def rbf_np(z: np.ndarray, centers: np.ndarray, bandwidth: float) -> np.ndarray:
    """NumPy Gaussian design matrix. ``z`` shape (...) -> output (..., P)."""
    d = (np.asarray(z, np.float32)[..., None] - np.asarray(centers, np.float32)) / bandwidth
    return np.exp(-0.5 * d * d).astype(np.float32)


def make_bspline_knots(lo: float, hi: float, n: int) -> tuple[np.ndarray, int]:
    """Return a clamped, evenly spaced knot vector for ``n`` B-spline bases on ``[lo, hi]``.

    The degree is capped at ``n - 1`` so a valid (non-empty) basis exists even
    for a small ``n``. Clamping (knot multiplicity ``degree + 1`` at both
    ends) makes the first and last basis functions interpolate the domain
    boundary, matching how the RBF centers span ``[lo, hi]``.
    """
    n = max(int(n), 1)
    degree = min(_BSPLINE_DEFAULT_DEGREE, n - 1)
    n_interior = n - degree - 1
    interior = np.linspace(lo, hi, n_interior + 2, dtype=np.float64)[1:-1] if n_interior > 0 else (
        np.empty(0, dtype=np.float64)
    )
    knots = np.concatenate([
        np.full(degree + 1, lo, dtype=np.float64),
        interior,
        np.full(degree + 1, hi, dtype=np.float64),
    ]).astype(np.float32)
    return knots, degree


def bspline_np(z: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """NumPy B-spline design matrix.

    ``z`` shape (...) -> output (..., P) where ``P = len(knots) - degree - 1``.
    Points outside ``[knots[degree], knots[-degree - 1]]`` are clamped to the
    boundary, so the basis extrapolates as a constant rather than zeroing out.
    """
    knots = np.asarray(knots, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    clamped = np.clip(z, knots[degree], knots[knots.shape[0] - degree - 1]).ravel()
    basis = BSpline.design_matrix(clamped, knots, degree).toarray()
    return basis.reshape(z.shape + (-1,)).astype(np.float32)


_BASES = {
    "rbf": (make_rbf_centers, rbf_np),
    "bspline": (make_bspline_knots, bspline_np),
}


def make_basis_params(basis: str, lo: float, hi: float, n: int):
    """Dispatch to the parameter builder for ``basis`` ('rbf' or 'bspline')."""
    return _BASES[basis][0](lo, hi, n)


def basis_design(basis: str, z, param_a, param_b) -> np.ndarray:
    """Dispatch to the design-matrix builder for ``basis`` ('rbf' or 'bspline')."""
    return _BASES[basis][1](z, param_a, param_b)


def second_diff_matrix_np(n: int) -> np.ndarray:
    """NumPy second-order difference operator, shape ``(n-2, n)``.

    ``||D c||^2 = sum_i (c_{i-1} - 2 c_i + c_{i+1})^2`` penalises curvature of
    the coefficient sequence — the discrete analogue of a spline's integrated
    squared second derivative. Returns an empty ``(0, n)`` matrix when there
    are too few centers to define curvature.
    """
    return np.diff(np.eye(int(n)), 2, axis=0)
