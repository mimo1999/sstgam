"""Outcome families, validation split and bootstrap bagging shared by both backends."""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import train_test_split

_MAX_LOG_MEAN = 50.0  # keeps exp() finite while a Poisson fit is still far off


class Family(NamedTuple):
    """A GLM outcome family with its canonical link.

    With a canonical link the gradient of the log-likelihood in the linear
    predictor ``eta`` is always ``y - mean(eta)`` and the curvature is
    ``variance(mean(eta))``, so both backends need only these four functions.
    """

    mean: Callable[[np.ndarray], np.ndarray]  # inverse link: eta -> E[y]
    variance: Callable[[np.ndarray], np.ndarray]  # d mean / d eta, as a function of the mean
    loss: Callable[[np.ndarray, np.ndarray], np.ndarray]  # per-row negative log-likelihood (eta, y), up to a constant
    link: Callable[[float], float]  # E[y] -> eta, for the starting intercept


FAMILIES = {
    "binomial": Family(
        mean=expit,
        variance=lambda mu: mu * (1.0 - mu),
        loss=lambda eta, y: np.logaddexp(0.0, eta) - y * eta,
        link=lambda mu: float(logit(np.clip(mu, 1e-6, 1.0 - 1e-6))),
    ),
    "gaussian": Family(
        mean=lambda eta: eta,
        variance=np.ones_like,
        loss=lambda eta, y: 0.5 * (y - eta) ** 2,
        link=float,
    ),
    "poisson": Family(
        mean=lambda eta: np.exp(np.minimum(eta, _MAX_LOG_MEAN)),
        variance=lambda mu: mu,
        loss=lambda eta, y: np.exp(np.minimum(eta, _MAX_LOG_MEAN)) - y * eta,
        link=lambda mu: float(np.log(max(mu, 1e-6))),
    ),
}


def make_validation_split(target: np.ndarray, fraction: float, random_state: int):
    """Create a reproducible train/validation split.

    Stratification is used when both classes have enough samples. Otherwise a
    random split is used so small or highly imbalanced datasets remain usable.
    """
    target = np.asarray(target).ravel()
    n_samples = target.size
    if n_samples < 2:
        raise ValueError("At least two samples are required to create a validation split.")
    if not 0.0 < fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    indices = np.arange(n_samples)
    n_validation = min(max(1, int(round(n_samples * fraction))), n_samples - 1)
    _, counts = np.unique(target, return_counts=True)
    can_stratify = counts.size > 1 and counts.min() >= 2 and n_validation >= counts.size
    return train_test_split(
        indices,
        test_size=n_validation,
        random_state=random_state,
        stratify=target if can_stratify else None,
    )


def bootstrap_bags(train_indices: np.ndarray, n_bags: int, rng: np.random.RandomState):
    """Yield each bag's training rows: the full split for one bag, bootstrap resamples otherwise."""
    for _ in range(n_bags):
        yield (train_indices if n_bags == 1
               else rng.choice(train_indices, size=train_indices.size, replace=True))


def percentile_band(samples: np.ndarray):
    """``(lower, mean, upper)`` over the first axis: the 2.5/97.5 percentiles across bags and their mean."""
    lower, upper = np.percentile(samples, [2.5, 97.5], axis=0)
    return lower, samples.mean(axis=0), upper


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Create deduplicated quantile bin edges with at least one bin."""
    if values.size == 0:
        return np.array([0.0, 1.0])
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size == 1:
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    return edges.astype(np.float64)


def distinct_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin edges ``[low, *thresholds, high]`` that never merge distinct values needlessly.

    A value ``v`` falls in bin ``searchsorted(thresholds, v, "right")``. With at most ``n_bins`` distinct values each
    gets its own bin (thresholds at the midpoints), so binary and few-level covariates are kept apart. Otherwise the
    thresholds are the distinct interior quantiles above the minimum; one at the maximum (a mass point there) is
    kept, so the maximum is not lumped into the bin below it. Always at least one bin.
    """
    if values.size == 0:
        return np.array([0.0, 1.0])
    distinct = np.unique(values)
    if distinct.size == 1:
        return np.array([distinct[0] - 0.5, distinct[0] + 0.5])
    if distinct.size <= n_bins:
        return np.concatenate([[distinct[0]], (distinct[:-1] + distinct[1:]) / 2.0, [distinct[-1]]]).astype(np.float64)
    thresholds = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))[1:-1])
    thresholds = thresholds[thresholds > distinct[0]]
    high = np.nextafter(distinct[-1], np.inf) if thresholds.size and thresholds[-1] == distinct[-1] else distinct[-1]
    return np.concatenate([[distinct[0]], thresholds, [high]]).astype(np.float64)


def pair_cells(values, observed, static_values, pair):
    """Raw ``(u, w, mask)``, each ``(n, cells)``, for a pair of axes ``("temporal", k)`` / ``("static", j)``.

    Signal x signal: one cell per time, used when both were measured. Signal
    x static: one cell per time the signal was measured, the static value
    repeated. Static x static: one cell per row. A temporal axis comes first.
    """
    (kind_u, u_index), (kind_w, w_index) = pair
    if kind_u == "static":
        return (static_values[:, u_index, None], static_values[:, w_index, None],
                np.ones((static_values.shape[0], 1)))
    u, mask = values[:, u_index, :], observed[:, u_index, :]
    if kind_w == "temporal":
        return u, values[:, w_index, :], mask * observed[:, w_index, :]
    return u, np.broadcast_to(static_values[:, w_index, None], u.shape), mask


def fast_gain(gradient, hessian, cells, mask, n_u: int, n_w: int) -> float:
    """FAST pair score (Lou et al., KDD 2013) on the main-effects residual.

    Bins each row's gradient and curvature into the pair's ``(n_u, n_w)``
    grid (``cells`` holds ``u_bin * n_w + w_bin``), then scores the best
    split into four quadrants by one cut on each axis: the Newton loss
    reduction ``sum G^2/H`` over the quadrants minus that of no split.
    """
    def grid(row_values):
        return np.bincount(cells.ravel(), (row_values[:, None] * mask).ravel(), n_u * n_w).reshape(n_u, n_w)

    if n_u < 2 or n_w < 2:
        return 0.0
    gradient_sums = grid(gradient).cumsum(0).cumsum(1)
    hessian_sums = grid(hessian).cumsum(0).cumsum(1)

    def quadrants(sums):
        low_low, low, left, total = sums[:-1, :-1], sums[:-1, -1:], sums[-1:, :-1], sums[-1, -1]
        return low_low, low - low_low, left - low_low, total - low - left + low_low

    gain = sum(g ** 2 / np.maximum(h, 1e-12) for g, h in zip(quadrants(gradient_sums), quadrants(hessian_sums)))
    return float(gain.max() - gradient_sums[-1, -1] ** 2 / max(hessian_sums[-1, -1], 1e-12))


def center_pair(table: np.ndarray, weight: np.ndarray, sweeps: int = 10) -> np.ndarray:
    """Remove a pair table's weighted row and column means, leaving pure interaction (EBM's purification).

    ``weight`` is the data count per cell. Alternating the two projections
    converges to zero weighted marginals; a few sweeps are plenty.
    """
    table = table.copy()
    for axis in (1, 0) * sweeps:
        totals = weight.sum(axis=axis)
        means = np.divide((weight * table).sum(axis=axis), totals, out=np.zeros_like(totals), where=totals > 0)
        table -= np.expand_dims(means, axis)
    return table
