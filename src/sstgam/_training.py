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
