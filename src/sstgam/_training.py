"""Validation split, bagging and boosting loop shared by both backends."""

import numpy as np
from scipy.special import expit
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


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


def boost_bag(update_terms, snapshot, fit_target, fit_weight, validation_target,
              validation_weight, learning_rate: float, max_rounds: int, patience: int):
    """Boost one bag until validation log loss stops improving for ``patience`` rounds.

    Starts from the base-rate intercept. Each round takes a damped Newton step
    on the intercept, then calls ``update_terms(fit_score, validation_score)``,
    which updates the model terms and both score arrays in place. Returns
    ``(intercept, snapshot(), best_round)`` from the best validation round.
    """
    base_rate = np.clip(np.average(fit_target, weights=fit_weight), 1e-6, 1.0 - 1e-6)
    intercept = float(np.log(base_rate / (1.0 - base_rate)))
    fit_score = np.full(fit_target.size, intercept)
    validation_score = np.full(validation_target.size, intercept)

    best_loss, best, rounds_without_improvement = np.inf, None, 0
    for round_index in range(1, max_rounds + 1):
        probability = expit(fit_score)
        step = learning_rate * (np.sum(fit_weight * (fit_target - probability)) /
                                max(np.sum(fit_weight * probability * (1.0 - probability)), 1e-6))
        intercept += step
        fit_score += step
        validation_score += step
        update_terms(fit_score, validation_score)

        loss = log_loss(validation_target, expit(validation_score),
                        sample_weight=validation_weight, labels=[0, 1])
        if loss < best_loss - 1e-6:
            best_loss, best, rounds_without_improvement = loss, (intercept, snapshot(), round_index), 0
        else:
            rounds_without_improvement += 1
            if rounds_without_improvement >= patience:
                break
    return best if best is not None else (intercept, snapshot(), 0)
