"""Normalization helpers for longitudinal and static values."""

import numpy as np


def robust_center_scale(values: np.ndarray, observed: np.ndarray):
    """Return per-feature median centres and robust scales."""
    n_features = values.shape[1]
    centers = np.zeros(n_features, dtype=np.float32)
    scales = np.ones(n_features, dtype=np.float32)
    for feature_index in range(n_features):
        feature_values = values[:, feature_index, :][observed[:, feature_index, :] > 0.5]
        feature_values = feature_values[np.isfinite(feature_values)]
        if feature_values.size == 0:
            continue
        centers[feature_index] = np.median(feature_values)
        lower, upper = np.percentile(feature_values, [25, 75])
        scale = (upper - lower) / 1.349
        if scale <= 1e-8:
            scale = feature_values.std()
        scales[feature_index] = scale if scale > 1e-8 else 1.0
    return centers, scales


def fit_quantile_tables(values: np.ndarray, observed: np.ndarray):
    """Fit empirical CDF tables that map each feature into ``(-1, 1)``."""
    tables = []
    for feature_index in range(values.shape[1]):
        feature_values = values[:, feature_index, :][observed[:, feature_index, :] > 0.5]
        feature_values = feature_values[np.isfinite(feature_values)]
        if feature_values.size < 2:
            tables.append((np.array([-1.0, 1.0]), np.array([-1.0, 1.0])))
            continue

        unique_values, counts = np.unique(feature_values, return_counts=True)
        # Mean mid-rank of each tied group in the sorted values.
        unique_ranks = (np.cumsum(counts) - counts / 2.0) / feature_values.size
        tables.append((unique_values.astype(np.float64), (2.0 * unique_ranks - 1.0).astype(np.float64)))
    return tables


def static_stats(values: np.ndarray):
    """Return per-column finite-value means and standard deviations."""
    means = np.zeros(values.shape[1], dtype=np.float32)
    scales = np.ones(values.shape[1], dtype=np.float32)
    for column_index in range(values.shape[1]):
        column = values[:, column_index]
        column = column[np.isfinite(column)]
        if column.size:
            means[column_index] = column.mean()
            standard_deviation = column.std()
            scales[column_index] = standard_deviation if standard_deviation > 1e-8 else 1.0
    return means, scales
