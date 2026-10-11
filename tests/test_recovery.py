"""Ground-truth correctness checks: does SSTGAM recover a known generating
process, does bagging reduce prediction variance, does the tree backend's
adaptive split-order logic pick the axis that actually carries the signal,
and is the model calibrated on data drawn from its own assumed form.

These complement ``test_sstgam.py``'s shape/API smoke tests with checks
against a known ground truth.
"""

import numpy as np
import pandas as pd
import pytest

from sstgam import SSTGAM


def test_spline_recovers_known_value_time_surface():
    generator = np.random.RandomState(101)
    n_samples = 900
    n_times = 4
    time_ids = np.arange(n_times)
    day_weight = np.linspace(-1.0, 1.0, n_times)  # true effect sign flips over time

    raw_values = generator.normal(size=(n_samples, n_times))
    observed = generator.random_sample((n_samples, n_times)) > 0.1
    contribution = np.where(observed, day_weight[None, :] * raw_values, 0.0)
    logit = 0.1 + contribution.sum(axis=1)
    probability_true = 1.0 / (1.0 + np.exp(-logit))
    target = generator.binomial(1, probability_true)

    values_with_nan = np.where(observed, raw_values, np.nan)
    X = pd.DataFrame({f"marker_{t}": values_with_nan[:, t] for t in time_ids})

    model = SSTGAM(
        method="spline", n_value_bases=8, n_time_bases=5,
        max_rounds=150, early_stopping_rounds=25, random_state=0,
    ).fit(X, target)

    value_grid = np.array([-2.0, -1.0, 1.0, 2.0])
    fitted_surface = model.temporal_surface("marker", value_grid, time_ids)
    true_surface = day_weight[None, :] * value_grid[:, None]

    correlation = np.corrcoef(fitted_surface.ravel(), true_surface.ravel())[0, 1]
    assert correlation > 0.85


def test_tree_recovers_known_value_time_surface():
    generator = np.random.RandomState(102)
    n_samples = 1200
    n_times = 4
    time_ids = np.arange(n_times)
    day_weight = np.linspace(-1.0, 1.0, n_times)

    raw_values = generator.normal(size=(n_samples, n_times))
    observed = generator.random_sample((n_samples, n_times)) > 0.1
    contribution = np.where(observed, day_weight[None, :] * raw_values, 0.0)
    logit = 0.1 + contribution.sum(axis=1)
    probability_true = 1.0 / (1.0 + np.exp(-logit))
    target = generator.binomial(1, probability_true)

    values_with_nan = np.where(observed, raw_values, np.nan)
    X = pd.DataFrame({f"marker_{t}": values_with_nan[:, t] for t in time_ids})

    model = SSTGAM(
        method="tree", n_value_bins=10, tree_depth=3,
        max_rounds=150, early_stopping_rounds=25, random_state=0,
    ).fit(X, target)

    edges, table = model.value_time_table("marker")
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    true_table = day_weight[None, :] * bin_centers[:, None]

    correlation = np.corrcoef(table.ravel(), true_table.ravel())[0, 1]
    assert correlation > 0.6


def test_bagging_reduces_prediction_variance():
    """Averaging more bootstrap-resampled bags should reduce prediction
    variance, the standard ``1/k`` bagging variance-reduction law.

    Fits once with many bags on a single dataset, then reconstructs each
    bag's own (unaveraged) prediction via ``predict_proba_per_bag`` and
    resamples groups of bags to show variance shrinks as group size grows --
    isolating bagging's own effect from unrelated sources of variation (e.g.
    which rows land in a train/validation split).
    """
    generator = np.random.RandomState(7)
    n_samples = 200
    first = generator.normal(size=n_samples)
    second = first + generator.normal(scale=0.3, size=n_samples)
    static = generator.normal(size=n_samples)
    target = (first + static + generator.normal(scale=0.6, size=n_samples) > 0).astype(int)
    X = pd.DataFrame({"signal_0": first, "signal_3": second, "context": static})

    model = SSTGAM(
        method="spline", max_rounds=60, early_stopping_rounds=15,
        outer_bags=24, random_state=3,
    ).fit(X, target)

    test_X = X.iloc[:60].reset_index(drop=True)
    values, observed, static_values = model._build_tensors(model._prepare_predict_input(test_X))
    per_bag_probability = model.model_.predict_proba_per_bag(values, observed, static_values)
    n_bags = per_bag_probability.shape[0]

    def variance_for_group_size(group_size, rng, n_trials=300):
        grouped_predictions = np.stack([
            per_bag_probability[rng.choice(n_bags, size=group_size, replace=True)].mean(axis=0)
            for _ in range(n_trials)
        ], axis=0)
        return grouped_predictions.var(axis=0).mean()

    variance_one_bag = variance_for_group_size(1, np.random.RandomState(0))
    variance_eight_bags = variance_for_group_size(8, np.random.RandomState(1))

    assert variance_eight_bags < variance_one_bag


def test_tree_backend_axis_order_follows_the_signal():
    def build_dataset(mode, generator):
        n_samples = 1500
        n_times = 4
        day_weight = np.linspace(-3.0, 3.0, n_times)
        raw_values = generator.normal(size=(n_samples, n_times))
        observed = generator.random_sample((n_samples, n_times)) > 0.2

        if mode == "value":
            contribution = np.where(observed, 1.5 * raw_values, 0.0)
        else:
            contribution = np.where(observed, day_weight[None, :], 0.0)

        logit = contribution.sum(axis=1)
        probability_true = 1.0 / (1.0 + np.exp(-logit))
        target = generator.binomial(1, probability_true)
        values_with_nan = np.where(observed, raw_values, np.nan)
        X = pd.DataFrame({f"marker_{t}": values_with_nan[:, t] for t in range(n_times)})
        return X, target

    def fitted_table(mode, seed):
        X, target = build_dataset(mode, np.random.RandomState(seed))
        model = SSTGAM(
            method="tree", n_value_bins=8, tree_depth=3,
            max_rounds=150, early_stopping_rounds=25, random_state=0,
        ).fit(X, target)
        _, table = model.value_time_table("marker")
        return table

    value_only_table = fitted_table("value", 201)
    std_across_time_per_bin = value_only_table.std(axis=1).mean()
    std_across_value_per_day = value_only_table.std(axis=0).mean()
    assert std_across_value_per_day > 2.0 * std_across_time_per_bin

    day_only_table = fitted_table("day", 202)
    std_across_time_per_bin = day_only_table.std(axis=1).mean()
    std_across_value_per_day = day_only_table.std(axis=0).mean()
    assert std_across_time_per_bin > 1.5 * std_across_value_per_day


def test_predicted_probabilities_are_calibrated_on_own_generative_model():
    generator = np.random.RandomState(33)
    n_train, n_test = 400, 400
    n_total = n_train + n_test

    value_a = generator.normal(size=n_total)
    value_b = generator.normal(size=n_total)
    static = generator.normal(size=n_total)
    logit = 0.3 + 0.8 * value_a + 0.8 * value_b + 0.6 * static
    probability_true = 1.0 / (1.0 + np.exp(-logit))
    target = generator.binomial(1, probability_true)

    X = pd.DataFrame({"marker_0": value_a, "marker_3": value_b, "context": static})
    X_train, X_test = X.iloc[:n_train].reset_index(drop=True), X.iloc[n_train:].reset_index(drop=True)
    y_train, y_test = target[:n_train], target[n_train:]

    model = SSTGAM(
        method="spline", max_rounds=100, early_stopping_rounds=20, random_state=6,
    ).fit(X_train, y_train)
    predicted = model.predict_proba(X_test)[:, 1]

    brier_score = float(np.mean((predicted - y_test) ** 2))
    baseline_brier = float(np.mean((y_train.mean() - y_test) ** 2))

    assert brier_score < baseline_brier
    assert brier_score < 0.24
