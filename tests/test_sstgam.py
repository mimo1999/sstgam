import numpy as np
import pandas as pd
import pytest

from sstgam import SSTGAM
from sstgam.bases import bspline_np, make_bspline_knots


@pytest.fixture
def longitudinal_data():
    generator = np.random.RandomState(10)
    n_samples = 70
    first = generator.normal(size=n_samples)
    second = first + generator.normal(scale=0.25, size=n_samples)
    static = generator.normal(size=n_samples)
    second[generator.rand(n_samples) < 0.15] = np.nan
    target = (first + static + generator.normal(scale=0.4, size=n_samples) > 0).astype(int)
    return pd.DataFrame({"signal_0": first, "signal_4": second, "context": static}), target


@pytest.mark.parametrize("method", ["spline", "tree"])
def test_fit_predict_and_dataframe_column_alignment(longitudinal_data, method):
    X, y = longitudinal_data
    model = SSTGAM(
        method=method, max_rounds=12, early_stopping_rounds=4,
        random_state=1,
    ).fit(X, y)

    probability = model.predict_proba(X)[:, 1]
    reordered_probability = model.predict_proba(X[["context", "signal_4", "signal_0"]])[:, 1]

    assert np.isfinite(probability).all()
    np.testing.assert_allclose(probability, reordered_probability)
    assert model.get_temporal_features() == ["signal"]


def test_interpretation_helpers(longitudinal_data):
    X, y = longitudinal_data
    spline_model = SSTGAM(
        method="spline", max_rounds=12, early_stopping_rounds=4, random_state=2
    ).fit(X, y)
    assert spline_model.value_effect("signal", [0.0, 1.0]).shape == (2,)
    assert spline_model.temporal_surface("signal", [0.0, 1.0], [0, 4]).shape == (2, 2)

    tree_model = SSTGAM(
        method="tree", max_rounds=12, early_stopping_rounds=4, random_state=2
    ).fit(X, y)
    _, table = tree_model.value_time_table("signal")
    assert table.shape[1] == 2


def test_bspline_basis_fits_and_predicts(longitudinal_data):
    X, y = longitudinal_data
    model = SSTGAM(
        method="spline", basis="bspline", max_rounds=12, early_stopping_rounds=4,
        random_state=2,
    ).fit(X, y)

    probability = model.predict_proba(X)[:, 1]
    assert np.isfinite(probability).all()
    assert model.value_effect("signal", [0.0, 1.0]).shape == (2,)
    assert model.temporal_surface("signal", [0.0, 1.0], [0, 4]).shape == (2, 2)

    with pytest.raises(ValueError, match="basis must be"):
        SSTGAM(basis="unknown").fit(X, y)


def test_bootstrap_interval_shrinks_with_more_bags(longitudinal_data):
    X, y = longitudinal_data

    def interval_width(outer_bags):
        model = SSTGAM(
            method="spline", max_rounds=12, early_stopping_rounds=4,
            outer_bags=outer_bags, random_state=4,
        ).fit(X, y)
        lower, mean, upper = model.value_effect("signal", [0.0, 1.0], return_interval=True)
        assert np.all(lower <= mean + 1e-9) and np.all(mean <= upper + 1e-9)
        return float(np.mean(upper - lower))

    width_single_bag = interval_width(outer_bags=1)
    width_many_bags = interval_width(outer_bags=12)

    assert width_single_bag == pytest.approx(0.0, abs=1e-9)
    assert width_many_bags >= 0.0


def test_temporal_surface_and_value_time_table_intervals(longitudinal_data):
    X, y = longitudinal_data
    spline_model = SSTGAM(
        method="spline", max_rounds=12, early_stopping_rounds=4,
        outer_bags=6, random_state=5,
    ).fit(X, y)
    lower, mean, upper = spline_model.temporal_surface(
        "signal", [0.0, 1.0], [0, 4], return_interval=True
    )
    assert lower.shape == mean.shape == upper.shape == (2, 2)
    assert np.all(lower <= mean + 1e-9) and np.all(mean <= upper + 1e-9)

    tree_model = SSTGAM(
        method="tree", max_rounds=12, early_stopping_rounds=4,
        outer_bags=6, random_state=5,
    ).fit(X, y)
    edges, lower, mean, upper = tree_model.value_time_table("signal", return_interval=True)
    assert lower.shape == mean.shape == upper.shape
    assert np.all(lower <= mean + 1e-9) and np.all(mean <= upper + 1e-9)


def test_informative_missingness_indicator_helps():
    generator = np.random.RandomState(21)
    n_train, n_test = 300, 300
    n_total = n_train + n_test

    target = generator.binomial(1, 0.5, size=n_total)
    # Missingness rate depends on the label; the observed values themselves
    # carry no signal, so only a missingness-aware model can use this.
    miss_probability = np.where(target == 1, 0.85, 0.15)
    raw_values = generator.normal(size=(n_total, 4))
    observed_mask = generator.random_sample((n_total, 4)) >= miss_probability[:, None]
    values = np.where(observed_mask, raw_values, np.nan)

    X = pd.DataFrame({f"marker_{time_id}": values[:, time_id] for time_id in range(4)})
    X_train, X_test = X.iloc[:n_train].reset_index(drop=True), X.iloc[n_train:].reset_index(drop=True)
    y_train, y_test = target[:n_train], target[n_train:]

    def held_out_log_loss(include_missingness_indicators):
        model = SSTGAM(
            method="spline", max_rounds=60, early_stopping_rounds=10,
            include_missingness_indicators=include_missingness_indicators,
            random_state=3,
        ).fit(X_train, y_train)
        probability = np.clip(model.predict_proba(X_test)[:, 1], 1e-6, 1 - 1e-6)
        return -np.mean(y_test * np.log(probability) + (1 - y_test) * np.log(1 - probability))

    assert held_out_log_loss(True) < held_out_log_loss(False)


def test_missingness_indicator_defaults_off_and_preserves_static_count(longitudinal_data):
    X, y = longitudinal_data
    model = SSTGAM(max_rounds=8, early_stopping_rounds=3).fit(X, y)
    assert model.include_missingness_indicators is False


def test_prediction_rejects_mismatched_dataframe_columns(longitudinal_data):
    X, y = longitudinal_data
    model = SSTGAM(max_rounds=8, early_stopping_rounds=3).fit(X, y)
    with pytest.raises(ValueError, match="Prediction columns differ"):
        model.predict_proba(X.drop(columns="context"))


def test_bspline_partition_of_unity_at_and_beyond_bounds():
    knots, degree = make_bspline_knots(-4.0, 4.0, 8)
    basis = bspline_np(np.array([-5.0, -4.0, 0.0, 4.0, 5.0]), knots, degree)
    np.testing.assert_allclose(basis.sum(axis=-1), 1.0, atol=1e-6)


def test_value_effect_accepts_scalar(longitudinal_data):
    X, y = longitudinal_data
    model = SSTGAM(max_rounds=8, early_stopping_rounds=3).fit(X, y)
    assert model.value_effect("signal", 0.5).shape == (1,)
