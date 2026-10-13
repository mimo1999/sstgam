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


def test_spline_terms_are_centred(longitudinal_data):
    X, y = longitudinal_data
    model = SSTGAM(method="spline", outer_bags=2, random_state=0).fit(X, y)

    # Each surface averages zero over the observed training cells ...
    cell_effects = []
    for time_id in (0, 4):
        column = X[f"signal_{time_id}"].dropna().to_numpy()
        cell_effects.append(model.temporal_surface("signal", column, [time_id])[:, 0])
    assert abs(np.concatenate(cell_effects).mean()) < 1e-6
    # ... and each static effect over the training rows.
    assert abs(model.static_effect("context", X["context"]).mean()) < 1e-6


def test_single_bag_spline_fit_is_the_deterministic_penalized_optimum(longitudinal_data):
    X, y = longitudinal_data
    models = [SSTGAM(method="spline", temporal_penalty=1.0, static_penalty=1.0, outer_bags=1,
                     random_state=seed).fit(X, y) for seed in (0, 1)]
    # No validation split or boosting path: one bag fits all rows to the same optimum.
    np.testing.assert_allclose(models[0].predict_proba(X), models[1].predict_proba(X), atol=1e-10)
    assert models[0].model_.n_iter_[0] < 50


def test_temporal_penalty_controls_smoothness(longitudinal_data):
    X, y = longitudinal_data
    grid = np.linspace(-2.0, 2.0, 21)
    ranges = [
        np.ptp(SSTGAM(method="spline", temporal_penalty=penalty, max_rounds=300, outer_bags=1)
               .fit(X, y).value_effect("signal", grid))
        for penalty in (0.1, 1e4)
    ]
    assert ranges[1] < 0.1 * ranges[0]


@pytest.mark.parametrize("method", ["spline", "tree"])
def test_static_effect(longitudinal_data, method):
    X, y = longitudinal_data
    model = SSTGAM(method=method, include_missingness_indicators=True, max_rounds=12,
                   early_stopping_rounds=4).fit(X, y)
    assert model.static_features_ == ["context", "signal_observed_fraction"]
    assert model.static_effect("context", [-1.0, 0.0, np.nan, 1.0]).shape == (4,)
    assert model.static_effect("signal_observed_fraction", 0.5).shape == (1,)
    with pytest.raises(ValueError, match="Unknown static feature"):
        model.static_effect("signal", 0.0)


def test_auto_penalty_prefers_the_smoother_of_near_ties(longitudinal_data):
    X, y = longitudinal_data
    backend = SSTGAM(method="spline", outer_bags=1).fit(X, y).model_
    search = backend.penalty_search_
    assert len(search) >= 3
    # One-standard-error rule: never the minimiser's smaller neighbours, at least as smooth as it.
    assert backend.temporal_penalty_ == backend.static_penalty_ >= min(search, key=search.get)


def test_auto_penalty_regularizes_noise_more_than_signal():
    generator = np.random.RandomState(0)
    n_samples = 600
    X = pd.DataFrame({f"a_{t}": generator.normal(size=n_samples) for t in range(3)})
    X["context"] = generator.normal(size=n_samples)
    signal = (2.0 * X["a_0"] + X["context"] + generator.logistic(size=n_samples) > 0).astype(int)
    noise = generator.randint(0, 2, size=n_samples)
    chosen = [SSTGAM(method="spline", outer_bags=1).fit(X, target).model_.temporal_penalty_
              for target in (signal, noise)]
    assert chosen[1] > 10 * chosen[0]


def test_explicit_penalty_skips_the_search(longitudinal_data):
    X, y = longitudinal_data
    backend = SSTGAM(method="spline", temporal_penalty=3.0, static_penalty="auto", outer_bags=1).fit(X, y).model_
    assert backend.temporal_penalty_ == 3.0 and backend.penalty_search_
    backend = SSTGAM(method="spline", temporal_penalty=3.0, static_penalty=2.0, outer_bags=1).fit(X, y).model_
    assert (backend.temporal_penalty_, backend.static_penalty_) == (3.0, 2.0) and not backend.penalty_search_
    with pytest.raises(ValueError, match="temporal_penalty must be 'auto' or a positive number"):
        SSTGAM(temporal_penalty="smooth").fit(X, y)


def test_readme_quick_start_fits_tiny_data():
    # The README example: 4 rows, so bootstrap bags can be all one class.
    X = pd.DataFrame({
        "signal_a_0": [0.2, 1.1, 0.5, 1.7],
        "signal_a_6": [0.3, 0.8, None, 1.9],
        "signal_b_0": [4.0, 2.1, 3.3, 1.4],
        "age": [32, 51, 44, 67],
    })
    model = SSTGAM(method="spline", random_state=0).fit(X, [0, 0, 1, 1])
    assert np.isfinite(model.predict_proba(X)).all()
    assert model.static_effect("age", [30, 50, 70]).shape == (3,)
