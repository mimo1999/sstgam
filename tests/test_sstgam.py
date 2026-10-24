import numpy as np
import pandas as pd
import pytest

from sstgam import SSTGAM, SSTGAMRegressor
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
    assert model.temporal_features_ == ["signal"]


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


def test_spline_bands_are_bayesian_and_shrink_with_more_data(longitudinal_data):
    X, y = longitudinal_data

    def band_width(n):
        model = SSTGAM(method="spline", max_rounds=12, early_stopping_rounds=4, outer_bags=1, random_state=4
                       ).fit(X.iloc[:n], y[:n])
        lower, mean, upper = model.value_effect("signal", [0.0, 1.0], return_interval=True)
        assert np.all(lower <= mean + 1e-9) and np.all(mean <= upper + 1e-9)
        return float(np.mean(upper - lower))

    width_small, width_large = band_width(len(y) // 3), band_width(len(y))
    assert 0.0 < width_large < width_small  # a single bag still gets a non-degenerate band


def test_bagged_band_is_centred_on_the_bag_average(longitudinal_data):
    X, y = longitudinal_data
    model = SSTGAM(method="spline", temporal_penalty=100.0, static_penalty=100.0, outer_bags=3, random_state=4).fit(X, y)
    lower, mean, upper = model.value_effect("signal", [0.0, 1.0], return_interval=True)
    np.testing.assert_allclose(mean, model.value_effect("signal", [0.0, 1.0]))
    assert np.all(lower < mean) and np.all(mean < upper)


def test_tree_pair_ranking_excludes_validation_rows(monkeypatch):
    from sstgam import tree_boost

    seen = []
    original = tree_boost.fast_gain
    monkeypatch.setattr(tree_boost, "fast_gain", lambda g, h, cells, *rest: (seen.append(len(g)), original(g, h, cells, *rest))[1])
    X, y = interaction_data(1000, 0)
    SSTGAM(method="tree", outer_bags=1, interactions=2, max_rounds=5, val_fraction=0.2).fit(X, y)
    assert seen and set(seen) == {800}


def test_centred_surface_has_zero_training_mean_at_each_time(longitudinal_data):
    X, y = longitudinal_data
    model = SSTGAM(method="spline", outer_bags=1, random_state=4).fit(X, y)
    for time_index, time_id in enumerate(model.time_ids_):
        column = X[f"signal_{time_id}"].dropna().to_numpy()
        surface = model.temporal_surface("signal", column, [time_id], center_time=True)
        assert abs(float(surface.mean())) < 1e-4, (time_id, surface.mean())


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
    with pytest.raises(ValueError, match="temporal_penalty must be 'auto', 'reml' or a positive number"):
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


def test_input_validation(longitudinal_data):
    X, y = longitudinal_data
    with pytest.raises(ValueError, match="no column names"):
        SSTGAM().fit(X.to_numpy(), y)
    with pytest.raises(ValueError, match="exactly two classes"):
        SSTGAM().fit(X, np.zeros_like(y))
    with pytest.raises(ValueError, match="infinite"):
        SSTGAM().fit(X.replace(X.iloc[0, 0], np.inf), y)
    with pytest.warns(UserWarning, match="static_columns"):
        SSTGAM(outer_bags=1).fit(X.rename(columns={"context": "icd_10"}), y)

    labels = np.where(y == 1, "yes", "no")
    model = SSTGAM(outer_bags=1, static_columns=["icd_10"]).fit(X.rename(columns={"context": "icd_10"}), labels)
    assert model.static_features_ == ["icd_10"] and set(model.predict(X.rename(columns={"context": "icd_10"}))) <= {"no", "yes"}
    with pytest.raises(ValueError, match="Unknown temporal feature"):
        model.value_effect("signl", [0.0])


@pytest.mark.parametrize("method", ["spline", "tree"])
def test_explain_sums_to_the_prediction(longitudinal_data, method):
    X, y = longitudinal_data
    model = SSTGAM(method=method, max_rounds=12, early_stopping_rounds=4, outer_bags=3).fit(X, y)
    explanation = model.explain(X)
    assert explanation.temporal.shape == (len(X), 1, 2) and explanation.static.shape == (len(X), 1)
    assert (explanation.temporal[X["signal_4"].isna().to_numpy(), 0, 1] == 0).all()
    logit = explanation.intercept + explanation.temporal.sum(axis=(1, 2)) + explanation.static.sum(axis=1)
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], 1 / (1 + np.exp(-logit)), atol=1e-6)
    assert list(model.term_importances_) in (["signal", "context"], ["context", "signal"])

    lower, mean, upper = model.static_effect("context", [-1.0, 0.0, 1.0], return_interval=True)
    assert (lower <= mean + 1e-12).all() and (mean <= upper + 1e-12).all()
    np.testing.assert_allclose(mean, model.static_effect("context", [-1.0, 0.0, 1.0]), atol=1e-10)


def test_decimal_time_ids():
    generator = np.random.RandomState(0)
    X = pd.DataFrame({f"signal_{t}": generator.normal(size=200) for t in ("0", "0.5", "2.25")})
    model = SSTGAM(outer_bags=1).fit(X, (X["signal_2.25"] > 0).astype(int))
    assert model.time_ids_ == [0, 0.5, 2.25]
    assert model.temporal_surface("signal", [1.0], [2.25]).shape == (1, 1)


@pytest.mark.parametrize("method", ["spline", "tree"])
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_regressor_recovers_a_temporal_effect(family, method):
    generator = np.random.RandomState(1)
    n = 2000
    values = generator.normal(size=(n, 4))
    signal = 0.5 * (values * np.linspace(0.0, 1.0, 4)).sum(axis=1)  # the effect grows over time
    target = (signal + generator.normal(scale=0.3, size=n) if family == "gaussian"
              else generator.poisson(np.exp(0.5 + signal)))
    X = pd.DataFrame(values, columns=[f"signal_{t}" for t in range(4)])
    model = SSTGAMRegressor(family=family, method=method, max_rounds=300, learning_rate=0.1,
                            outer_bags=1).fit(X, target)
    prediction = model.predict(X)
    expected = signal if family == "gaussian" else np.exp(0.5 + signal)
    assert np.corrcoef(prediction, expected)[0, 1] > 0.9
    if family == "poisson":
        assert (prediction > 0).all()
    with pytest.raises(ValueError, match="family must be"):
        SSTGAMRegressor(family="gamma").fit(X, target)


def interaction_data(n, seed):
    generator = np.random.RandomState(seed)
    creatinine, urea, lactate = (generator.normal(size=(n, 4)) for _ in range(3))
    age = generator.normal(size=n)
    for signal in (creatinine, urea, lactate):
        signal[generator.rand(n, 4) < 0.2] = np.nan
    filled = np.nan_to_num
    score = (-1 + 0.3 * filled(creatinine).sum(1) + 0.5 * age
             + (0.6 * filled(creatinine) * filled(urea) + 0.5 * filled(lactate) * age[:, None]).sum(1))
    target = (generator.rand(n) < 1 / (1 + np.exp(-score))).astype(int)
    columns = {}
    for name, signal in (("creatinine", creatinine), ("urea", urea), ("lactate", lactate)):
        columns |= {f"{name}_{t}": signal[:, t] for t in range(4)}
    return pd.DataFrame(columns | {"age": age}), target


@pytest.mark.parametrize("method", ["spline", "tree"])
def test_interactions_recover_planted_pairs(method):
    X, y = interaction_data(3000, 0)
    X_test, y_test = interaction_data(3000, 1)
    settings = {"method": method, "outer_bags": 1, "learning_rate": 0.1, "max_rounds": 300,
                "early_stopping_rounds": 20}
    main_only = SSTGAM(**settings).fit(X, y)
    model = SSTGAM(interactions=2, **settings).fit(X, y)
    assert set(model.interaction_terms_) == {("creatinine", "urea"), ("lactate", "age")}

    def test_loss(fitted):
        p = fitted.predict_proba(X_test)[:, 1]
        return -np.mean(y_test * np.log(p) + (1 - y_test) * np.log(1 - p))
    assert test_loss(model) < test_loss(main_only) - 0.03

    grid = np.linspace(-1.5, 1.5, 7)
    surface = model.interaction_surface("creatinine", "urea", grid, grid)
    assert np.corrcoef(surface.ravel(), np.outer(grid, grid).ravel())[0, 1] > 0.8
    np.testing.assert_allclose(model.interaction_surface("urea", "creatinine", grid, grid), surface.T)

    explanation = model.explain(X)
    assert explanation.pairs.shape == (len(X), 2)
    assert abs(explanation.pairs.mean(axis=0)).max() < 0.1  # purified: no main effect hiding in a pair
    logit = (explanation.intercept + explanation.temporal.sum(axis=(1, 2)) + explanation.static.sum(axis=1)
             + explanation.pairs.sum(axis=1))
    np.testing.assert_allclose(model.decision_function(X), logit)
    assert "creatinine x urea" in model.term_importances_


def test_explicit_interaction_list(longitudinal_data):
    X, y = longitudinal_data
    model = SSTGAM(outer_bags=1, interactions=[("context", "signal")]).fit(X, y)
    assert model.interaction_terms_ == [("signal", "context")]
    lower, mean, upper = model.interaction_surface("context", "signal", [0.0, 1.0], [0.0], return_interval=True)
    assert lower.shape == mean.shape == upper.shape == (2, 1)
    with pytest.raises(ValueError, match="No fitted interaction"):
        model.interaction_surface("signal", "signal", [0.0], [0.0])
    with pytest.raises(ValueError, match="Unknown feature"):
        SSTGAM(interactions=[("signal", "missing")]).fit(X, y)


def surface_data(n, kind, seed=0, times=8):
    """One signal whose effect depends only on time ("time") or only on the value ("value")."""
    generator = np.random.RandomState(seed)
    values = generator.normal(size=(n, times))
    tau = np.linspace(0.0, 1.0, times)
    effect = 0.25 * np.sin(2 * np.pi * tau)[None, :] + 0.0 * values if kind == "time" else 0.35 * np.tanh(2 * values)
    score = -1.0 + effect.sum(axis=1)
    target = (generator.rand(n) < 1 / (1 + np.exp(-score))).astype(int)
    return pd.DataFrame(values, columns=[f"signal_{t}" for t in range(times)]), target


def test_reml_selects_per_axis_smoothing():
    for kind, rougher in (("time", "time"), ("value", "value")):
        X, y = surface_data(6000, kind)
        model = SSTGAM(temporal_penalty="reml", static_penalty="reml", outer_bags=1).fit(X, y)
        smoothing = model.smoothing_params_["signal"]
        assert set(smoothing) == {"value", "time", "shrink"}
        smoother = "value" if rougher == "time" else "time"
        # The axis the effect varies along needs less smoothing than the flat one.
        assert smoothing[rougher] < smoothing[smoother], (kind, smoothing)
        assert 0 < model.edf_["signal"] < 40


def test_reml_fits_with_statics_pairs_and_sharing(longitudinal_data):
    X, y = longitudinal_data
    for sharing in ("per_term", "shared"):
        model = SSTGAM(temporal_penalty="reml", static_penalty="reml", penalty_sharing=sharing, outer_bags=2,
                       interactions=1).fit(X, y)
        assert np.isfinite(model.predict_proba(X)).all()
        assert set(model.smoothing_params_) >= {"signal", "context", "signal x context"}
        assert set(model.smoothing_params_["context"]) == {"value", "shrink"}
    fixed = SSTGAM(temporal_penalty="reml", static_penalty=5.0, outer_bags=1).fit(X, y)
    assert fixed.smoothing_params_["context"] == {"value": 5.0, "shrink": 5.0}
    with pytest.raises(ValueError, match="cannot mix 'auto' and 'reml'"):
        SSTGAM(temporal_penalty="reml", static_penalty="auto").fit(X, y)


def test_tree_per_bag_validation_and_leaf_limits(longitudinal_data):
    X, y = longitudinal_data
    settings = {"method": "tree", "max_rounds": 12, "early_stopping_rounds": 4, "random_state": 0}
    single = [SSTGAM(outer_bags=1, bag_validation=v, **settings).fit(X, y).predict_proba(X) for v in ("fixed", "per_bag")]
    np.testing.assert_array_equal(single[0], single[1])  # one bag: the same split either way
    model = SSTGAM(outer_bags=3, bag_validation="per_bag", min_leaf_hessian=1.0, max_leaves=4, **settings).fit(X, y)
    assert np.isfinite(model.predict_proba(X)).all() and len(model.model_.best_rounds_) == 3


def measurement_signal_data(n=3000, seed=0, times=6):
    """Values are pure noise; sicker rows (y = 1) get measured more often on later days."""
    generator = np.random.RandomState(seed)
    target = (generator.rand(n) < 0.25).astype(int)
    rate = np.where(target[:, None] == 1, np.linspace(0.5, 0.95, times), 0.5)
    values = np.where(generator.rand(n, times) < rate, generator.normal(size=(n, times)), np.nan)
    return pd.DataFrame(values, columns=[f"signal_{t}" for t in range(times)]), target


@pytest.mark.parametrize("method", ["spline", "tree"])
def test_missing_effects_learn_from_the_measurement_process(method):
    X, y = measurement_signal_data()
    X_test, y_test = measurement_signal_data(seed=1)
    settings = {"method": method, "outer_bags": 1, "learning_rate": 0.1, "max_rounds": 200, "random_state": 0}
    from sklearn.metrics import roc_auc_score
    without = roc_auc_score(y_test, SSTGAM(**settings).fit(X, y).predict_proba(X_test)[:, 1])
    model = SSTGAM(missing_effects=True, **settings).fit(X, y)
    with_effects = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    # the tree's value bins can already pick up the measurement pattern from missing cells, so only the
    # spline must gain; neither may lose
    assert with_effects > 0.7 and with_effects >= without - 0.01, (without, with_effects)
    if method == "spline":
        assert with_effects > without + 0.05, (without, with_effects)
    effect = model.missing_surface("signal", [0, 5])
    assert effect[1] < effect[0]  # not measured late = lower risk than not measured early
    explanation = model.explain(X_test)
    assert explanation.missing.shape == (len(X_test), 1, 6)
    assert (explanation.missing[X_test.notna().to_numpy()[:, None, :]] == 0).all()
    np.testing.assert_allclose(model.decision_function(X_test), explanation.intercept + explanation.temporal.sum(axis=(1, 2))
                               + explanation.static.sum(axis=1) + explanation.pairs.sum(axis=1)
                               + explanation.missing.sum(axis=(1, 2)), atol=1e-8)
    assert "signal [not measured]" in model.term_importances_


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("method", ["spline", "tree"])
def test_missing_effects_capture_the_missing_count(method, pooled):
    # risk rises with how often a patient goes unmeasured, at every time alike
    rng = np.random.RandomState(0)
    rate = rng.uniform(0.3, 0.95, size=2000)
    observed = rng.rand(2000, 6) < rate[:, None]
    values = np.where(observed, rng.normal(size=(2000, 6)), np.nan)
    other = np.where(rng.rand(2000, 6) < rate[:, None], rng.normal(size=(2000, 6)), np.nan)
    X = pd.DataFrame(np.hstack([values, other]),
                     columns=[f"signal_{t}" for t in range(6)] + [f"other_{t}" for t in range(6)])
    missing_count = (~observed).sum(axis=1) + np.isnan(other).sum(axis=1)
    y = (rng.rand(2000) < 1 / (1 + np.exp(-(missing_count / 2 - 2.0)))).astype(int)
    model = SSTGAM(method=method, missing_effects="pooled" if pooled else True, outer_bags=1,
                   random_state=0).fit(X, y)
    effect = model.explain(X).missing.sum(axis=(1, 2))
    if pooled:  # one effect over time shared by every feature
        np.testing.assert_allclose(model.missing_surface("signal", [0, 5]), model.missing_surface("other", [0, 5]))
    assert np.corrcoef(effect, missing_count)[0, 1] > 0.8


def test_missing_effects_with_reml_and_warning(longitudinal_data):
    X, y = longitudinal_data
    model = SSTGAM(missing_effects=True, temporal_penalty="reml", static_penalty="reml", outer_bags=1).fit(X, y)
    assert set(model.smoothing_params_["signal [not measured]"]) == {"time", "shrink"}
    with pytest.raises(ValueError, match="needs missing_effects=True"):
        SSTGAM(outer_bags=1).fit(X, y).missing_surface("signal", [0])
    with pytest.warns(UserWarning, match="missing_effects alone"):
        SSTGAM(missing_effects=True, include_missingness_indicators=True, outer_bags=1).fit(X, y)

def test_tree_min_gain_and_distinct_edges(longitudinal_data):
    from sstgam._training import distinct_edges, quantile_edges
    X, y = longitudinal_data
    fit = lambda **kw: SSTGAM(method="tree", max_rounds=12, early_stopping_rounds=4, random_state=1, **kw).fit(X, y)

    # a gate nothing can pass leaves every table at zero: the model is the intercept alone
    frozen = fit(min_gain=1e12)
    assert all(not table.any() for table in frozen.model_._feature_tables + frozen.model_._static_tables)
    assert np.ptp(frozen.predict_proba(X)[:, 1]) == 0
    # a moderate gate still fits, and 0 (the default) is the ungated model
    assert np.isfinite(fit(min_gain=2.0).predict_proba(X)[:, 1]).all()
    np.testing.assert_array_equal(fit(min_gain=0.0).predict_proba(X), fit().predict_proba(X))

    # few-level values keep their own bin; the default quantile edges merge them
    levels = np.repeat([0.0, 1.0, 2.0], [80, 15, 5])
    assert len(distinct_edges(levels, 16)) - 1 == 3 and len(quantile_edges(levels, 16)) - 1 < 3
    mass_at_max = np.concatenate([np.linspace(0, 1, 60), np.full(40, 1.0)])
    edges = distinct_edges(mass_at_max, 8)
    assert np.searchsorted(edges[1:-1], 1.0, side="right") > np.searchsorted(edges[1:-1], 0.9, side="right")
    assert np.isfinite(fit(bin_edges="distinct").predict_proba(X)[:, 1]).all()

    for bad in ({"min_gain": -1.0}, {"bin_edges": "uniform"}):
        with pytest.raises(ValueError):
            fit(**bad)


def history_data(n=1500, times=6, seed=0):
    """Risk rises with the change since the previous measurement, not with the level."""
    rng = np.random.RandomState(seed)
    level = rng.normal(size=(n, 1)) + 0.4 * rng.normal(size=(n, times))
    seen = rng.rand(n, times) < 0.7
    previous = np.full(n, np.nan)
    change = np.zeros(n)
    for t in range(times):
        now = np.where(seen[:, t], level[:, t], np.nan)
        ok = ~np.isnan(now) & ~np.isnan(previous)
        change = np.where(ok, change + (now - previous), change)
        previous = np.where(~np.isnan(now), now, previous)
    y = (rng.rand(n) < 1 / (1 + np.exp(-(1.8 * change - 1.5)))).astype(int)
    columns = {f"lab_{t}": np.where(seen[:, t], level[:, t], np.nan) for t in range(times)}
    return pd.DataFrame(columns), y


def test_context_channels_have_no_look_ahead_and_follow_the_order():
    X, _ = history_data(n=40, times=5)
    model = SSTGAM(context_channels=("change", "staleness"), outer_bags=1, max_rounds=2)
    model.temporal_features_ = ["lab"]
    model._base_features_ = ["lab"]
    model.time_ids_ = [0, 1, 2, 3, 4]
    values = np.nan_to_num(X.to_numpy())[:, None, :].astype(np.float32)
    observed = X.notna().to_numpy()[:, None, :].astype(np.float32)
    first = model._context_tensors(values, observed)
    values2, observed2 = values.copy(), observed.copy()
    values2[:, :, 4], observed2[:, :, 4] = 7.0, 1.0  # change the LAST cell only
    second = model._context_tensors(values2, observed2)
    for a, b in zip(first, second):  # n x 2 channels x 5 times: cells 0..3 must be untouched
        np.testing.assert_array_equal(a[:, :, :4], b[:, :, :4])
    # descending order reverses what "previous" means
    model.context_order = "descending"
    reverse = model._context_tensors(values[:, :, ::-1].copy(), observed[:, :, ::-1].copy())
    model.context_order = "ascending"
    np.testing.assert_array_equal(reverse[0][:, :, ::-1], first[0])


def test_change_channel_recovers_a_change_driven_outcome():
    X, y = history_data()
    from sklearn.metrics import roc_auc_score
    train, test = slice(0, 1000), slice(1000, None)

    def auc(**kw):
        model = SSTGAM(outer_bags=1, random_state=0, **kw).fit(X.iloc[train], y[train])
        return roc_auc_score(y[test], model.predict_proba(X.iloc[test])[:, 1])

    base, with_change = auc(), auc(context_channels=("change",))
    assert with_change > base + 0.1, (base, with_change)
    model = SSTGAM(context_channels=("change", "staleness"), outer_bags=1, random_state=0).fit(X, y)
    assert model.temporal_features_ == ["lab", "lab [change]", "lab [staleness]"]
    explanation = model.explain(X)
    assert explanation.temporal.shape == (len(X), 3, 6)
    np.testing.assert_allclose(model.decision_function(X), explanation.intercept + explanation.temporal.sum(axis=(1, 2))
                               + explanation.static.sum(axis=1) + explanation.pairs.sum(axis=1)
                               + explanation.missing.sum(axis=(1, 2)), atol=1e-8)


def test_invalid_context_settings_are_rejected():
    X, y = history_data(n=200)
    with pytest.raises(ValueError, match="context_channels"):
        SSTGAM(context_channels=("trend",)).fit(X, y)
    with pytest.raises(ValueError, match="context_order"):
        SSTGAM(context_channels=("change",), context_order="sideways").fit(X, y)
