"""Synthetic longitudinal data and model fits shared by the figure scripts.

One lab measured over 15 days (~25% of days missing) with a known value-time
effect, so plots can compare what each model recovers against the truth.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import sstgam.boosting as spline_backend  # noqa: F401  (figure scripts patch its iteration cap)
from sstgam import SSTGAM

OUT = Path(__file__).resolve().parent
N_DAYS = 15
DAYS = np.arange(N_DAYS)
VALUE_GRID = np.linspace(-2.5, 2.5, 101)

# Reference palette (dataviz skill, light mode).
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SPLINE, TREE, EBM = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 -- reserved for the models
SMOOTH, WIGGLY = "#e87ba4", "#4a3aa7"  # slots 5, 7 for non-model series (validated with slot 1)
DIVERGING_STOPS = ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#9c2121"]


def true_effect(value, day):
    """Ground truth: monotone early on, U-shaped by the last days."""
    late = np.asarray(day) / (N_DAYS - 1)
    value = np.asarray(value)
    return 0.5 * np.tanh(1.5 * value) * (1.0 - late) + 0.35 * (value ** 2 - 1.0) * late


def simulate(n_patients, seed):
    rng = np.random.RandomState(seed)
    level = rng.normal(0.0, 0.8, size=(n_patients, 1))
    trend = rng.normal(0.0, 0.8, size=(n_patients, 1))
    values = level + trend * DAYS / (N_DAYS - 1) + rng.normal(0.0, 0.5, size=(n_patients, N_DAYS))
    values[rng.rand(n_patients, N_DAYS) < 0.25] = np.nan
    observed = np.isfinite(values)
    score = -1.5 + np.where(observed, true_effect(np.nan_to_num(values), DAYS), 0.0).sum(axis=1)
    target = (rng.rand(n_patients) < expit(score)).astype(int)
    X = {f"lab_{day}": values[:, day] for day in DAYS}
    return X, values, target


def frame(columns):
    return np.column_stack(list(columns.values())), list(columns)


X_train, train_values, y_train = simulate(4000, seed=1)
X_test, test_values, y_test = simulate(4000, seed=2)
Xtr, names = frame(X_train)
Xte, _ = frame(X_test)
train_cells = train_values[np.isfinite(train_values)]
train_cell_days = np.broadcast_to(DAYS, train_values.shape)[np.isfinite(train_values)]
truth_offset = true_effect(train_cells, train_cell_days).mean()  # centre truth like the spline


def tree_lookup(model, values, days):
    edges, table = model.value_time_table("lab")
    bins = np.searchsorted(edges[1:-1], values, side="right")
    return table[bins, days]


def tree_surface(model):
    """Tree table on the value grid, centred over observed training cells like the others."""
    grid = np.array([[tree_lookup(model, v, d) for d in DAYS] for v in VALUE_GRID])
    return grid - tree_lookup(model, train_cells, train_cell_days).mean()


def spline_surface(model):
    return model.temporal_surface("lab", VALUE_GRID, DAYS)


def truth_surface():
    return true_effect(VALUE_GRID[:, None], DAYS[None, :]) - truth_offset


def test_loss(model):
    return log_loss(y_test, model.predict_proba(Xte)[:, 1])


def fit_spline(**kwargs):
    return SSTGAM(method="spline", n_value_bases=8, n_time_bases=5, outer_bags=1,
                  random_state=0, **kwargs).fit(Xtr, y_train, feature_names=names)


def fit_ebm():
    """EBM on the same 15 columns: one shape function per day, nothing shared across days.

    ``interactions=0`` keeps it purely additive, so its per-day curves stack
    into a value x day surface comparable with SSTGAM's.
    """
    from interpret.glassbox import ExplainableBoostingClassifier
    return ExplainableBoostingClassifier(interactions=0, random_state=0).fit(Xtr, y_train)


def ebm_surface(model):
    """EBM's 15 per-day shape functions on the value grid, one column per day (not yet centred)."""
    return model.eval_terms(np.tile(VALUE_GRID[:, None], (1, N_DAYS)))


def ebm_surface_centred(model):
    """EBM surface centred over all observed training cells, like the SSTGAM surfaces."""
    surface = ebm_surface(model)
    cells = [np.interp(train_values[:, day][np.isfinite(train_values[:, day])], VALUE_GRID, surface[:, day])
             for day in DAYS]
    return surface - np.concatenate(cells).mean()


def ebm_cells(model, row_values):
    """EBM's contribution from each day of one patient, including missing days (EBM learns a score for them)."""
    return model.eval_terms(np.asarray(row_values, dtype=float)[None, :])[0]


def fit_tree(rounds):
    return SSTGAM(method="tree", n_value_bins=16, tree_depth=2, learning_rate=0.05, max_rounds=rounds,
                  early_stopping_rounds=rounds, outer_bags=1, random_state=0).fit(Xtr, y_train, feature_names=names)
