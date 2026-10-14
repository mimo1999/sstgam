# SSTGAM

SSTGAM is a small, standalone Python package for explainable models of
outcomes driven by multiple time-varying features: binary classification
(`SSTGAM`), and continuous or count outcomes (`SSTGAMRegressor`). It fits one shared
value-time effect per feature, so the effect of a value can change smoothly or
discretely over time while remaining directly inspectable.

## Why not EBM or pyGAM?

Give EBM or pyGAM a lab measured on 15 days and it sees 15 unrelated columns:
each day gets its own shape function, learned only from that day's
measurements. Day 7 learns nothing from days 6 and 8, so the curves come out
noisy and jump from day to day, and nothing ties them into one story about the
lab. SSTGAM fits a single value-by-time surface per feature from every
observed day at once. Neighbouring days share strength, the effect can still
drift over time, and the whole feature is explained by one picture.

<p align="center">
  <img src="https://raw.githubusercontent.com/mimo1999/sstgam/main/docs/figures/fig1_what_each_model_learns.png"
       alt="Synthetic lab over 15 days: the true value-by-day effect, SSTGAM spline's smooth surface, SSTGAM tree's value-bin by day table, and EBM's 15 noisy per-day functions, with day 1, 8 and 15 cross-sections"
       width="900">
</p>

On synthetic data with a known effect (monotone early, U-shaped late), the
spline surface tracks the truth; EBM's per-day functions are noisier,
especially on late days and at extreme values where each day has few
observations. Scripts for all figures are in
[docs/figures](https://github.com/mimo1999/sstgam/tree/main/docs/figures).

See [ARCHITECTURE.md](https://github.com/mimo1999/sstgam/blob/main/ARCHITECTURE.md) for the model design and its
stated assumptions/limits, and [BENCHMARKS.md](https://github.com/mimo1999/sstgam/blob/main/BENCHMARKS.md) for
comparative results against other model families.

## Installation

```bash
pip install sstgam
```

The package depends only on NumPy, SciPy, and scikit-learn. The examples below
use pandas, which is optional (`pip install pandas`).

## Data format

`SSTGAM` accepts a wide matrix. Name each temporal column
`<feature_name>_<time_id>`, where `time_id` is a non-negative number such as
`6` or `0.5`. Time ids are placed at their actual distances, so uneven
spacing (days 0, 3, 7, 14) is fine; every row shares the same set of time
columns (see [Irregular timestamps](#irregular-timestamps) when each patient
is measured at their own times). All other columns are treated as static
covariates.

When fitting a NumPy array instead of a DataFrame, pass the matching column
names explicitly through `feature_names`; without names `fit` raises an error.
Inside a scikit-learn `Pipeline` or `GridSearchCV`, column names are only
preserved if every preceding step outputs DataFrames, e.g.
`preprocessor.set_output(transform="pandas")`.

A static column whose name happens to end in `_<number>` (e.g. `icd_10`)
would be read as a temporal feature; list it in `static_columns=["icd_10"]`.
`fit` warns when a lone column like this sits at a time no other feature
uses. Static covariates must be numeric: encode categories before fitting.

For example:

| Column | Meaning |
| --- | --- |
| `signal_a_0` | `signal_a` at time 0 |
| `signal_a_6` | `signal_a` at time 6 |
| `signal_b_0` | `signal_b` at time 0 |
| `age` | static covariate |

Missing values are `NaN` (infinite values raise an error). Missing temporal
values make no temporal contribution. Input DataFrames are aligned to the fitted column names at
prediction time; missing or unexpected columns raise an error.

## Quick start

```python
import pandas as pd
from sstgam import SSTGAM

X_train = pd.DataFrame({
    "signal_a_0": [0.2, 1.1, 0.5, 1.7],
    "signal_a_6": [0.3, 0.8, None, 1.9],
    "signal_b_0": [4.0, 2.1, 3.3, 1.4],
    "age": [32, 51, 44, 67],
})
y_train = [0, 0, 1, 1]  # any two labels, e.g. ["no", "no", "yes", "yes"]

model = SSTGAM(method="spline", random_state=0)
model.fit(X_train, y_train)
probability = model.predict_proba(X_train)[:, 1]
```

For a continuous or count outcome, use `SSTGAMRegressor` with the same data
format and parameters; `predict` returns the expected outcome:

```python
from sstgam import SSTGAMRegressor

regressor = SSTGAMRegressor(family="poisson")  # or "gaussian" (the default)
regressor.fit(X_train, [0, 2, 1, 5])
expected_count = regressor.predict(X_train)
```

## Irregular timestamps

When each patient is measured at their own times, keep the data long (one row
per measurement) and let `bin_timestamps` put it on a shared grid. It needs
pandas.

```python
from sstgam import bin_timestamps

observations = pd.DataFrame({
    "patient": [1, 1, 1, 2, 2],
    "hour":    [0.5, 3.0, 13.0, 2.0, 30.0],
    "lab":     ["lactate", "lactate", "lactate", "lactate", "urea"],
    "value":   [1.8, 2.4, 1.1, 3.0, 7.5],
})
wide = bin_timestamps(observations, id_col="patient", time_col="hour", feature_col="lab",
                      value_col="value", bin_width=12, n_bins=3)
# columns lactate_0, lactate_12, lactate_24, urea_0, urea_12, urea_24; one row per patient
```

- Time zero is each patient's first measurement (`origin="first"`). Use
  `origin="absolute"` for the raw clock, or pass a Series of per-patient start
  times such as admission.
- Measurements in the same bin are averaged (`agg="mean"`); any pandas
  aggregation works, e.g. `"last"` or `"max"`.
- Bins with no measurement are NaN, which the model treats as missing.
- `counts=True` adds `<feature>_n_<bin>` columns with the number of
  measurements per bin, for when measurement frequency carries signal.
- The bins depend only on the arguments, so binning test data with the same
  arguments gives the same columns. Join static covariates by patient id.

Datetime times work with a `pd.Timedelta` bin width; the columns are then
numbered by bin.

## Interactions (experimental)

Like EBM/GA2M, SSTGAM can add pairwise interaction terms on top of its main
effects. They are opt-in and still being validated: the pair ranking has known
limitations described in [ARCHITECTURE.md](https://github.com/mimo1999/sstgam/blob/main/ARCHITECTURE.md).

```python
model = SSTGAM(interactions=3).fit(X_train, y_train)  # the 3 strongest pairs
model.interaction_terms_  # e.g. [("signal_a", "signal_b"), ("signal_a", "age"), ...]
```

- `interactions=K` ranks every signal x signal, signal x static and static x
  static pair with FAST (Lou et al., KDD 2013) on what the main effects leave
  unexplained, and fits the top K. Pass a list such as
  `[("creatinine", "urea"), ("lactate", "age")]` to choose the pairs yourself.
- A signal x signal pair adds `h(value_a(t), value_b(t))` at every time both
  were measured; a signal x static pair adds `h(value_a(t), age)` at every
  time the signal was measured. The surface `h` is shared across time, like
  the main effects.
- Pairs are purified: they hold only what the two terms do together beyond
  their separate effects, so the main-effect plots keep their meaning.
- Inspect one with `interaction_surface`; `explain(X).pairs` and
  `term_importances_` include them.

```python
import matplotlib.pyplot as plt
import numpy as np

grid = np.linspace(-2, 2, 40)
surface = model.interaction_surface("signal_a", "signal_b", grid, grid)  # (40, 40), log-odds
plt.pcolormesh(grid, grid, surface.T, cmap="RdBu_r", shading="nearest")
plt.xlabel("signal_a"); plt.ylabel("signal_b"); plt.colorbar(label="log-odds")
```

Each pair adds parameters and fit time, so start with a few and keep the ones
`term_importances_` ranks high.

## Interpreting outputs

Every effect is on the model's link scale: **log-odds** of the second class
in `model.classes_` for `SSTGAM`, the outcome's own units for a gaussian
`SSTGAMRegressor`, and log-rate for a poisson one. Effects are centred:
spline effects average zero over the training data, so positive means
"raises the outcome relative to a typical observation".

| Method | Backend | Returns |
| --- | --- | --- |
| `temporal_surface(feature, values, time_coordinates)` | spline | array `(len(values), len(time_coordinates))`: the effect one observation at that value and time adds |
| `value_effect(feature, values)` | spline | array `(len(values),)`: the surface averaged over time |
| `value_time_table(feature)` | tree | `(value_bin_edges, table)`, table `(n_bins, len(model.time_ids_))`, edges in raw units |
| `static_effect(feature, values)` | both | array `(len(values),)` for a static covariate |
| `interaction_surface(a, b, values_a, values_b)` | both | array `(len(values_a), len(values_b))` for a pair in `model.interaction_terms_` |
| `explain(X)` | both | `Explanation(intercept, temporal, static, pairs)`: per-row terms, `temporal` `(n_rows, len(model.temporal_features_), len(model.time_ids_))`, `static` `(n_rows, len(model.static_features_))`, `pairs` `(n_rows, len(model.interaction_terms_))` |
| `decision_function(X)` | both | array `(n_rows,)`: the link-scale prediction, the sum of `explain(X)` |
| `term_importances_` | both | dict of each term's mean absolute contribution over the training rows, largest first |

`temporal_surface`, `value_effect`, `static_effect`, `interaction_surface` and `value_time_table` take
`return_interval=True` and then return `(lower, mean, upper)`: the 2.5 and
97.5 percentiles across the `outer_bags` bootstrap fits. With the default 4
bags this is a rough stability check, not a calibrated interval; raise
`outer_bags` (e.g. 20) when the bands matter. Time coordinates are the ids
from the column names; outside the range seen in training the surface is
extrapolated and unreliable.

A patient's score is the sum of what each observed cell adds, so a prediction
can be explained day by day. `explain(X)` returns exactly these per-day
terms; missing days are 0:

<p align="center">
  <img src="https://raw.githubusercontent.com/mimo1999/sstgam/main/docs/figures/fig4_scoring_one_patient.png"
       alt="One patient's lab values over 15 days and the log-odds each model adds per day; SSTGAM adds zero for missing days, EBM scores them too"
       width="700">
</p>

```python
import matplotlib.pyplot as plt
import numpy as np

values = np.linspace(X_train["signal_a_0"].min(), X_train["signal_a_0"].max(), 50)
times = model.time_ids_
surface = model.temporal_surface("signal_a", values, times)
plt.pcolormesh(times, values, surface, cmap="RdBu_r", shading="nearest")
plt.xlabel("time"); plt.ylabel("signal_a"); plt.colorbar(label="log-odds")

explanation = model.explain(X_train)
row = 0  # what each observed time of signal_a adds for the first row
dict(zip(model.time_ids_, explanation.temporal[row, model.temporal_features_.index("signal_a")]))
```

## Model

<p align="center">
  <img src="https://raw.githubusercontent.com/mimo1999/sstgam/main/docs/figures/sstgam_architecture.png"
       alt="SSTGAM architecture: each feature's measurements over 15 days pass through that feature's shared temporal shape function, and the per-feature contributions are summed into the output"
       width="480">
</p>

For sample `i`, temporal feature `k`, time index `d`, and static feature `s`,
the model is:

```text
logit(p_i) = intercept
           + sum_k sum_d observed[i, k, d] * f_k(value[i, k, d], time_d)
           + sum_s g_s(static[i, s])
```

`f_k` is one value-time surface per temporal feature. `g_s` is a
one-dimensional static effect. This is an additive time-varying model: it
does not directly model lags, slopes, or interactions between successive
measurements. Supply those as input features if they are needed.

Pairwise interactions between features are optional and experimental (see
[Interactions](#interactions-experimental)).

The spline method is a penalized GAM solved directly by P-IRLS, with the
smoothness penalty chosen on held-out data. The tree method is fitted by
cyclic Newton boosting with shallow trees and early stopping. Both average
`outer_bags` bootstrap fits, which leaves the model in the same form. See
[ARCHITECTURE.md](https://github.com/mimo1999/sstgam/blob/main/ARCHITECTURE.md)
for the fitting details.

## Methods

- `method="spline"` (default) uses smooth tensor-product surfaces with
  roughness penalties. Use this when you want smooth effect plots; unevenly
  spaced time ids (e.g. days 0, 3, 7, 14) are placed at their actual
  distances. Set `basis="rbf"` (default) for
  fixed-bandwidth Gaussian bumps or `basis="bspline"` for a clamped B-spline
  basis of the same size; both use the same roughness penalty.
- `method="tree"` uses quantile value bins and shallow tree updates to learn a
  discrete value-time table. Use `value_time_table(feature)` to inspect it.

With `include_missingness_indicators=True`, `static_effect` also accepts
`<feature>_observed_fraction`; `model.static_features_` lists every name it
accepts.

## Key parameters

- `n_value_bases`, `n_time_bases`, `temporal_penalty`: spline surface
  complexity and smoothness. `n_static_bases`, `static_penalty`: the same for
  static effects. The penalties default to `"auto"`: the right value differs
  by orders of magnitude between datasets (it tracks how many minority-class
  examples each coefficient has), so it is chosen by held-out loss -- one
  validation split on large data, 5-fold cross-validation when there are
  under ~500 held-out minority examples (rows, for `SSTGAMRegressor`). Pass
  numbers to fix them.
- `n_value_bins`, `tree_depth`: tree table resolution and update complexity.
- `learning_rate`, `max_rounds`, `early_stopping_rounds`: boosting controls
  for the tree method; the spline method ignores them. `val_fraction`: the
  tree's early-stopping split and the spline's `"auto"` penalty split.
- `outer_bags` (default 4): bagged fits to average. Set to 1 to disable
  bagging and fit once on the full training split; cost scales linearly.
- `value_transform`: use `"quantile"` for robust rank-based scaling or
  `"zscore"` for robust center/scale normalization. The spline method applies
  it to temporal values and static covariates alike.
- `include_missingness_indicators` (default `False`): adds one extra static
  covariate per temporal feature (its per-sample observed-fraction), so the
  model can learn from informative missingness instead of assuming
  unobserved cells carry no information. See [ARCHITECTURE.md](https://github.com/mimo1999/sstgam/blob/main/ARCHITECTURE.md) for the
  assumption this opts out of.
