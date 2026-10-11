# SSTGAM

SSTGAM is a small, standalone Python package for explainable binary
classification with multiple time-varying features. It fits one shared
value-time effect per feature, so the effect of a value can change smoothly or
discretely over time while remaining directly inspectable.

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
`<feature_name>_<time_id>`, where `time_id` is a non-negative integer. All
other columns are treated as static covariates.

When fitting a NumPy array instead of a DataFrame, pass the matching column
names explicitly through `feature_names`. Inside a scikit-learn `Pipeline` or
`GridSearchCV`, column names are only preserved if every preceding step
outputs DataFrames, e.g. `preprocessor.set_output(transform="pandas")`;
otherwise `fit` sees no temporal columns and raises an error.

For example:

| Column | Meaning |
| --- | --- |
| `signal_a_0` | `signal_a` at time 0 |
| `signal_a_6` | `signal_a` at time 6 |
| `signal_b_0` | `signal_b` at time 0 |
| `age` | static covariate |

Missing temporal values may be represented by `NaN`. They make no temporal
contribution. Input DataFrames are aligned to the fitted column names at
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
y_train = [0, 0, 1, 1]

model = SSTGAM(method="spline", random_state=0)
model.fit(X_train, y_train)
probability = model.predict_proba(X_train)[:, 1]
```

## Model

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

Both methods are fitted by cyclic (coordinate-descent) gradient boosting with
early stopping on validation log loss. The tree method's leaf values are true
per-round Newton steps (fresh gradient/Hessian each round); the spline
method's coefficient updates use a fixed quasi-Newton preconditioner from a
worst-case Hessian bound computed once per bag (see [ARCHITECTURE.md](https://github.com/mimo1999/sstgam/blob/main/ARCHITECTURE.md)). Both
methods are bagged: `outer_bags` independent fits are run on
bootstrap resamples of the training rows (the validation split is held fixed
across bags) and their fitted parameters are averaged. Averaging leaves the
model in the same form — a mean of coefficient vectors over a shared basis is
another coefficient vector, and a mean of value-time tables is another table —
so bagging reduces variance without changing what `f_k` is.

## Methods

- `method="spline"` (default) uses smooth tensor-product surfaces with
  roughness penalties. Use this when you want smooth effect plots and support
  for irregular numeric time coordinates. Set `basis="rbf"` (default) for
  fixed-bandwidth Gaussian bumps or `basis="bspline"` for a clamped B-spline
  basis of the same size; both use the same roughness penalty.
- `method="tree"` uses quantile value bins and shallow tree updates to learn a
  discrete value-time table. Use `value_time_table(feature)` to inspect it.

Each tree update chooses its split order adaptively rather than letting the
tree pick freely. Depth-1 candidate splits are fitted on the value axis and the
time axis separately and scored by weighted-SSE gain minus `2 * log(m)`, where
`m` is the number of distinct candidate thresholds on that axis; the
higher-scoring axis splits first and the other refines within each resulting
group. This is a candidate-count penalty on greedy SSE gain, not a formal
Bonferroni or BIC correction (there is no p-value or likelihood term here) —
just the same intuition: an axis with more candidate thresholds gets more
chances for a greedy split to look good by chance, so the penalty scales with
the log of the candidate count. Without it the value axis wins on candidate
count alone, since it usually has more quantile bins than there are
observation times.

For spline models, inspect fitted effects with:

```python
value_effect = model.value_effect("signal_a", values=[0.0, 0.5, 1.0])
surface = model.temporal_surface(
    "signal_a", values=[0.0, 0.5, 1.0], time_coordinates=[0, 3, 6]
)
```

## Key parameters

- `n_value_bases`, `n_time_bases`, `temporal_penalty`: spline surface
  complexity and smoothness.
- `n_value_bins`, `tree_depth`: tree table resolution and update complexity.
- `learning_rate`, `max_rounds`, `early_stopping_rounds`: optimization
  controls.
- `outer_bags` (default 4): bagged fits to average. Set to 1 to disable
  bagging and fit once on the full training split; cost scales linearly.
- `value_transform`: use `"quantile"` for robust rank-based scaling or
  `"zscore"` for robust center/scale normalization.
- `include_missingness_indicators` (default `False`): adds one extra static
  covariate per temporal feature (its per-sample observed-fraction), so the
  model can learn from informative missingness instead of assuming
  unobserved cells carry no information. See [ARCHITECTURE.md](https://github.com/mimo1999/sstgam/blob/main/ARCHITECTURE.md) for the
  assumption this opts out of.
