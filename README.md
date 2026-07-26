# SSTGAM

SSTGAM is a small, standalone Python package for explainable binary
classification with multiple time-varying features. It fits one shared
value-time effect per feature, so the effect of a value can change smoothly or
discretely over time while remaining directly inspectable.

## Installation

```bash
pip install -e .
```

The package depends only on NumPy and scikit-learn.

## Data format

`SSTGAM` accepts a wide matrix. Name each temporal column
`<feature_name>_<time_id>`, where `time_id` is a non-negative integer. All
other columns are treated as static covariates.

When fitting a NumPy array instead of a DataFrame, pass the matching column
names explicitly through `feature_names`.

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

## Methods

- `method="spline"` (default) uses smooth Gaussian radial-basis surfaces with
  roughness penalties. Use this when you want smooth effect plots and support
  for irregular numeric time coordinates.
- `method="tree"` uses quantile value bins and shallow tree updates to learn a
  discrete value-time table. Use `value_time_table(feature)` to inspect it.

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
- `value_transform`: use `"quantile"` for robust rank-based scaling or
  `"zscore"` for robust center/scale normalization.
