# `SSTGAM.__init__`

**File:** `src/sstgam/classifier.py:47-87`
**Kind:** constructor

## Purpose

Validates and stores every user-facing hyperparameter for both backends — the
same `SSTGAM` object's constructor accepts params for both `method="spline"`
and `method="tree"`, only the relevant subset is used later by
[`_make_trainer`](make_trainer.md).

## Walkthrough

- `L66-67`: `method` must be `"spline"` or `"tree"`.
- `L68-69`: basis counts (`n_value_bases`, `n_time_bases`, `n_static_bases`),
  `n_value_bins`, and `tree_depth` must all be ≥1.
- `L70-71`: `temporal_penalty`, `static_penalty`, `learning_rate` must be `> 0`.
- `L72-73`: `max_rounds`, `early_stopping_rounds`, `outer_bags` must be ≥1.
- `L74`: `super().__init__(val_fraction=..., random_state=..., verbose=...)` —
  stores the params shared with [`_SSTGAMBase`](../sklearn_base/index.md).
- `L75-87`: store every method-specific parameter as a plain attribute — no
  branching here, following sklearn's `get_params`/`set_params` convention
  (constructor args become identically-named attributes, untouched).

## Calls

- `_SSTGAMBase.__init__`

## Called by

- User code: `SSTGAM(method="spline", ...)`

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
