# `_SSTGAMBase.fit`

**File:** `src/sstgam/_base.py:101-135`
**Kind:** instance method

## Purpose

The sklearn-facing `fit()` entry point: validates `X`/`y`/`sample_weight`,
parses columns, builds tensors, and delegates the actual optimization to
whichever backend subclass implements `_fit_trainer`.

## Walkthrough

- `L102-107`: parse the input matrix/names ([`_prepare_fit_input`](prepare_fit_input.md)); validate
  the target is binary `{0, 1}` and row counts match.
- `L108-114`: validate optional `sample_weight` — non-negative, positive
  total, matching row count.
- `L116-118`: [`_parse_columns`](parse_columns.md); require at least one temporal column.
- `L120-127`: stash fitted metadata: `self.feature_names_in_`,
  `self._temporal_columns`, `self._static_column_indices`,
  `self.temporal_features_` (sorted feature names),
  `self.time_ids_` (sorted union of every time id across all features).
- `L129-133`: [`_build_tensors`](build_tensors.md), then call `self._fit_trainer(...)` — the
  one call site where the abstract hook is invoked, dispatching to the
  concrete override in `classifier.SSTGAM`.
- `L134`: `self.classes_ = np.array([0, 1])` for sklearn's `ClassifierMixin`
  conventions.

## Calls

- [`_prepare_fit_input`](prepare_fit_input.md)
- [`_parse_columns`](parse_columns.md)
- [`_build_tensors`](build_tensors.md)
- `_fit_trainer` (abstract — concrete implementation at [`../classifier/fit_trainer.md`](../classifier/fit_trainer.md))

## Called by

- User code: `model.fit(X, y)`

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
