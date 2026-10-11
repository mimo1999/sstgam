# `SharedShapeTemporalGAMTreeBoost._feature_bins`

**File:** `src/sstgam/tree_boost.py:86-88`
**Kind:** instance method

## Purpose

Bins every temporal feature's raw value tensor using the already-fitted
`self._value_edges`.

## Walkthrough

- `L87-88`: per feature, `np.nan_to_num(values[:, feature_index, :], nan=0.0)`
  then [`_bin`](bin.md) against that feature's fitted edges. NaN-filling is
  harmless here since unobserved cells are masked out by `observed` wherever
  this result is used downstream.

## Calls

- [`bin.md`](bin.md)

## Called by

- [`fit.md`](fit.md)
- [`predict_proba.md`](predict_proba.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
