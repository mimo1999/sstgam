# `SharedShapeTemporalGAMTreeBoost.predict_proba`

**File:** `src/sstgam/tree_boost.py:337-349`
**Kind:** instance method

## Purpose

Tree-backend prediction: bin the new data with the fitted edges, gather each
cell's fitted table value, and sum with the intercept.

## Walkthrough

- `L341-344`: per feature, compute cell ids from bins
  ([`feature_bins.md`](feature_bins.md)), gather
  `self._feature_tables[feature][cell_ids]`, mask by `observed`, sum across
  time.
- `L345-347`: per static column, gather `self._static_tables[column][bins]`
  ([`static_bins.md`](static_bins.md)).
- `L348-349`: `sigmoid(score)`, clipped away from 0/1.

## Calls

- [`feature_bins.md`](feature_bins.md)
- [`static_bins.md`](static_bins.md)

## Called by

- [`../sklearn_base/positive_probability.md`](../sklearn_base/positive_probability.md) (via `self.model_.predict_proba`, when `self.model_` is a `SharedShapeTemporalGAMTreeBoost`)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
