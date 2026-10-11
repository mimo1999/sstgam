# `SSTGAM.value_effect`

**File:** `src/sstgam/classifier.py:123-130`
**Kind:** instance method (public API, spline-only)

## Purpose

Returns the fitted spline effect of a feature's *value* alone, averaged out
over time.

## Walkthrough

- `L125-126`: guard `self.method == "spline"`; `check_is_fitted`.
- `L128`: `feature_index = self.temporal_features_.index(feature)`; normalize
  query `values` via [`../sklearn_base/normalise_feature_values.md`](../sklearn_base/normalise_feature_values.md).
- `L130`: `self.model_.mean_value_effect_normalized(feature_index, normalized_values)`.

## Calls

- [`../sklearn_base/normalise_feature_values.md`](../sklearn_base/normalise_feature_values.md)
- [`../boosting/mean_value_effect_normalized.md`](../boosting/mean_value_effect_normalized.md)

## Called by

- User code: `model.value_effect("signal_a", values=[0.0, 0.5, 1.0])`

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
