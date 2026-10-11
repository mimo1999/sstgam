# `apply_value_normalization`

**File:** `src/sstgam/normalization.py:74-76`
**Kind:** module-level function

## Purpose

Prediction-time counterpart to [`fit_value_normalization`](fit_value_normalization.md) — applies whichever
transform was fit, given only the transform tag and the fitted table.

## Walkthrough

- `L76`: `"quantile"` → [`apply_quantile_tables`](apply_quantile_tables.md) against the
  fitted tables; anything else (`"zscore"`) → pass values through unchanged
  (centering/scaling happens later inside `_build_designs`, same as at fit
  time).

## Calls

- [`apply_quantile_tables`](apply_quantile_tables.md)

## Called by

- [`../boosting/predict_proba.md`](../boosting/predict_proba.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
