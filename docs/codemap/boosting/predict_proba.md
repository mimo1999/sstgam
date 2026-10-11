# `SharedShapeTemporalGAMBoost.predict_proba`

**File:** `src/sstgam/boosting.py:260-273`
**Kind:** instance method

## Purpose

Spline-backend prediction: reapply the fitted value normalization and design
construction to new data, then a plain linear score through the intercept
plus the bagged-averaged coefficients.

## Walkthrough

- `L261-264`: [`../normalization/apply_value_normalization.md`](../normalization/apply_value_normalization.md).
- `L265-267`: [`build_designs.md`](build_designs.md) on the normalized new data.
- `L268-271`: `score = intercept + einsum("nfp,fp->n", temporal_design, temporal_coefficients)`
  `+ einsum("nsp,sp->n", static_design, static_coefficients)` (if static columns exist).
- `L272-273`: `sigmoid(score)`, clipped away from 0/1.

## Calls

- [`../normalization/apply_value_normalization.md`](../normalization/apply_value_normalization.md)
- [`build_designs.md`](build_designs.md)

## Called by

- [`../sklearn_base/positive_probability.md`](../sklearn_base/positive_probability.md) (via `self.model_.predict_proba`, when `self.model_` is a `SharedShapeTemporalGAMBoost`)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
