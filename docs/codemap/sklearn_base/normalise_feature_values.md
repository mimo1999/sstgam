# `_SSTGAMBase._normalise_feature_values`

**File:** `src/sstgam/_base.py:166-173`
**Kind:** instance method

## Purpose

Maps raw user-supplied values for one feature into the same normalized
coordinate the spline backend was trained on, so `value_effect`/
`temporal_surface` can be evaluated at arbitrary query points. Only
meaningful for `method="spline"` — the tree backend doesn't expose this path.

## Walkthrough

- `L168-170`: `"quantile"` transform — reuse the fitted backend's
  `self.model_._quantile_tables[feature_index]` via `np.interp`, then scale
  by `self.model_.clip`. Mirrors [`../normalization/apply_quantile_tables.md`](../normalization/apply_quantile_tables.md)
  but reads the table straight off the fitted spline model rather than going
  through the module function.
- `L171-173`: `"zscore"` transform — subtract the backend's fitted
  `_feature_center[feature_index]`, divide by `_feature_scale[feature_index]`,
  clip to `[-clip, clip]`.

## Calls

None — reimplements the transform inline against `self.model_`'s fitted state.

## Called by

- [`../classifier/value_effect.md`](../classifier/value_effect.md)
- [`../classifier/temporal_surface.md`](../classifier/temporal_surface.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
