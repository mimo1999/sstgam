# `fit_value_normalization`

**File:** `src/sstgam/normalization.py:57-71`
**Kind:** module-level function

## Purpose

Dispatches to one of the two supported `value_transform` strategies and
returns a uniform 4-tuple `(normalized_values, center, scale, tables)`
regardless of which was used, so callers don't need to branch.

## Walkthrough

- `L60-67`: `"quantile"` — [`fit_quantile_tables`](fit_quantile_tables.md), then
  [`apply_quantile_tables`](apply_quantile_tables.md) on the training data
  itself; `center`/`scale` returned as placeholder `0`/`1` since this
  transform doesn't use them.
- `L68-70`: `"zscore"` — [`robust_center_scale`](robust_center_scale.md)
  computes `center`/`scale`, but `values` are returned **unnormalized** here;
  the actual `(v - center) / scale` happens later, inside
  `boosting._build_designs`.
- `L71`: unknown transform → `ValueError`.

## Calls

- [`fit_quantile_tables`](fit_quantile_tables.md)
- [`apply_quantile_tables`](apply_quantile_tables.md)
- [`robust_center_scale`](robust_center_scale.md)

## Called by

- [`../boosting/fit.md`](../boosting/fit.md) — once, during setup, before building the temporal design matrix

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
