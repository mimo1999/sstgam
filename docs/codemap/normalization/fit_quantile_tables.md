# `fit_quantile_tables`

**File:** `src/sstgam/normalization.py:25-44`
**Kind:** module-level function

## Purpose

Fits one empirical-CDF lookup table per feature, mapping raw observed values
to ranks in `(-1, 1)` — the default `value_transform="quantile"`.

## Walkthrough

- `L28-29`: gather observed, finite values for one feature, pooled across time.
- `L31-33`: fewer than 2 values → fall back to a trivial 2-point identity table `([-1,1],[-1,1])`.
- `L35-36`: sort values; assign each its midpoint rank `(i+0.5)/n` in `[0,1]`.
- `L37-42`: collapse duplicate values to their mean rank — `np.unique(..., return_inverse=True)` groups ties, `np.add.at` accumulates rank sums and counts per unique value.
- `L43`: rescale unique ranks from `[0,1]` to `(-1,1)`; store `(unique_values, unique_ranks)` as the table used for interpolation.

## Calls

None.

## Called by

- [`fit_value_normalization.md`](fit_value_normalization.md) — `"quantile"` branch

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
