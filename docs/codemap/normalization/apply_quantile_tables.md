# `apply_quantile_tables`

**File:** `src/sstgam/normalization.py:47-54`
**Kind:** module-level function

## Purpose

Applies a fitted quantile table to a values tensor via linear interpolation,
scaling the result into `(-clip, clip)`.

## Walkthrough

- `L50-53`: per feature, `np.interp(values, x_values, y_values) * clip` — maps
  each raw value against the table's `(x, y)` pairs built by
  [`fit_quantile_tables`](fit_quantile_tables.md), then scales.

## Calls

None.

## Called by

- [`fit_value_normalization.md`](fit_value_normalization.md) — to transform training data right after fitting the table
- [`apply_value_normalization.md`](apply_value_normalization.md) — at prediction time

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
