# `SharedShapeTemporalGAMTreeBoost._fit_bins`

**File:** `src/sstgam/tree_boost.py:72-84`
**Kind:** instance method

## Purpose

Fits quantile bin edges once per temporal feature (from pooled *observed*
values across all times) and once per static column (from finite values),
also recording each static column's median as its fill value for missing
entries.

## Walkthrough

- `L74-76`: per temporal feature, pool observed values across the time axis
  (`observed > 0.5` mask), compute quantile edges ([`quantile_edges.md`](quantile_edges.md)).
- `L78-84`: per static column, pool finite values; store their median as
  `self._static_fill[column_index]` (used to impute missing static values at
  both train and predict time); compute quantile edges.

## Calls

- [`quantile_edges.md`](quantile_edges.md)

## Called by

- [`fit.md`](fit.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
