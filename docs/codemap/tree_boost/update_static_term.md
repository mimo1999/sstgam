# `SharedShapeTemporalGAMTreeBoost._update_static_term`

**File:** `src/sstgam/tree_boost.py:180-184`
**Kind:** instance method

## Purpose

Same aggregation idea as [`_update_temporal_term`](update_temporal_term.md)
but for a single static column — no observation mask needed, since missing
static values are already imputed by [`_static_bins`](static_bins.md) before
this is ever called.

## Walkthrough

- `L182-183`: `np.bincount(bins, weights=gradient/hessian, minlength=n_bins)`.

## Calls

- [`tree_update.md`](tree_update.md)

## Called by

- [`fit_one_bag.md`](fit_one_bag.md) — once per static column, per round

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
