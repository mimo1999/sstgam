# `SharedShapeTemporalGAMTreeBoost._static_bins`

**File:** `src/sstgam/tree_boost.py:90-93`
**Kind:** instance method

## Purpose

Fills missing static values with the fitted per-column median, then bins with
the fitted `self._static_edges`.

## Walkthrough

- `L91`: `np.where(np.isfinite(static_values), static_values, self._static_fill)`.
- `L92-93`: [`_bin`](bin.md) per column against that column's fitted edges.

## Calls

- [`bin.md`](bin.md)

## Called by

- [`fit.md`](fit.md)
- [`predict_proba.md`](predict_proba.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
