# `SharedShapeTemporalGAMTreeBoost._bin`

**File:** `src/sstgam/tree_boost.py:68-70`
**Kind:** staticmethod

## Purpose

Maps continuous values to integer bin indices given a set of edges, via
binary search.

## Walkthrough

- `L70`: `np.searchsorted(edges[1:-1], values, side="right")` — uses only the
  interior edges (drops the two outer boundary edges) so values land in
  `[0, n_bins - 1]`.

## Calls

None.

## Called by

- [`feature_bins.md`](feature_bins.md)
- [`static_bins.md`](static_bins.md)
- [`predict_proba.md`](predict_proba.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
