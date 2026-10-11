# `SharedShapeTemporalGAMTreeBoost._update_temporal_term`

**File:** `src/sstgam/tree_boost.py:171-178`
**Kind:** instance method

## Purpose

Aggregates per-sample gradient/Hessian into per-`(value_bin, time)` cell
sums, respecting the observation mask, then hands them to the adaptive tree
update.

## Walkthrough

- `L173-174`: broadcast `gradient`/`hessian` (one per sample) across the time
  axis and zero out cells where `observed` is 0.
- `L175-177`: `np.bincount(flat_cells, weights=...)` accumulates
  `gradient_sum`/`hessian_sum` per flattened cell id (`cell_ids.ravel()`).

## Calls

- [`adaptive_tree_update.md`](adaptive_tree_update.md)

## Called by

- [`fit_one_bag.md`](fit_one_bag.md) — once per temporal feature, per round

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
