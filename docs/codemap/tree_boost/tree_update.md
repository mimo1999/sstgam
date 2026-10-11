# `SharedShapeTemporalGAMTreeBoost._tree_update`

**File:** `src/sstgam/tree_boost.py:95-106`
**Kind:** instance method

## Purpose

A plain single-stage regression tree Newton update for a 1-D (static) term —
no adaptive axis choice needed since there's only one axis to split on.

## Walkthrough

- `L97-99`: `active = hessian_sum > _MIN_HESSIAN` — skip bins whose
  accumulated Hessian is too small to trust; bail out entirely (`return
  None`) if fewer than 2 usable bins remain.
- `L100`: Newton target per active bin = `gradient_sum / hessian_sum`.
- `L101-105`: fit a `DecisionTreeRegressor(max_depth=self.max_depth)` on the
  active bins' coordinates, weighted by their Hessian.
- `L106`: predict on *every* bin (including inactive ones, which inherit
  their nearest active leaf's prediction).

## Calls

- `sklearn.tree.DecisionTreeRegressor` (external)

## Called by

- [`update_static_term.md`](update_static_term.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
