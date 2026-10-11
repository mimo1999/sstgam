# `SharedShapeTemporalGAMTreeBoost._weighted_gain`

**File:** `src/sstgam/tree_boost.py:108-116`
**Kind:** staticmethod

## Purpose

Weighted-SSE reduction from predicting the overall weighted mean vs.
predicting `prediction` — the raw gain term before the axis-choice penalty is
subtracted in [`_adaptive_tree_update`](adaptive_tree_update.md).

## Walkthrough

- `L111-112`: return `0.0` if total weight is non-positive.
- `L113`: `mean = weighted_average(target, weight)`.
- `L114-115`: `loss_root` = weighted SSE against that mean; `loss_leaf` =
  weighted SSE against `prediction`.
- `L116`: `max(loss_root - loss_leaf, 0.0)`.

## Calls

None.

## Called by

- [`adaptive_tree_update.md`](adaptive_tree_update.md) — to score each axis's depth-1 candidate split

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
