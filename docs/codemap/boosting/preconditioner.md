# `SharedShapeTemporalGAMBoost._preconditioner`

**File:** `src/sstgam/boosting.py:74-78`
**Kind:** staticmethod

## Purpose

Precomputes a fixed inverse-Hessian-bound matrix per feature (or per static
column) — this is what makes the per-round Newton step in
[`_update_block`](update_block.md) a single matrix-vector multiply instead of
a fresh linear solve every round.

## Walkthrough

- `L76`: `0.25 * sample_weight` — `p(1-p) ≤ 0.25` for any Bernoulli
  probability, so this is an *upper bound* on the true per-sample Hessian.
  Using a fixed bound (rather than the true, score-dependent Hessian) is what
  lets the preconditioner stay valid across the whole boosting run.
- `L77`: `hessian_bound = design^T @ weighted_design + penalty` — the bounded
  Gauss-Newton Hessian plus the feature's roughness penalty.
- `L78`: `np.linalg.inv(hessian_bound)`.

## Calls

- `np.linalg.inv` (external)

## Called by

- [`fit_one_bag.md`](fit_one_bag.md) — once per feature (and once per static
  column) at the start of each bag, not once per round

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
