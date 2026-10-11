# `SharedShapeTemporalGAMTreeBoost._grad_hess`

**File:** `src/sstgam/tree_boost.py:186-191`
**Kind:** staticmethod

## Purpose

Logistic gradient/Hessian at the current score — recomputed fresh before
*every single term's* update (unlike the spline backend, which precomputes
one fixed preconditioner per bag and reuses it all round).

## Walkthrough

- `L188`: `probability = sigmoid(score)`.
- `L189`: `gradient = weight * (target - probability)`.
- `L190`: `hessian = weight * probability * (1 - probability)`.

## Calls

None.

## Called by

- [`fit_one_bag.md`](fit_one_bag.md) — once for the intercept step, then again
  before every feature's and every static column's update within a round,
  since each update changes the score the next one sees

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
