# `SharedShapeTemporalGAMBoost._static_penalty`

**File:** `src/sstgam/boosting.py:68-72`
**Kind:** staticmethod

## Purpose

Same idea as [`_penalty_matrix`](penalty_matrix.md) but for a single 1-D
static-effect coefficient vector — no Kronecker product needed since there's
only one axis.

## Walkthrough

- `L70`: second-difference operator on `n_bases` ([`../bases/second_diff_matrix_np.md`](../bases/second_diff_matrix_np.md)).
- `L71-72`: `strength * (D^T D) + _RIDGE * I`.

## Calls

- [`../bases/second_diff_matrix_np.md`](../bases/second_diff_matrix_np.md)

## Called by

- [`fit.md`](fit.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
