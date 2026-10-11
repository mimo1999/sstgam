# `weighted_log_loss`

**File:** `src/sstgam/_training.py:39-45`
**Kind:** module-level function

## Purpose

Weighted binary cross-entropy. This is the metric both backends evaluate on
the fixed validation split every boosting round, to decide whether to
snapshot the current coefficients as the new "best" state.

## Walkthrough

- `L41`: clip `probability` into `[1e-7, 1-1e-7]` so `log(0)` never happens.
- `L42-44`: `-average(y*log(p) + (1-y)*log(1-p), weights=weight)`.

## Calls

None — pure NumPy.

## Called by

- [`../boosting/fit_one_bag.md`](../boosting/fit_one_bag.md) — every round, on `validation_probability`
- [`../tree_boost/fit_one_bag.md`](../tree_boost/fit_one_bag.md) — every round, on `validation_probability`

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
