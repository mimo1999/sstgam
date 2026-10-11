# `static_stats`

**File:** `src/sstgam/normalization.py:79-90`
**Kind:** module-level function

## Purpose

Per-static-column finite-value mean/std, used to normalize static covariates
before RBF basis placement.

## Walkthrough

- `L84`: drop non-finite entries for a column.
- `L86-87`: mean/std of what's left.
- `L88-89`: fall back to `scale=1.0` if std is ~0 (constant column).

## Calls

None.

## Called by

- [`../boosting/fit.md`](../boosting/fit.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
