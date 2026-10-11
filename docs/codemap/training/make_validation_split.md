# `make_validation_split`

**File:** `src/sstgam/_training.py:7-36`
**Kind:** module-level function

## Purpose

Builds one reproducible train/validation row split, stratified by the binary
target when both classes have enough samples, otherwise a plain random split.
Both boosting backends call this **once**, before their `outer_bags` loop, so
every bag is judged for early stopping on the exact same held-out rows — only
the bootstrapped *training* rows vary across bags.

## Walkthrough

- `L13-16`: ravel `target`, require at least 2 samples.
- `L17-18`: validate `0.0 < fraction < 1.0`.
- `L20-23`: compute `n_validation` (at least 1, at most `n_samples - 1`); count
  per-class occurrences; `can_stratify` requires both classes present, each
  with ≥2 samples, and `n_validation` large enough to hold at least one of
  each class.
- `L24-30`: if stratifiable, delegate straight to sklearn's
  `train_test_split(..., stratify=target)`.
- `L32-36`: otherwise, seed a private `RandomState(random_state)` and sample
  `n_validation` indices without replacement as validation; everything else
  is training. This path keeps small or heavily imbalanced datasets usable
  even when stratification isn't possible.

## Calls

- `sklearn.model_selection.train_test_split` (external)

## Called by

- [`../boosting/fit.md`](../boosting/fit.md) — `L162-164`, before the bagging loop
- [`../tree_boost/fit.md`](../tree_boost/fit.md) — `L229-231`, before the bagging loop

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
