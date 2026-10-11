# `_SSTGAMBase._prepare_fit_input`

**File:** `src/sstgam/_base.py:50-62`
**Kind:** instance method

## Purpose

Resolves the training matrix and one column name per column, validating that
names line up with the data and are unique.

## Walkthrough

- `L51`: convert `X` via [`_as_float_matrix`](as_float_matrix.md).
- `L52-57`: `feature_names` defaults to `X.columns` if `X` is a DataFrame,
  else synthetic `feature_0`, `feature_1`, ... names.
- `L58-61`: validate the name count matches the column count, and names are unique.

## Calls

- [`_as_float_matrix`](as_float_matrix.md)

## Called by

- [`fit.md`](fit.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
