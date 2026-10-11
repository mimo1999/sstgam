# `_SSTGAMBase._as_float_matrix`

**File:** `src/sstgam/_base.py:43-48`
**Kind:** staticmethod

## Purpose

Normalizes either a DataFrame or an array-like into a plain 2-D `float32`
NumPy array.

## Walkthrough

- `L45`: `X.values` if `X` has that attribute (DataFrame), else `X` as-is; cast to `float32`.
- `L46-47`: reject anything that isn't exactly 2-D.

## Calls

None.

## Called by

- [`prepare_fit_input.md`](prepare_fit_input.md)
- [`prepare_predict_input.md`](prepare_predict_input.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
