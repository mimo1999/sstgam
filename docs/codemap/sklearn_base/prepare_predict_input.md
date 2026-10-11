# `_SSTGAMBase._prepare_predict_input`

**File:** `src/sstgam/_base.py:64-77`
**Kind:** instance method

## Purpose

At prediction time, realigns an incoming DataFrame's columns to the fitted
column order (or just validates column count for plain arrays), guarding
against silently-misaligned inputs.

## Walkthrough

- `L65-73`: if `X` has `.columns`, compute `missing`/`extra` vs.
  `self.feature_names_in_` and raise `ValueError` if either is non-empty;
  otherwise reorder `X` to `self.feature_names_in_` via `.loc[:, ...]`.
- `L74-76`: convert to a float matrix; for the plain-array path (which skipped
  the name check above), validate the column count matches.

## Calls

- [`_as_float_matrix`](as_float_matrix.md)

## Called by

- [`positive_probability.md`](positive_probability.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
