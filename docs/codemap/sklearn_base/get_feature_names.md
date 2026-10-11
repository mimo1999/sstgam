# `_SSTGAMBase.get_feature_names`

**File:** `src/sstgam/_base.py:156-159`
**Kind:** instance method

## Purpose

Public introspection API — returns all fitted input column names, in their
required (original) order.

## Walkthrough

- `L158`: `check_is_fitted(self, "model_")`.
- `L159`: `list(self.feature_names_in_)`.

## Calls

None.

## Called by

None internally — public API for user code.

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
