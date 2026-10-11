# `_SSTGAMBase._positive_probability`

**File:** `src/sstgam/_base.py:141-144`
**Kind:** instance method

## Purpose

Shared prediction path used by both `predict_proba` and `predict`: rebuild
the tensors for new `X` and ask the fitted backend (`self.model_`) for
`P(y=1)`.

## Walkthrough

- `L142`: `check_is_fitted(self, "model_")`.
- `L143`: [`_prepare_predict_input`](prepare_predict_input.md) then [`_build_tensors`](build_tensors.md).
- `L144`: delegate to `self.model_.predict_proba(values, observed, static_values)`.

## Calls

- [`_prepare_predict_input`](prepare_predict_input.md)
- [`_build_tensors`](build_tensors.md)
- Backend `predict_proba` — either [`../boosting/predict_proba.md`](../boosting/predict_proba.md) or
  [`../tree_boost/predict_proba.md`](../tree_boost/predict_proba.md), whichever class `self.model_` is an instance of.

## Called by

- [`predict_proba.md`](predict_proba.md)
- [`predict.md`](predict.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
