# `_SSTGAMBase.predict_proba`

**File:** `src/sstgam/_base.py:146-148`
**Kind:** instance method

## Purpose

sklearn-standard two-column output `[P(y=0), P(y=1)]`.

## Walkthrough

- `L147`: `probability = self._positive_probability(X)`.
- `L148`: `np.column_stack((1.0 - probability, probability))`.

## Calls

- [`positive_probability.md`](positive_probability.md)

## Called by

- User code (`model.predict_proba(X)`)
- [`predict_log_proba.md`](predict_log_proba.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
