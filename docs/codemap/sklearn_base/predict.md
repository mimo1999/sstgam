# `_SSTGAMBase.predict`

**File:** `src/sstgam/_base.py:150-151`
**Kind:** instance method

## Purpose

Hard 0/1 class label at the standard 0.5 probability threshold.

## Walkthrough

- `L151`: `(self._positive_probability(X) >= 0.5).astype(int)`.

## Calls

- [`positive_probability.md`](positive_probability.md)

## Called by

- User code (`model.predict(X)`)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
