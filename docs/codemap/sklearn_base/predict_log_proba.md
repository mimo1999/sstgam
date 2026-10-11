# `_SSTGAMBase.predict_log_proba`

**File:** `src/sstgam/_base.py:153-154`
**Kind:** instance method

## Purpose

Log-probabilities, clipped away from 0 to avoid `-inf`.

## Walkthrough

- `L154`: `np.log(np.clip(self.predict_proba(X), 1e-10, 1.0))`.

## Calls

- [`predict_proba.md`](predict_proba.md)

## Called by

- User code / sklearn conventions

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
