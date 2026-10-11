# `_SSTGAMBase._fit_trainer` (abstract)

**File:** `src/sstgam/_base.py:137-139`
**Kind:** instance method (raises `NotImplementedError`)

## Purpose

Placeholder seam between the generic sklearn wrapper (`_base.py`) and the two
concrete boosting backends. The base class always raises; `classifier.SSTGAM`
supplies the real implementation.

## Walkthrough

- `L138-139`: `raise NotImplementedError`.

## Calls

None.

## Called by

- [`fit.md`](fit.md) — `L130-133`

## Overridden by

- [`../classifier/fit_trainer.md`](../classifier/fit_trainer.md) — `SSTGAM._fit_trainer`

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
