# `SSTGAM._fit_trainer`

**File:** `src/sstgam/classifier.py:89-95`
**Kind:** instance method (overrides abstract hook)

## Purpose

The concrete implementation of the hook `_SSTGAMBase.fit` calls: builds a
fresh backend trainer and fits it on the already-tensorized data.

## Walkthrough

- `L91`: `trainer = self._make_trainer()`.
- `L92-95`: `return trainer.fit(values, observed, static_values, time_coordinates, target, sample_weight=sample_weight)`.

The returned fitted trainer becomes `self.model_` back in `_base.fit`.

## Calls

- [`make_trainer.md`](make_trainer.md)
- Then the chosen backend's `fit` — [`../boosting/fit.md`](../boosting/fit.md) or
  [`../tree_boost/fit.md`](../tree_boost/fit.md), whichever `_make_trainer` returned.

## Called by

- [`../sklearn_base/fit.md`](../sklearn_base/fit.md) — `L130-133`, as the abstract hook's concrete implementation

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
