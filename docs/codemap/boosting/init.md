# `SharedShapeTemporalGAMBoost.__init__`

**File:** `src/sstgam/boosting.py:22-51`
**Kind:** constructor

## Purpose

Stores every spline-backend hyperparameter as a plain attribute, plus fixes
`self.clip = _NORMALIZATION_CLIP` (4.0) — the bound every normalized
value/coordinate is clipped to before basis evaluation.

## Walkthrough

- `L38-50`: assign each constructor argument (`n_value_bases`, `n_time_bases`,
  `n_static_bases`, `value_transform`, `temporal_penalty`, `static_penalty`,
  `learning_rate`, `max_rounds`, `outer_bags`, `early_stopping_rounds`,
  `val_fraction`, `random_state`, `verbose`) directly to `self`.
- `L51`: `self.clip = _NORMALIZATION_CLIP`.

No branching logic — pure attribute assignment.

## Calls

None.

## Called by

- [`../classifier/make_trainer.md`](../classifier/make_trainer.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
