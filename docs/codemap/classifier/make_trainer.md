# `SSTGAM._make_trainer`

**File:** `src/sstgam/classifier.py:97-121`
**Kind:** instance method

## Purpose

Constructs the right backend object from `self.method`, forwarding only the
parameters that backend actually uses.

## Walkthrough

- `L98-106`: `common_arguments` — the boosting-control params shared by both
  backends: `learning_rate`, `max_rounds`, `outer_bags`,
  `early_stopping_rounds`, `val_fraction`, `random_state`, `verbose`.
- `L107-112`: `method == "tree"` →
  `SharedShapeTemporalGAMTreeBoost(n_value_bins=self.n_value_bins, max_depth=self.tree_depth, **common_arguments)`.
- `L113-121`: else (`"spline"`) →
  `SharedShapeTemporalGAMBoost(n_value_bases=..., n_time_bases=..., n_static_bases=..., value_transform=..., temporal_penalty=..., static_penalty=..., **common_arguments)`.

## Calls

- `SharedShapeTemporalGAMTreeBoost.__init__` — [`../tree_boost/init.md`](../tree_boost/init.md)
- `SharedShapeTemporalGAMBoost.__init__` — [`../boosting/init.md`](../boosting/init.md)

## Called by

- [`fit_trainer.md`](fit_trainer.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
