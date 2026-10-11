# `SharedShapeTemporalGAMBoost.mean_value_effect_normalized`

**File:** `src/sstgam/boosting.py:287-291`
**Kind:** instance method

## Purpose

The value-only effect exposed by `classifier.value_effect` — evaluates the
fitted surface on a dense time grid and averages across time.

## Walkthrough

- `L288-290`: [`temporal_surface_normalized.md`](temporal_surface_normalized.md) on the
  requested value grid crossed with `np.linspace(0, 1, n_time_bases*3)`
  (an oversampled time grid, denser than the fitted time basis, for a smooth average).
- `L291`: `surface.mean(axis=1)` — average across the time axis.

## Calls

- [`temporal_surface_normalized.md`](temporal_surface_normalized.md)

## Called by

- [`../classifier/value_effect.md`](../classifier/value_effect.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
