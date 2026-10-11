# `SSTGAM.temporal_surface`

**File:** `src/sstgam/classifier.py:132-142`
**Kind:** instance method (public API, spline-only)

## Purpose

Evaluates the full value×time log-odds surface for one feature at arbitrary
query grids.

## Walkthrough

- `L134-136`: guard `method == "spline"`; `check_is_fitted`.
- `L138`: normalize query `values` via [`../sklearn_base/normalise_feature_values.md`](../sklearn_base/normalise_feature_values.md).
- `L139`: normalize `time_coordinates` via [`../sklearn_base/normalise_time.md`](../sklearn_base/normalise_time.md).
- `L140-142`: `self.model_.temporal_surface_normalized(feature_index, normalized_values, normalized_time)`.

## Calls

- [`../sklearn_base/normalise_feature_values.md`](../sklearn_base/normalise_feature_values.md)
- [`../sklearn_base/normalise_time.md`](../sklearn_base/normalise_time.md)
- [`../boosting/temporal_surface_normalized.md`](../boosting/temporal_surface_normalized.md)

## Called by

- User code: `model.temporal_surface("signal_a", values=..., time_coordinates=...)`

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
