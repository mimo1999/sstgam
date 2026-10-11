# `_SSTGAMBase._normalise_time`

**File:** `src/sstgam/_base.py:175-177`
**Kind:** instance method

## Purpose

Rescales raw time coordinates into the `[0, 1]` range the spline backend's
time basis was fit on.

## Walkthrough

- `L176-177`: `(t - min_fitted_time) / max(max_fitted_time - min_fitted_time, 1.0)`,
  using `self.time_ids_[0]`/`self.time_ids_[-1]` as the fitted bounds.

## Calls

None.

## Called by

- [`../classifier/temporal_surface.md`](../classifier/temporal_surface.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
