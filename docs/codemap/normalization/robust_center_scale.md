# `robust_center_scale`

**File:** `src/sstgam/normalization.py:6-22`
**Kind:** module-level function

## Purpose

Per-feature median center and IQR-based robust scale — the `value_transform="zscore"` alternative to quantile normalization.

## Walkthrough

- `L11-13`: gather only observed and finite values for a feature, pooled across all time points.
- `L14-15`: leave defaults (center=0, scale=1) if there are no observed values at all.
- `L16`: median as the center.
- `L17-18`: `scale = (P75 - P25) / 1.349` — a robust stand-in for standard deviation (the 1.349 constant makes it ≈ std under normality).
- `L19-21`: if that scale is ~0 (e.g. a near-constant feature), fall back to raw `std`; if even that is ~0, fall back to `1.0`.

## Calls

None.

## Called by

- [`fit_value_normalization.md`](fit_value_normalization.md) — `"zscore"` branch

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
