# `make_rbf_centers`

**File:** `src/sstgam/bases.py:17-27`
**Kind:** module-level function

## Purpose

Returns `n` equally-spaced RBF centers on `[lo, hi]` and a bandwidth equal to
the center spacing, which gives roughly 50% neighbour overlap — smooth
reconstruction without gaps between basis "bumps."

## Walkthrough

- `L23`: guard `n >= 1`.
- `L24`: `np.linspace(lo, hi, n)` for the centers.
- `L25`: `spacing = (hi - lo) / (n - 1)` if `n > 1`, else the full range (single-center case).
- `L26`: `bandwidth = spacing` unless it's ~0 (degenerate range or `n=1`), in which case fall back to `1.0`.

## Calls

None.

## Called by

- [`../boosting/fit.md`](../boosting/fit.md) — three times: value centers, static centers, time centers.

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
