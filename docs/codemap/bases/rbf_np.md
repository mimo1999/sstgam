# `rbf_np`

**File:** `src/sstgam/bases.py:30-33`
**Kind:** module-level function

## Purpose

Vectorized Gaussian RBF design matrix: evaluates every center against every
input value in one broadcasted call. `z` of shape `(...)` produces output of
shape `(..., P)` where `P = len(centers)`.

## Walkthrough

- `L32`: `d = (z[..., None] - centers) / bandwidth` — broadcasts `z` against
  every center along a new trailing axis, producing scaled distances.
- `L33`: `exp(-0.5 * d**2)` — a Gaussian bump centered on each RBF center.

## Calls

None.

## Called by

- [`../boosting/build_designs.md`](../boosting/build_designs.md) — value basis and static basis
- [`../boosting/fit.md`](../boosting/fit.md) — time basis, evaluated once up front
- [`../boosting/temporal_surface_normalized.md`](../boosting/temporal_surface_normalized.md) — value and time basis for inspection/plotting

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
