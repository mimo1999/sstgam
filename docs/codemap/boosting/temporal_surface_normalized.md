# `SharedShapeTemporalGAMBoost.temporal_surface_normalized`

**File:** `src/sstgam/boosting.py:275-285`
**Kind:** instance method

## Purpose

Evaluates one feature's fitted `f_k(value, time)` surface at an arbitrary
grid of already-normalized value/time coordinates — the introspection API
behind `classifier.temporal_surface`.

## Walkthrough

- `L278-281`: evaluate the value RBF basis and the time RBF basis at the
  requested grids ([`../bases/rbf_np.md`](../bases/rbf_np.md)).
- `L282-284`: reshape that feature's flat fitted coefficient vector back into
  `(n_value_bases, n_time_bases)`.
- `L285`: `value_basis @ coefficients @ time_basis.T` — the tensor-product
  surface evaluation (matches the `f_k` equation in
  [`ARCHITECTURE.md`](../../../ARCHITECTURE.md)).

## Calls

- [`../bases/rbf_np.md`](../bases/rbf_np.md)

## Called by

- [`../classifier/temporal_surface.md`](../classifier/temporal_surface.md)
- [`mean_value_effect_normalized.md`](mean_value_effect_normalized.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
