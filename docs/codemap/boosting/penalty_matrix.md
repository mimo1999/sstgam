# `SharedShapeTemporalGAMBoost._penalty_matrix`

**File:** `src/sstgam/boosting.py:53-66`
**Kind:** staticmethod

## Purpose

Builds the roughness-penalty matrix for one feature's flattened
`(n_value_bases × n_time_bases)` tensor-product coefficient vector —
penalizing curvature along the value axis, along the time axis, and adding a
small ridge for numerical stability.

## Walkthrough

- `L56-57`: 1-D second-difference operators for the value axis and time axis
  ([`../bases/second_diff_matrix_np.md`](../bases/second_diff_matrix_np.md)).
- `L58-61`: turn each into a roughness matrix `D^T D` (a zero matrix if there
  are too few centers to define curvature).
- `L62-66`: combine via Kronecker products:
  `kron(value_penalty, I_time)` penalizes value-axis curvature while leaving
  time free; `kron(I_value, time_penalty)` does the reverse; plus
  `strength * I` (an isotropic ridge scaled by `temporal_penalty`) and
  `_RIDGE * I` (a tiny numerical-stability ridge).

## Calls

- [`../bases/second_diff_matrix_np.md`](../bases/second_diff_matrix_np.md)

## Called by

- [`fit.md`](fit.md) — once, building the shared temporal penalty applied
  identically to every feature's coefficients

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
