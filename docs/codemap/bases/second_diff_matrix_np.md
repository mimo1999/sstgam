# `second_diff_matrix_np`

**File:** `src/sstgam/bases.py:36-52`
**Kind:** module-level function

## Purpose

Builds the discrete second-difference operator `D`, shape `(n-2, n)`, such
that `||D c||^2 = sum_i (c_{i-1} - 2c_i + c_{i+1})^2` penalizes curvature of
coefficient sequence `c` — the discrete analogue of a spline's integrated
squared second derivative.

## Walkthrough

- `L45-46`: fewer than 3 centers means curvature can't be defined; return an
  empty `(0, n)` matrix.
- `L47-51`: otherwise build `(n-2, n)` with the `[1, -2, 1]` stencil on row `i`
  at columns `i, i+1, i+2`.

## Calls

None.

## Called by

- [`../boosting/penalty_matrix.md`](../boosting/penalty_matrix.md) — squared (`D^T D`) for both the value and time axes
- [`../boosting/static_penalty.md`](../boosting/static_penalty.md) — squared for the 1-D static-effect axis

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
