# `SharedShapeTemporalGAMBoost.fit`

**File:** `src/sstgam/boosting.py:124-193`
**Kind:** instance method

## Purpose

Top-level spline-backend fit: normalizes inputs and builds the fixed
bases/designs/penalties exactly once, then runs `outer_bags` independent
boosted fits on bootstrap resamples and averages their coefficients.

## Walkthrough

- `L135-138`: fit the value normalization once on the full training tensor
  ([`../normalization/fit_value_normalization.md`](../normalization/fit_value_normalization.md));
  fit static column mean/scale ([`../normalization/static_stats.md`](../normalization/static_stats.md)).
- `L139-145`: place value-axis RBF centers (span depends on the transform:
  full `clip` for quantile, `0.9*clip` for zscore to leave headroom) and
  static-axis RBF centers ([`../bases/make_rbf_centers.md`](../bases/make_rbf_centers.md)).
- `L146-151`: normalize `time_coordinates` to `[0,1]`; place time-axis RBF
  centers; evaluate the fixed time basis once
  ([`../bases/rbf_np.md`](../bases/rbf_np.md)) — this basis is shared across
  every feature and every bag, since time coordinates don't vary per bag.
- `L153-159`: build the full temporal/static design matrices
  ([`build_designs.md`](build_designs.md)) and the penalty matrices
  ([`penalty_matrix.md`](penalty_matrix.md), [`static_penalty.md`](static_penalty.md))
  — all exactly once, not once per bag.
- `L162-164`: [`../training/make_validation_split.md`](../training/make_validation_split.md) —
  fixed validation indices reused by every bag.
- `L166-192`: loop `outer_bags` times: draw a bootstrap resample of the
  training indices (or reuse them unchanged if `outer_bags == 1`), call
  [`fit_one_bag.md`](fit_one_bag.md), accumulate the returned
  intercept/coefficients. After the loop, divide every accumulator by
  `outer_bags` — the bagged average. Averaging is valid here because the
  model is linear-in-coefficients over a *shared, fixed* basis: a mean of
  coefficient vectors is another coefficient vector in the same form (see
  [`ARCHITECTURE.md`](../../../ARCHITECTURE.md), "Bagging").

## Calls

- [`../normalization/fit_value_normalization.md`](../normalization/fit_value_normalization.md)
- [`../normalization/static_stats.md`](../normalization/static_stats.md)
- [`../bases/make_rbf_centers.md`](../bases/make_rbf_centers.md)
- [`../bases/rbf_np.md`](../bases/rbf_np.md)
- [`build_designs.md`](build_designs.md)
- [`penalty_matrix.md`](penalty_matrix.md)
- [`static_penalty.md`](static_penalty.md)
- [`../training/make_validation_split.md`](../training/make_validation_split.md)
- [`fit_one_bag.md`](fit_one_bag.md)

## Called by

- [`../classifier/fit_trainer.md`](../classifier/fit_trainer.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
