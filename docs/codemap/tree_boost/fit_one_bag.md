# `SharedShapeTemporalGAMTreeBoost._fit_one_bag`

**File:** `src/sstgam/tree_boost.py:261-335`
**Kind:** instance method

## Purpose

One bagged run's cyclic boosting loop with early stopping — the tree-backend
counterpart of `boosting._fit_one_bag`, using tree-based table updates
instead of a closed-form Newton step per basis coefficient.

## Walkthrough

- `L265-268`: intercept initialized at the (weighted) base-rate log-odds;
  zero-initialize each feature's and static column's flat table.
- `L278-331` (loop over rounds):
  - `L279-283`: Newton-update the intercept alone using
    [`_grad_hess`](grad_hess.md).
  - `L285-304`: **per temporal feature** — recompute gradient/Hessian against
    the *current* score, call [`update_temporal_term.md`](update_temporal_term.md),
    add the learning-rate-scaled update into that feature's flat table, and
    update both `fit_score`/`validation_score` by gathering the update at
    each sample's `(bin, time)` cell id and summing across observed times.
  - `L306-316`: **per static column** — same pattern via
    [`update_static_term.md`](update_static_term.md).
  - `L318-330`: evaluate [`../training/weighted_log_loss.md`](../training/weighted_log_loss.md)
    on validation; track `best_state`; break after `early_stopping_rounds`
    rounds without improvement.
- Same cyclic pattern as the spline backend's `_fit_one_bag`: each
  feature/column in a round sees residuals already updated by everything
  processed earlier in that same round.

## Calls

- [`grad_hess.md`](grad_hess.md)
- [`update_temporal_term.md`](update_temporal_term.md)
- [`update_static_term.md`](update_static_term.md)
- [`../training/weighted_log_loss.md`](../training/weighted_log_loss.md)

## Called by

- [`fit.md`](fit.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
