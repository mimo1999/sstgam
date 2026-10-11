# `SharedShapeTemporalGAMBoost._fit_one_bag`

**File:** `src/sstgam/boosting.py:195-258`
**Kind:** instance method

## Purpose

One full cyclic-Newton-boosting run, with early stopping, on one bag's
bootstrap-resampled training rows, evaluated against the fixed validation
split.

## Walkthrough

- `L198-203`: slice the precomputed designs/target/weight down to this bag's
  fit rows (`fit_indices`) and the fixed validation rows.
- `L205-212`: build one [`_preconditioner`](preconditioner.md) per feature,
  and (if static columns exist) one per static column — fixed for the whole
  bag, not recomputed per round.
- `L214-219`: `intercept` initialized at the log-odds of the (weighted) base
  rate; coefficient arrays zeroed; `fit_score`/`validation_score` initialized
  at that intercept for every row.
- `L225-254`: boosting loop, up to `max_rounds`:
  - `L226-231`: Newton-update the intercept alone first (plain 1-D Newton
    step, no design matrix involved).
  - `L233-241`: run [`_update_block`](update_block.md) over the temporal
    block, then (if present) the static block.
  - `L243-244`: [`../training/weighted_log_loss.md`](../training/weighted_log_loss.md)
    on the validation rows.
  - `L245-250`: if this is the best validation loss so far, snapshot
    `(intercept, temporal_coefficients, static_coefficients)` as `best_state`
    and reset the no-improvement counter.
  - `L251-254`: otherwise increment the counter; break once it reaches
    `early_stopping_rounds`.
- `L256-258`: fall back to the final (unsnapshotted) state if validation loss
  never improved even once; return `(*best_state, best_round)`.

## Calls

- [`preconditioner.md`](preconditioner.md)
- [`update_block.md`](update_block.md)
- [`../training/weighted_log_loss.md`](../training/weighted_log_loss.md)

## Called by

- [`fit.md`](fit.md) — once per outer bag

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
