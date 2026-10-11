# `SharedShapeTemporalGAMBoost._update_block`

**File:** `src/sstgam/boosting.py:112-122`
**Kind:** staticmethod

## Purpose

One cyclic-coordinate Newton step over a whole block of units (e.g. every
temporal feature, or every static column) — sequentially, one unit at a time,
each seeing the residual already left by the *previous* unit's update in the
same round. This is the core mechanic behind "cyclic Newton boosting."

## Walkthrough

- `L116-121`: for each `unit_index` in the block:
  - recompute `probability = sigmoid(train_score)` from the **current**
    running score (already updated by prior units this round);
  - form the weighted residual `residual = sample_weight * (target - probability)`;
  - the preconditioned Newton step is
    `coefficient_update = preconditioner @ (train_design[:, unit_index, :].T @ residual)`;
  - scale by `learning_rate`, add to that unit's coefficients;
  - add the corresponding score contribution to **both** `train_score` and
    `validation_score`, in place (mutating the caller's arrays by reference).

This mutate-in-place pattern is exactly what implements the "cyclic" part:
each unit's update is a function of the residual left by everything processed
earlier in the same round, not a joint solve over all units at once. See
[`ARCHITECTURE.md`](../../../ARCHITECTURE.md) for the model-level framing.

## Calls

None.

## Called by

- [`fit_one_bag.md`](fit_one_bag.md) — twice per round: once for the temporal
  block, once for the static block (if any static columns exist)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
