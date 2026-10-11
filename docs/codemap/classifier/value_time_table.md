# `SSTGAM.value_time_table`

**File:** `src/sstgam/classifier.py:144-149`
**Kind:** instance method (public API, tree-only)

## Purpose

Returns the tree backend's fitted value-bin by time-bin table for one
feature.

## Walkthrough

- `L146-148`: guard `self.method == "tree"`; `check_is_fitted`.
- `L149`: `self.model_.value_time_table(self.temporal_features_.index(feature))`.

## Calls

- [`../tree_boost/value_time_table.md`](../tree_boost/value_time_table.md)

## Called by

- User code: `model.value_time_table("signal_a")`

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
