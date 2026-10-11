# `SharedShapeTemporalGAMTreeBoost.fit`

**File:** `src/sstgam/tree_boost.py:193-259`
**Kind:** instance method

## Purpose

Top-level tree-backend fit — mirrors `boosting.fit`'s structure: fix bins and
coordinates once, then run `outer_bags` bagged fits and average the
resulting tables.

## Walkthrough

- `L205-207`: [`_fit_bins`](fit_bins.md), then bin every temporal feature
  ([`_feature_bins`](feature_bins.md)) and every static column
  ([`_static_bins`](static_bins.md)) once on the full training tensor.
- `L211-213`: normalize `time_coordinates` to `[0,1]`.
- `L214-220`: `temporal_coordinates[feature]` — a
  `(bin_count * n_times, 2)` array enumerating every `(value_bin,
  normalized_time)` cell for that feature; these are the fitting coordinates
  [`adaptive_tree_update`](adaptive_tree_update.md) splits on.
- `L221-224`: `temporal_cell_ids[feature]` — maps each `(sample, time)`
  position to its flat `value_bin * n_times + time_index` cell id, used both
  to scatter gradients ([`update_temporal_term.md`](update_temporal_term.md)) and to
  gather table values back out at predict time.
- `L225-226`: `static_coordinates` — just `[0, 1, 2, ...]` per static column
  (1-D, no time axis).
- `L229-231`: [`../training/make_validation_split.md`](../training/make_validation_split.md) —
  fixed across bags, same as the spline backend.
- `L233-258`: loop `outer_bags` times: bootstrap-resample training rows, call
  [`fit_one_bag.md`](fit_one_bag.md), accumulate intercept and per-feature/
  per-static tables; divide by `outer_bags` at the end.

## Calls

- [`fit_bins.md`](fit_bins.md)
- [`feature_bins.md`](feature_bins.md)
- [`static_bins.md`](static_bins.md)
- [`../training/make_validation_split.md`](../training/make_validation_split.md)
- [`fit_one_bag.md`](fit_one_bag.md)

## Called by

- [`../classifier/fit_trainer.md`](../classifier/fit_trainer.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
