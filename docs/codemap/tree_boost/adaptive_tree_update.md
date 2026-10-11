# `SharedShapeTemporalGAMTreeBoost._adaptive_tree_update`

**File:** `src/sstgam/tree_boost.py:118-169`
**Kind:** instance method

## Purpose

Joint `(value_bin, normalized_time)` Newton update for one temporal feature,
choosing **per round** which axis splits first based on a penalized
weighted-SSE gain, rather than letting a single joint tree pick greedily.
This is the split-order logic documented in
[`ARCHITECTURE.md`](../../../ARCHITECTURE.md) under "Tree backend" — the
file's most involved function.

## Walkthrough

- `L124-126`: bail out (`return None`) if fewer than 2 bins have enough
  accumulated Hessian (`active = hessian_sum > _MIN_HESSIAN`).
- `L127-129`: Newton target per bin, guarded against division by zero;
  split `coordinates` into `value_column` and `time_column`.
- `L131-137` (`fit_depth1` closure): fit a **depth-1** tree on one axis alone;
  score it as `weighted_gain - _AXIS_PENALTY_LAMBDA * log(candidate_count)`,
  where `candidate_count` is the number of distinct values on that axis among
  active bins. This log-candidate-count penalty is a Bonferroni/BIC-style
  multiplicity correction — without it the value axis would win purely on
  having more distinct thresholds than there are observation times, since it
  usually has many more quantile bins.
- `L139-140`: fit and score both axes' depth-1 candidates
  (`time_tree1`/`time_score`, `value_tree1`/`value_score`).
- `L142-149`: whichever axis scores higher becomes `first_column`; its depth
  budget is `min(_FIRST_AXIS_DEPTH=2, max_depth-1)`, and the remainder of
  `max_depth` goes to the second (refining) axis.
- `L151-156`: if the first axis's depth budget is exactly 1, reuse the
  depth-1 tree already fit for scoring instead of refitting it; otherwise fit
  a fresh tree at the full first-axis depth.
- `L157-159`: `first_tree.apply(first_column)` assigns every bin (active or
  not) to a leaf group; `first_tree.predict(first_column)` gives the
  baseline prediction for that group.
- `L160-168`: within each leaf group with ≥2 active bins, fit a **second**
  tree restricted to that group's `second_column` and overwrite `update` for
  that group's rows — this is what lets the second axis's actual split
  points differ per first-axis group (the "refines within each resulting
  group" behavior from the README/ARCHITECTURE docs).

## Calls

- [`weighted_gain.md`](weighted_gain.md)
- `sklearn.tree.DecisionTreeRegressor` (external)

## Called by

- [`update_temporal_term.md`](update_temporal_term.md)

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
