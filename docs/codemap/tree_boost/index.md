# `tree_boost.py` — class `SharedShapeTemporalGAMTreeBoost`

The tree backend (`method="tree"`): cyclic boosted GAM with one value-bin by
time-bin table per feature, updated via shallow regression trees fit to
aggregated Newton statistics. Values are binned per feature from pooled
training quantiles. Bagged the same way as the spline backend.

Module constants: `_MIN_HESSIAN = 1e-6`, `_AXIS_PENALTY_LAMBDA = 2.0`
(per-log-candidate-count penalty correcting CART's bias toward axes with more
candidate thresholds), `_FIRST_AXIS_DEPTH = 2` (depth budget for whichever
axis splits first each round).

| Function/Method | Purpose |
| --- | --- |
| [`_quantile_edges`](quantile_edges.md) | Module-level: deduplicated quantile bin edges for one feature |
| [`__init__`](init.md) | Store tree-backend hyperparameters |
| [`_bin`](bin.md) | Map continuous values to integer bin indices |
| [`_fit_bins`](fit_bins.md) | Fit quantile bin edges per feature and per static column |
| [`_feature_bins`](feature_bins.md) | Bin every temporal feature's value tensor |
| [`_static_bins`](static_bins.md) | Bin every static column (with median imputation) |
| [`_tree_update`](tree_update.md) | Plain single-stage tree update (static, 1-D terms) |
| [`_weighted_gain`](weighted_gain.md) | Weighted-SSE gain of a candidate prediction |
| [`_adaptive_tree_update`](adaptive_tree_update.md) | Joint (value, time) update with adaptive split-order choice |
| [`_update_temporal_term`](update_temporal_term.md) | Aggregate gradient/Hessian into cells, call adaptive update |
| [`_update_static_term`](update_static_term.md) | Aggregate gradient/Hessian into bins, call plain tree update |
| [`_grad_hess`](grad_hess.md) | Logistic gradient/Hessian at the current score |
| [`fit`](fit.md) | Top-level fit: fix bins/coordinates once, bag, average |
| [`_fit_one_bag`](fit_one_bag.md) | One bag's boosting loop with early stopping |
| [`predict_proba`](predict_proba.md) | Score new data by binning + table lookup |
| [`value_time_table`](value_time_table.md) | Public API: one feature's value×time table |
| [`static_effect_table`](static_effect_table.md) | Public API: one static covariate's effect table |

**Depends on:** [`../training/index.md`](../training/index.md).
**Built by:** [`../classifier/make_trainer.md`](../classifier/make_trainer.md).
**See also:** the adaptive split-order logic is described in
[`ARCHITECTURE.md`](../../../ARCHITECTURE.md) under "Tree backend".

**Back to:** [../INDEX.md](../INDEX.md)
