# `boosting.py` — class `SharedShapeTemporalGAMBoost`

The spline backend (`method="spline"`): cyclic Newton boosting over smooth
tensor-product RBF value-time surfaces, bagged across `outer_bags` bootstrap
resamples with a fixed validation split.

Module constants: `_NORMALIZATION_CLIP = 4.0`, `_RIDGE = 1e-6`.

| Method | Purpose |
| --- | --- |
| [`__init__`](init.md) | Store spline-backend hyperparameters |
| [`_penalty_matrix`](penalty_matrix.md) | Roughness penalty for one feature's tensor-product coefficients |
| [`_static_penalty`](static_penalty.md) | Roughness penalty for a 1-D static-effect coefficient vector |
| [`_preconditioner`](preconditioner.md) | Fixed inverse-Hessian-bound, precomputed once per bag |
| [`_build_designs`](build_designs.md) | Build the RBF design matrices from normalized tensors |
| [`_update_block`](update_block.md) | One cyclic-coordinate Newton step over a block of units |
| [`fit`](fit.md) | Top-level fit: setup once, `outer_bags` bagged runs, average |
| [`_fit_one_bag`](fit_one_bag.md) | One bag's boosting loop with early stopping |
| [`predict_proba`](predict_proba.md) | Score new data through the averaged fitted model |
| [`temporal_surface_normalized`](temporal_surface_normalized.md) | Evaluate one feature's fitted surface on a grid |
| [`mean_value_effect_normalized`](mean_value_effect_normalized.md) | Value-only effect (surface averaged over time) |

**Depends on:** [`../training/index.md`](../training/index.md), [`../bases/index.md`](../bases/index.md), [`../normalization/index.md`](../normalization/index.md).
**Built by:** [`../classifier/make_trainer.md`](../classifier/make_trainer.md).

**Back to:** [../INDEX.md](../INDEX.md)
