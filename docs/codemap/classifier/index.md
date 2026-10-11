# `classifier.py` — class `SSTGAM(_SSTGAMBase)`

The public entry point. Thin wrapper whose main job is choosing and building
the right backend (`SharedShapeTemporalGAMBoost` for splines,
`SharedShapeTemporalGAMTreeBoost` for trees) and exposing a few
backend-specific inspection methods.

| Method | Purpose |
| --- | --- |
| [`__init__`](init.md) | Validate + store every hyperparameter for both backends |
| [`_fit_trainer`](fit_trainer.md) | Concrete override of the abstract hook — builds a trainer and fits it |
| [`_make_trainer`](make_trainer.md) | Constructs the right backend object from `self.method` |
| [`value_effect`](value_effect.md) | Public API: value-only spline effect, averaged over time |
| [`temporal_surface`](temporal_surface.md) | Public API: full value×time spline surface |
| [`value_time_table`](value_time_table.md) | Public API: tree backend's discrete value×time table |

**Subclasses:** [`../sklearn_base/index.md`](../sklearn_base/index.md) (`_SSTGAMBase`).
**Delegates to:** [`../boosting/index.md`](../boosting/index.md), [`../tree_boost/index.md`](../tree_boost/index.md).

**Back to:** [../INDEX.md](../INDEX.md)
