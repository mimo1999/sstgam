# `_training.py`

Small helpers shared by both boosting backends — nothing here is specific to
splines or trees.

| Function | Purpose |
| --- | --- |
| [`make_validation_split`](make_validation_split.md) | Fixed train/validation row split, stratified if possible |
| [`weighted_log_loss`](weighted_log_loss.md) | Weighted binary cross-entropy, used for early stopping |

**Used by:** [../boosting/fit.md](../boosting/fit.md), [../boosting/fit_one_bag.md](../boosting/fit_one_bag.md), [../tree_boost/fit.md](../tree_boost/fit.md), [../tree_boost/fit_one_bag.md](../tree_boost/fit_one_bag.md).

**Back to:** [../INDEX.md](../INDEX.md)
