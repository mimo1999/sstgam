# `normalization.py`

Normalization helpers for longitudinal and static values — implements the two
`value_transform` strategies (`"quantile"` and `"zscore"`) used only by the
spline backend.

| Function | Purpose |
| --- | --- |
| [`robust_center_scale`](robust_center_scale.md) | Per-feature median center + IQR-based scale (zscore transform) |
| [`fit_quantile_tables`](fit_quantile_tables.md) | Fit empirical-CDF lookup tables (quantile transform) |
| [`apply_quantile_tables`](apply_quantile_tables.md) | Apply fitted CDF tables via interpolation |
| [`fit_value_normalization`](fit_value_normalization.md) | Dispatch to whichever transform is configured, at fit time |
| [`apply_value_normalization`](apply_value_normalization.md) | Dispatch counterpart, at predict time |
| [`static_stats`](static_stats.md) | Per-static-column mean/std |

**Used by:** [../boosting/index.md](../boosting/index.md) only.

**Back to:** [../INDEX.md](../INDEX.md)
