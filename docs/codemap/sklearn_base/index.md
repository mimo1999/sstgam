# `_base.py` — class `_SSTGAMBase`

The shared sklearn-facing interface. Parses wide longitudinal input (columns
named `<feature>_<time_id>` + plain static columns) into the dense tensors
both backends train on, and implements the standard `fit`/`predict`/
`predict_proba` surface once so `classifier.SSTGAM` only has to supply
`_fit_trainer`.

Module constant: `_TEMPORAL_COLUMN_PATTERN = re.compile(r"^(.+)_(\d+)$")`.

| Method                                                     | Purpose                                                                  |
| ---------------------------------------------------------- | ------------------------------------------------------------------------ |
| [`_parse_columns`](parse_columns.md)                       | Split column names into temporal-feature groups vs. static columns       |
| [`_as_float_matrix`](as_float_matrix.md)                   | Normalize DataFrame/array input to a 2-D float32 array                   |
| [`_prepare_fit_input`](prepare_fit_input.md)               | Resolve training matrix + column names                                   |
| [`_prepare_predict_input`](prepare_predict_input.md)       | Realign predict-time columns to the fitted order                         |
| [`_build_tensors`](build_tensors.md)                       | Reshape wide matrix → `(sample, feature, time)` tensors                  |
| [`fit`](fit.md)                                            | Public `fit()` — validates, parses, tensorizes, delegates                |
| `_fit_trainer`                                             | Abstract hook — see [`fit_trainer_abstract.md`](fit_trainer_abstract.md) |
| [`_positive_probability`](positive_probability.md)         | Shared prediction path (rebuild tensors, ask backend)                    |
| [`predict_proba`](predict_proba.md)                        | sklearn `[P(y=0), P(y=1)]`                                               |
| [`predict`](predict.md)                                    | Hard 0/1 label at threshold 0.5                                          |
| [`predict_log_proba`](predict_log_proba.md)                | Log-probabilities                                                        |
| [`get_feature_names`](get_feature_names.md)                | All fitted input column names                                            |
| [`get_temporal_features`](get_temporal_features.md)        | Just the temporal feature names                                          |
| [`_normalise_feature_values`](normalise_feature_values.md) | Map raw values into the fitted normalized coordinate (spline only)       |
| [`_normalise_time`](normalise_time.md)                     | Map raw time coordinates into `[0,1]` (spline only)                      |

**Subclassed by:** [`../classifier/index.md`](../classifier/index.md) (`SSTGAM`).

**Back to:** [../INDEX.md](../INDEX.md)
