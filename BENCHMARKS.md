# Benchmarks

Comparative results (SSTGAM vs. DL-GAM, EBM, and CatBoost) are produced in a
**separate, external repository**, evaluated on a separate public dataset
that carries no licensing restriction. No benchmark implementation code,
baseline models, or dataset-specific pipeline lives in this repository --
`sstgam` is a standalone modeling package, not a benchmark harness.

This file is a placeholder for those results. Once available, paste the
external repo's output table below in this schema:

| model | dataset | fold | auc_roc | auc_prc | brier_score |
| --- | --- | --- | --- | --- | --- |
| sstgam (spline) | | | | | |
| sstgam (tree) | | | | | |
| dl-gam | | | | | |
| ebm | | | | | |
| catboost | | | | | |

- `model`: the fitted model name, including which SSTGAM backend (`spline`
  or `tree`) was used.
- `dataset`: the external benchmark dataset's name/identifier.
- `fold`: cross-validation fold index, or `mean`/`std` summary rows across
  folds.
- `auc_roc`, `auc_prc`, `brier_score`: standard discrimination and
  calibration metrics, computed identically across all models being
  compared.

## Preliminary external results: IEEE-CIS fraud detection

First pass from the external `sstgam_testbed` repo, on a public (unlicensed)
dataset -- the IEEE-CIS Fraud Detection data. Not the chemo cohort (licensed,
never used for benchmarking here). Transactions were reshaped into a wide
entity-relative-time panel (K=5 preceding transactions per entity as
temporal history; entity proxy = `card1+card2+card3+card5+addr1+addr2`);
every model received the identical flattened feature set. Single time-based
train/val/test split (no k-fold yet), test set = 82,132 transactions, ~3.5%
fraud rate.

| model | dataset | fold | auc_roc | auc_prc | brier_score | fit_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | ieee_cis_fraud_wide_k5 | time_holdout | 0.852 | 0.431 | 0.105 | 26.1 |
| random_forest | ieee_cis_fraud_wide_k5 | time_holdout | 0.840 | 0.383 | 0.069 | 84.8 |
| ebm (interactions=0) | ieee_cis_fraud_wide_k5 | time_holdout | 0.823 | 0.380 | 0.132 | 340.7 |
| sstgam (tree, tuned) | ieee_cis_fraud_wide_k5 | time_holdout | 0.812 | 0.374 | **0.027** | 157.0 |
| sstgam (tree, tuned + sample_weight) | ieee_cis_fraud_wide_k5 | time_holdout | 0.816 | 0.329 | 0.163 | 148.8 |
| sstgam (tree, tuned, K=15) | ieee_cis_fraud_wide_k15 | time_holdout | 0.813 | 0.371 | 0.027 | 259.9 |
| sstgam (spline, rbf basis, first-guess) | ieee_cis_fraud_wide_k5 | time_holdout | 0.770 | 0.215 | 0.031 | **83.6** |
| sstgam (spline, bspline basis, first-guess) | ieee_cis_fraud_wide_k5 | time_holdout | 0.766 | 0.213 | 0.031 | 91.0 |
| sstgam (spline, rbf, n_value_bases=10/n_time_bases=5) | ieee_cis_fraud_wide_k5 | time_holdout | 0.768 | 0.211 | 0.031 | 192.9 |
| gam (pygam, 40K-row subsample) | ieee_cis_fraud_wide_k5 | time_holdout | 0.749 | 0.195 | 0.031 | 37.8 |

`sstgam (tree)` hyperparameters were chosen by `scripts/tune_sstgam.py`, a
7-candidate search on a training subsample scored against the validation
split (AUC-PRC primary, Brier secondary, fit time recorded): `learning_rate
=0.1, max_rounds=100, early_stopping_rounds=20, n_value_bins=16,
tree_depth=2, outer_bags=1`. This *replaced* the earlier untuned baseline
(`outer_bags=3, max_rounds=150, n_value_bins=16, tree_depth=3,
learning_rate=0.05`), which was both the slowest and the worst-performing
config in the search on validation data -- tuning bought a 3.9x speedup
(616.7s -> 157.0s) *and* a small quality improvement (AUC-PRC 0.365 ->
0.374), not a speed/quality tradeoff. `outer_bags=1` and `tree_depth=2` both
outperformed their more expensive alternatives (`outer_bags>1`,
`tree_depth=3`) on this dataset -- more bagging/depth wasn't buying
anything here.

`sstgam (spline)` is a first-guess, least-compute configuration, not tuned
like the tree backend: `n_value_bases=4, n_time_bases=3, n_static_bases=4,
learning_rate=0.1, max_rounds=60, early_stopping_rounds=15, outer_bags=1`.
Each temporal feature's tensor-product surface costs an
`n_value_bases * n_time_bases` square matrix inversion per bag, so this
config was chosen purely to be cheap, not to be competitive.

**Sample-weight experiment, and why it didn't do what was expected**: to test
whether SSTGAM (tree)'s calibration advantage over XGBoost/EBM was just an
artifact of it being the only model *not* given imbalance-correction weights,
it was refit with the same `n_negative/n_positive` `sample_weight` on fraud
rows XGBoost and EBM got, keeping the tuned hyperparameters unchanged. Result:
**both metrics got worse**, not a calibration-for-ranking tradeoff -- AUC-PRC
dropped (0.374 -> 0.329) and Brier score became the *worst* of every model
tested (0.027 -> 0.163, worse than EBM's 0.132). Two likely reasons, not yet
disentangled:
1. The tuned hyperparameters (`n_value_bins=16`, `outer_bags=1`) were chosen
   by a search that never used sample weighting -- reweighting on top of
   hyperparameters tuned for the unweighted objective is an unfair test of
   weighting itself. A proper comparison would re-tune with weighting
   enabled from the start.
2. The tree backend's value-time table has limited capacity (16 value bins x
   5 time slots per feature, `outer_bags=1` for no bagged averaging) --
   heavily upweighting the ~3.5% fraud rows may let a small number of
   examples dominate individual cells' Newton updates, producing noisier
   per-cell log-odds than a large ensemble (XGBoost) or a bagged, smoother
   fit (RF) can absorb. Early stopping also selects the checkpoint by
   *weighted* validation loss, which optimizes a different objective than
   the unweighted test metrics reported here -- a real mismatch worth fixing
   before drawing conclusions from this specific number.

This means the calibration-vs-ranking story for SSTGAM (tree) in the earlier
notes below is **not fully explained** by the imbalance-weighting asymmetry
alone -- something about how this specific tuned config responds to
weighting needs more investigation (e.g. re-running `tune_sstgam.py`'s
search *with* weighting enabled) before treating the weighted result as a
fair comparison point.

**K=15 diagnostic (does more temporal history help?)**: a hypothesis was
raised that SSTGAM's value-time modeling might not be showing its real
advantage because K=5 preceding transactions is too short a history to
reveal genuine temporal dynamics. Tested by rebuilding the panel with K=15
(120 temporal columns instead of 40, `scripts/prepare_data.py --k 15`,
preserved separately in `data/prepared_k15/` alongside the original K=5
data) and refitting SSTGAM (tree) with the same tuned hyperparameters.
Result: **no improvement** -- AUC-PRC 0.371 vs K=5's 0.374 (within noise,
if anything marginally worse), for 66% more fit time (157.0s -> 259.9s,
tripling the feature count roughly doubles cost). Small-subsample checks
during this run showed AUC-PRC climbing with subsample size (0.352 at 20K
rows, 0.371 at 60K rows) -- that trend turned out to be the ordinary effect
of more training data, not K=15 unlocking temporal signal, since it
converged to the same ceiling K=5 already reaches at full scale. This is
evidence *against* "insufficient history depth" as the explanation for
SSTGAM's ranking gap on this dataset: the IEEE-CIS temporal features here
(transaction-count aggregates, timedeltas) most likely don't carry the kind
of smoothly-evolving, value-dependent-over-time structure SSTGAM's shared
value-time surface is built to exploit -- more lag columns just added
redundant history, not new information. This is a reason to prefer testing
SSTGAM on a more genuinely longitudinal dataset (e.g. an ICU vitals/labs
cohort) over further tuning on this one specifically.

Notes:
- SSTGAM (tree, unweighted) is essentially tied with EBM on ranking (AUC-PRC
  0.374 vs 0.380) at under half EBM's fit time, and keeps the best
  calibration of all models by a wide margin -- 5x lower Brier score than
  EBM, the next GAM-family model.
- SSTGAM (spline) at this untuned, minimal-basis config trails SSTGAM (tree)
  by a wide margin (AUC-PRC ~0.21 vs 0.374) and only edges out plain pygam
  GAM -- reads as "too coarse a basis for this data's value-time structure,"
  not "the spline backend doesn't work here." It's the cheapest SSTGAM
  variant tested (83.6s, faster than RF) and hasn't been through a tuning
  pass the way the tree backend was; expect this gap to close somewhat with
  more bases/rounds, at proportional extra cost.
- Basis family (`"rbf"` vs `"bspline"`) made almost no difference at this
  basis size (AUC-PRC 0.215 vs 0.213, Brier essentially tied) -- `bspline`'s
  Cox-de Boor recursion cost ~9% more time (91.0s vs 83.6s) for no quality
  gain here.
- Bumping the RBF basis size from `n_value_bases=4, n_time_bases=3` (12
  coefficients/feature) to `n_value_bases=10, n_time_bases=5` (50
  coefficients/feature) more than doubled fit time (83.6s -> 192.9s) for
  *no* quality gain -- AUC-PRC went slightly down (0.215 -> 0.211), within
  noise. So basis coarseness was the wrong diagnosis for the spline
  backend's gap to the tree backend on this dataset; the other first-guess
  constraints (`learning_rate=0.1, max_rounds=60, outer_bags=1`) or a
  genuine better fit of the tree backend's discrete value-time table to this
  data's structure are more likely explanations, worth checking before
  spending more compute on basis size specifically.
- XGBoost's and EBM's `scale_pos_weight`/`sample_weight` imbalance correction
  helps their ranking metrics but visibly hurts calibration (their Brier
  scores are the two worst) -- a reminder not to compare Brier scores
  across models using different imbalance-handling strategies at face value.
- pygam was subsampled to 40K training rows (documented, not silent) since
  its dense P-IRLS solve doesn't scale to the full ~383K-row training set.
- This is one split, one hyperparameter setting per non-SSTGAM model (only
  SSTGAM went through an explicit tuning pass), not a fully tuned or
  cross-validated comparison -- treat as directional, not final.
