# `SharedShapeTemporalGAMBoost._build_designs`

**File:** `src/sstgam/boosting.py:80-110`
**Kind:** instance method

## Purpose

Turns normalized value/time tensors into the RBF design matrices actually
multiplied against coefficients — one call builds both the temporal design
(per feature) and the static design (per static column).

## Walkthrough

- `L81-83`: fill NaNs with `0.0` (irrelevant either way since those cells get
  masked by `observed`); read `n_samples, n_features`.
- `L85-86`: re-center/scale the values and clip to `[-clip, clip]`. (For the
  quantile transform this re-centering is a no-op since center=0, scale=1;
  for zscore it's the actual normalization.)
- `L88-97`: **per feature**: evaluate the value RBF basis at each
  `(sample, time)` cell ([`../bases/rbf_np.md`](../bases/rbf_np.md)), zero out
  cells that weren't observed, then contract with the fixed time basis
  (`np.einsum("ndp,dq->npq")`) to get one length-`(n_value_bases*n_time_bases)`
  design row per sample per feature — this *is* the tensor-product basis for
  `f_k(value, time)` from the model equation in
  [`ARCHITECTURE.md`](../../../ARCHITECTURE.md).
- `L99-100`: no static columns → return `(temporal_design, None)`.
- `L101-109`: static columns — fill missing with the fitted mean, normalize +
  clip, evaluate an RBF basis per static column.

## Calls

- [`../bases/rbf_np.md`](../bases/rbf_np.md)

## Called by

- [`fit.md`](fit.md) — on training data
- [`predict_proba.md`](predict_proba.md) — on new data

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
