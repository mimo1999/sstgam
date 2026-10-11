# SSTGAM architecture

SSTGAM is an additive binary classifier for wide longitudinal data. It keeps a
single effect surface for each temporal feature and evaluates that surface at
every observed time point.

## Input representation

Columns named `<feature_name>_<time_id>` become a dense tensor:

```text
values[sample, feature, time]
observed[sample, feature, time]
```

The observation mask is one for finite input values and zero for missing
values. Columns without a trailing integer are static covariates.

## Statistical model

```text
logit(p_i) = intercept
           + sum_k sum_d observed[i,k,d] * f_k(values[i,k,d], time_d)
           + sum_s g_s(static[i,s])
```

The model is additive across features and time points. It models a
time-varying effect of each individual measurement, not relationships between
measurements such as a lag, difference, or slope.

## Spline backend

The spline backend represents each temporal effect as a tensor product of
fixed basis functions:

```text
f_k(value, time) = sum_p sum_q coefficient[k,p,q]
                   * value_basis[p](value) * time_basis[q](time)
```

`value_basis` and `time_basis` are drawn from one of two interchangeable
families, selected by `basis`: fixed-bandwidth Gaussian radial basis
functions (`"rbf"`, default) or a clamped, evenly knotted B-spline basis
(`"bspline"`). Both are implemented in `bases.py` behind the same
`make_basis_params`/`basis_design` interface, so the rest of the boosting
code is basis-agnostic. Second-difference penalties on adjacent coefficients
keep the surface smooth along both the value and time axes regardless of
basis choice.

The model is fitted by cyclic (coordinate-descent) gradient boosting with
early stopping on validation log loss. Each coordinate-block update
(`_update_block`) uses a fixed quasi-Newton step, not literal Newton's
method: `_preconditioner` inverts a penalized *worst-case* Hessian bound
(evaluated at `p = 0.5`) once per bag, and reuses that same inverse for every
boosting round even as the true Hessian `p * (1 - p)` shrinks as the fit
improves. This is a majorization-minimization / IRLS-style bound — a common
and reasonable approximation, but distinct from recomputing the Hessian each
round.

## Tree backend

The tree backend bins each feature's observed values using pooled training
quantiles. It stores a value-bin by time-bin table per feature. At each
boosting round, a shallow regression tree smooths a Newton update over that
table's value and time coordinates.

The split order is chosen per round rather than left to the tree. Depth-1
candidate splits are fitted on the value axis and the time axis separately and
scored by weighted-SSE gain minus `2 * log(m)`, where `m` is the number of
distinct candidate thresholds on that axis. The higher-scoring axis splits
first; the other axis refines within each resulting group. This penalty is a
candidate-count penalty on greedy SSE gain — not a formal Bonferroni or BIC
correction, since neither a p-value nor a likelihood term is computed here —
motivated by the same intuition: an axis with more candidate thresholds gets
more chances for a greedy split to look good by chance alone. Without the
penalty the value axis wins on candidate count alone, since it normally has
more quantile bins than there are observation times.

Each tree leaf's update value is a true per-round Newton step
(`gradient_sum / hessian_sum`, recomputed from the current fit every round),
unlike the spline backend's fixed quasi-Newton preconditioner above.

Static terms have a single axis and use a plain one-stage tree.

The table is indexed by ordinal time position, but the trees split on the
normalized time coordinate, so irregular spacing between observation times is
respected when grouping time points during fitting.

## Bagging

Both backends fit `outer_bags` times on bootstrap resamples of the training
rows and average the fitted parameters. The validation split is held fixed
across bags, so early stopping is judged on the same held-out rows every time
and only the resampled training rows differ.

Averaging does not change the model form. A mean of coefficient vectors over a
shared basis is another coefficient vector, and a mean of value-time tables is
another table, so the fitted `f_k` stays exactly as defined above. Bagging is a
variance-reduction step only. Set `outer_bags=1` to disable it.

## Interpretation

Spline models expose `value_effect` and `temporal_surface`. Tree models expose
`value_time_table`. Each output is on the log-odds scale; positive values
increase the predicted probability relative to the intercept and negative
values decrease it.

## Related work

SSTGAM is a generalized additive model (GAM) in the classical sense
(Hastie & Tibshirani) — a sum of one-dimensional/low-dimensional shape
functions on a logit link — specialized to wide longitudinal panels by giving
each temporal feature a single shared value-*time* surface instead of a
value-only shape function. The cyclic boosting/bagging fitting procedure
follows the Explainable Boosting Machine (EBM) design: shallow, per-feature
boosting rounds cycled across features, with bagged bootstrap fits averaged
for variance reduction (`ebm_paper.pdf`). The pairwise (feature, feature)
interaction idea in GA2M (`ga2m_paper_existing.pdf`) is the closest prior
work to treating (value, time) as a joint two-dimensional interaction term
rather than two separate additive terms — SSTGAM applies that same
2-D-interaction treatment to a feature's own value against its own
observation time, rather than to two distinct features. Dynamic GAMs (DGAM,
`dgam_paper.pdf`) are the closest prior work on modeling additive effects
that vary over time in a discrete time series/forecasting setting; SSTGAM
differs by fitting one shared surface per feature across all sparsely and
irregularly observed time points at once, rather than a separate smooth per
forecast horizon.

## Assumptions and limitations

- **Missingness is treated as uninformative at the cell level.** An
  unobserved `(feature, time)` cell contributes exactly zero to the score
  (see `observed` in the statistical model above) — equivalent to assuming
  missingness is not, by itself, evidence about the outcome (a cell-level
  MAR assumption). This is a real modeling choice, not a neutral default: in
  settings where measurement frequency itself is informative (e.g. sicker
  patients are tested more often, or stop being tested because they were
  discharged or died), this assumption is questionable. See `README.md`'s
  `include_missingness_indicators` option for an opt-in, domain-agnostic way
  to let the model use missingness patterns as evidence, without asserting
  anything about *which* times or features that applies to.
- **No time-position-specific inductive bias is built in.** SSTGAM does not
  assume that later (or earlier) observations are structurally more
  predictive — every time point is treated symmetrically by the model
  class, with all differential weighting learned from data through the
  value-time surface itself. This is a deliberate design boundary: an
  experimental variant that hard-codes a "later observations matter more"
  day-weighting scheme (found in an unapplied patch, `boost.diff`, against
  an earlier and since-abandoned prototype of this package) was evaluated
  and rejected for this reason — that kind of assumption is a property of
  one dataset's temporal structure, not of longitudinal data in general, and
  baking it into the model would compromise the "agnostic baseline plus
  opt-in knobs" design this package otherwise follows (e.g. `basis`,
  `value_transform`).
