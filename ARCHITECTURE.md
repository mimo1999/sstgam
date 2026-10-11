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
`make_basis_params`/`basis_design` interface, so the rest of the fitting
code is basis-agnostic. Second-difference penalties on adjacent coefficients
keep the surface smooth along both the value and time axes regardless of
basis choice.

The fit maximises the penalized log-likelihood

```text
sum_i w_i loglik_i  -  1/2 sum_k c_k' P_k c_k  -  1/2 sum_s b_s' P_s b_s
```

directly, by P-IRLS (`_penalized_newton`), the standard way to fit a
penalized GAM: all terms share one design matrix (intercept, every surface
block, every static block), and each iteration takes a full Newton step with
gradient `X' w (y - p) - P beta` and Hessian `X' diag(w p (1 - p)) X + P`,
halving the step if it would not improve the objective. The model has few
coefficients (161 on the 8-feature IEEE benchmark), so each iteration is one
small dense solve, and it converges in under 20 iterations. The Hessian is
accumulated over row chunks to bound memory.

### Choosing the penalty

Because the fit is the exact penalized optimum, the penalty alone sets the
smoothness, and its right value varies by orders of magnitude between
datasets: roughly with the number of coefficients per minority-class
example: 0.01-0.1 on the 14k-event fraud benchmark with ~350 coefficients,
but in the thousands for a small, wide dataset with a few hundred events and
thousands of coefficients. No fixed default
serves both, so `temporal_penalty`/`static_penalty` default to `"auto"`
(`_select_penalty`): one shared penalty is chosen by held-out weighted log
loss and the model is refit on all rows. The search starts at
`100 x coefficients / minority examples`, brackets it by `sqrt(10)`, and
steps outward by 10x only while the edge value beats its neighbour by more
than one standard error of their paired per-row loss difference; it then
takes the largest penalty within one standard error of the best. Fits warm
start from the nearest penalty already tried.

The held-out loss comes from one stratified `val_fraction` split when that
split holds at least 500 minority examples. Below that, a single split is
too noisy to rank penalties, so the search uses 5-fold cross-validation and
compares out-of-fold losses over every row, at roughly 5x the search cost.
Two alternatives were measured and rejected: a fixed relative tolerance
(cannot be right for both large and small validation sets) and AIC on all
rows (with far more coefficients than events the model nearly separates the
classes, and AIC drove the penalty to its lower bound).

This replaced cyclic boosting, which updated one block at a time over
hundreds of rounds: same optimum, but it converged slowly when blocks were
correlated (13 minutes on the benchmark vs 9 seconds). Because the fit is
the exact penalized optimum, `temporal_penalty` and `static_penalty` are the
only smoothness controls; there are no boosting rounds, learning rate or
validation split, so the spline method trains on all rows.

With `value_transform="quantile"`, static covariates get the same empirical
CDF transform as temporal values before their basis is applied. Heavy-tailed
amounts, counts and integer codes otherwise bunch onto a few basis functions
under a z-score; on the benchmark this transform alone moved test AUC-ROC
from 0.785 to 0.809 at the same basis size.

### Centring (identifiability)

A constant added to a surface can be traded against the intercept, so an
unconstrained surface's level is arbitrary. Each surface is therefore
constrained to average zero over the observed training cells, and each static
effect to average zero over the training rows — the same sum-to-zero
identifiability constraint mgcv uses. It is imposed exactly by fitting in the
null space of the term's mean basis row (`_centering_basis`): coefficients are
`c = Z theta` with `Z` an orthonormal basis of `{c : mean_row @ c = 0}`.
Averages of centred coefficient vectors are centred, so bagging preserves it.

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
like a Newton step of the spline backend. Unlike the spline backend, the
tree's shrinkage (`learning_rate`, default 0.02) and the stopping round act as
regularizers.

Static terms have a single axis and use a plain one-stage tree.

The table is indexed by ordinal time position, but the trees split on the
normalized time coordinate, so irregular spacing between observation times is
respected when grouping time points during fitting.

## Bagging

Both backends fit `outer_bags` times on bootstrap resamples of the training
rows and average the fitted parameters. For the tree backend the validation
split is held fixed across bags, so early stopping is judged on the same
held-out rows every time and only the resampled training rows differ. The
spline backend has no validation split and resamples all rows.

Averaging does not change the model form. A mean of coefficient vectors over a
shared basis is another coefficient vector, and a mean of value-time tables is
another table, so the fitted `f_k` stays exactly as defined above. Bagging is a
variance-reduction step only. Set `outer_bags=1` to disable it.

## Interpretation

Spline models expose `value_effect` and `temporal_surface`. Tree models expose
`value_time_table`. Both expose `static_effect` for static covariates and
missingness indicators. Each output is on the log-odds scale. Spline effects
are centred (see above), so a positive value means the observation raises the
log-odds relative to an average observed cell of that feature (or an average
row, for a static effect), and a negative value lowers it.

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
  (see `observed` in the statistical model above). For the spline method,
  centring makes zero the contribution of an average observed cell, so a
  missing cell is scored as an average observation — equivalent to assuming
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
