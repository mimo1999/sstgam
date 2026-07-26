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
Gaussian radial basis functions:

```text
f_k(value, time) = sum_p sum_q coefficient[k,p,q]
                   * value_basis[p](value) * time_basis[q](time)
```

Second-difference penalties keep neighboring coefficients smooth along both
the value and time axes. The model is fitted by cyclic Newton boosting with
early stopping on validation log loss.

## Tree backend

The tree backend bins each feature's observed values using pooled training
quantiles. It stores a value-bin by time-bin table per feature. At each
boosting round, a shallow regression tree smooths a Newton update over that
table's value and time coordinates.

## Interpretation

Spline models expose `value_effect` and `temporal_surface`. Tree models expose
`value_time_table`. Each output is on the log-odds scale; positive values
increase the predicted probability relative to the intercept and negative
values decrease it.
