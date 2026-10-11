# `_SSTGAMBase._build_tensors`

**File:** `src/sstgam/_base.py:79-99`
**Kind:** instance method

## Purpose

The core reshape from a wide 2-D matrix into the model's `(sample, feature,
time)` tensors plus the static matrix — this is what "wide longitudinal data"
becomes internally, and it's used identically at both fit and predict time.

## Walkthrough

- `L80-85`: allocate `values`/`observed` as zero tensors sized
  `(n_samples, n_features, n_times)`; build a `time_id -> position` lookup
  from `self.time_ids_`.
- `L87-93`: for each temporal feature and each of its time columns, pull that
  column out of `matrix`, mark finite entries as `observed`, and write the
  value (or `0.0` where missing) into the tensor at the right `(feature,
  time)` slot.
- `L95-98`: gather the static columns via `self._static_column_indices`, or an
  empty `(n, 0)` array if there are none.

## Calls

None — pure NumPy indexing.

## Called by

- [`fit.md`](fit.md)
- [`positive_probability.md`](positive_probability.md)

Using the exact same tensorization function for both training and prediction
is what guarantees train/predict consistency (same missing-value handling,
same feature ordering).

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
