# `_SSTGAMBase._parse_columns`

**File:** `src/sstgam/_base.py:27-41`
**Kind:** staticmethod

## Purpose

Splits input column names into temporal-feature groups (matched against
`<feature_name>_<time_id>`) and a list of static column indices. Raises if the
same `(feature_name, time_id)` pair appears twice.

## Walkthrough

- `L32-35`: for each column, try `_TEMPORAL_COLUMN_PATTERN.match`; no match →
  append to `static_columns`.
- `L36-39`: match → extract `feature_name` and integer `time_id`; if that
  `time_id` was already recorded for this feature, raise
  `ValueError("Duplicate temporal column...")`.

## Calls

None.

## Called by

- [`fit.md`](fit.md) — parses `X`'s column names once at fit time; the resulting
  `temporal_columns`/`static_columns` become `self._temporal_columns` /
  `self._static_column_indices`, reused later by [`build_tensors.md`](build_tensors.md)
  both at fit and predict time.

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
