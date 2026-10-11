# `_quantile_edges`

**File:** `src/sstgam/tree_boost.py:19-26`
**Kind:** module-level function

## Purpose

Builds deduplicated quantile bin edges for one feature's pooled observed
values, guaranteeing at least one usable bin even for degenerate inputs.

## Walkthrough

- `L21-22`: empty input → trivial `[0.0, 1.0]` edges.
- `L23`: `np.unique(np.quantile(values, linspace(0,1,n_bins+1)))` — dedup
  handles ties or low-cardinality features that would otherwise produce
  fewer effective bins than `n_bins`.
- `L24-25`: if dedup collapsed everything to a single edge (a constant
  feature), widen it to a 1-unit interval around that value so there's still
  one usable bin.

## Calls

None.

## Called by

- [`fit_bins.md`](fit_bins.md) — once per temporal feature, once per static column

**Back to:** [index.md](index.md) · [../INDEX.md](../INDEX.md)
