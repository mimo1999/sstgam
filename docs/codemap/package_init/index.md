# `__init__.py`

**File:** `src/sstgam/__init__.py:1-5`

## Purpose

Package entry point — re-exports `SSTGAM` from `classifier.py` as the only
public name.

## Walkthrough

- `L3`: `from .classifier import SSTGAM`.
- `L5`: `__all__ = ["SSTGAM"]`.

## Calls

None (import only).

## Called by

- User code: `from sstgam import SSTGAM`.

**See:** [`../classifier/index.md`](../classifier/index.md)

**Back to:** [../INDEX.md](../INDEX.md)
