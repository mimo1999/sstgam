"""Turn long, per-patient-timestamped measurements into SSTGAM's wide format."""

import warnings

import numpy as np


def bin_timestamps(observations, *, id_col: str, time_col: str, feature_col: str, value_col: str,
                   bin_width: float, n_bins: int | None = None, origin="first", agg="mean",
                   counts: bool = False):
    """Bin irregular measurements onto a shared time grid, one row per patient.

    ``observations`` is a long pandas DataFrame with one row per measurement.
    Each measurement goes to bin ``floor((time - origin) / bin_width)`` and
    the result has one column ``<feature>_<bin start>`` per feature and bin,
    e.g. ``lactate_0``, ``lactate_6``, ``lactate_12`` for ``bin_width=6``
    (datetime times with a ``pd.Timedelta`` width are labelled by bin index).
    Bins a patient has no measurement in are NaN, which SSTGAM treats as
    missing. The bins depend only on the arguments, so training and test
    data binned with the same arguments get the same columns (pass
    ``n_bins`` to fix the column count; ``reindex`` fills any column the
    test data lacks).

    Parameters
    ----------
    origin : "first", "absolute" or pandas Series
        Time zero: each patient's first measurement, ``0`` on the raw time
        scale, or a Series of start times indexed by patient id (e.g.
        admission times).
    agg : str or callable, default="mean"
        How measurements in the same bin combine (any pandas aggregation:
        "mean", "last", "max", ...).
    n_bins : int, optional
        Keep the first ``n_bins`` bins and drop later measurements (with a
        warning giving how many).
    counts : bool, default=False
        Also add ``<feature>_n_<bin start>`` columns with the number of
        measurements per bin, a temporal signal for measurement frequency.
        Missing counts are 0, not NaN.
    """
    import pandas as pd

    frame = observations[[id_col, time_col, feature_col, value_col]].dropna(subset=[time_col, value_col])
    time = frame[time_col]
    if isinstance(origin, pd.Series):
        start = frame[id_col].map(origin)
        if start.isna().any():
            raise ValueError("origin has no start time for some patient ids.")
    elif origin == "first":
        start = frame.groupby(id_col)[time_col].transform("min")
    elif origin == "absolute":
        start = 0
    else:
        raise ValueError(f"origin must be 'first', 'absolute' or a Series, got {origin!r}")
    elapsed = time - start
    is_datetime = pd.api.types.is_timedelta64_dtype(elapsed)
    if is_datetime and not isinstance(bin_width, pd.Timedelta):
        raise ValueError("Datetime times need a pandas Timedelta bin_width, e.g. pd.Timedelta(hours=6).")
    bins = np.floor(elapsed / bin_width).astype(int)
    width_label = 1.0 if is_datetime else float(bin_width)  # datetime bins are labelled by index
    if (bins < 0).any():
        raise ValueError("Some measurements fall before origin.")
    if n_bins is not None and (bins >= n_bins).any():
        warnings.warn(f"Dropped {int((bins >= n_bins).sum())} measurements beyond n_bins={n_bins}.", stacklevel=2)
    keep = bins < n_bins if n_bins is not None else slice(None)
    frame = frame.assign(_bin=bins)[keep]

    grouped = frame.groupby([id_col, feature_col, "_bin"])[value_col]
    tables = {"": grouped.agg(agg).unstack([feature_col, "_bin"])}
    if counts:
        tables["n_"] = grouped.size().unstack([feature_col, "_bin"]).fillna(0)
    patients = observations[id_col].drop_duplicates()
    bin_ids = range(n_bins) if n_bins is not None else sorted(frame["_bin"].unique())
    features = sorted(frame[feature_col].unique())

    def start_label(bin_index: int) -> str:
        label = bin_index * width_label
        return str(int(label)) if float(label).is_integer() else str(label)

    columns = {}
    for prefix, table in tables.items():
        for feature in features:
            for bin_index in bin_ids:
                key = (feature, bin_index)
                column = table[key] if key in table.columns else pd.Series(np.nan, index=table.index)
                columns[f"{feature}_{prefix}{start_label(bin_index)}"] = (
                    column.reindex(patients).fillna(0) if prefix else column.reindex(patients))
    wide = pd.DataFrame(columns, index=pd.Index(patients, name=id_col))
    return wide
