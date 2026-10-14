import numpy as np
import pandas as pd
import pytest

from sstgam import SSTGAM, bin_timestamps


def long_table():
    return pd.DataFrame({
        "patient": ["a", "a", "a", "a", "b", "b", "c"],
        "hour": [10.0, 11.0, 17.0, 10.5, 3.0, 20.0, 5.0],
        "lab": ["lactate", "lactate", "lactate", "urea", "lactate", "lactate", "urea"],
        "value": [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 9.0],
    })


def test_bins_from_each_patients_first_measurement():
    wide = bin_timestamps(long_table(), id_col="patient", time_col="hour", feature_col="lab",
                          value_col="value", bin_width=6, counts=True)
    assert list(wide.index) == ["a", "b", "c"]
    # a starts at hour 10: hours 10, 11 -> bin 0 (mean 2), hour 17 -> bin 1; b: hour 3 -> 0, hour 20 -> 2.
    np.testing.assert_allclose(wide["lactate_0"], [2.0, 2.0, np.nan])
    np.testing.assert_allclose(wide["lactate_6"], [5.0, np.nan, np.nan])
    np.testing.assert_allclose(wide["lactate_12"], [np.nan, 4.0, np.nan])
    np.testing.assert_allclose(wide["lactate_n_0"], [2, 1, 0])
    np.testing.assert_allclose(wide["urea_0"], [7.0, np.nan, 9.0])


def test_fixed_bins_match_between_train_and_test():
    arguments = {"id_col": "patient", "time_col": "hour", "feature_col": "lab", "value_col": "value",
                 "bin_width": 6, "n_bins": 2, "origin": "absolute", "agg": "last"}
    with pytest.warns(UserWarning, match="Dropped 2 measurements"):
        train = bin_timestamps(long_table(), **arguments)
    test = bin_timestamps(long_table().iloc[[4]], **arguments)
    assert list(train.columns) == ["lactate_0", "lactate_6", "urea_0", "urea_6"]
    assert list(test.reindex(columns=train.columns).columns) == list(train.columns)
    assert train.loc["a", "lactate_6"] == 3.0  # absolute hours 10 and 11 -> bin 1, last value


def test_binned_output_fits():
    generator = np.random.RandomState(0)
    n = 300
    observations = pd.DataFrame({
        "patient": np.repeat(np.arange(n), 5),
        "hour": generator.uniform(0, 48, size=n * 5),
        "lab": "lactate",
        "value": generator.normal(size=n * 5),
    })
    wide = bin_timestamps(observations, id_col="patient", time_col="hour", feature_col="lab",
                          value_col="value", bin_width=12, n_bins=4, origin="absolute")
    target = (wide.mean(axis=1).fillna(0) > 0).astype(int)
    model = SSTGAM(outer_bags=1).fit(wide, target)
    assert model.time_ids_ == [0, 12, 24, 36]
