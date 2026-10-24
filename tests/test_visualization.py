import importlib.util

import numpy as np
import pytest

from sstgam import SSTGAM

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("plotly") is None,
    reason="Plotly is an optional visualization dependency",
)


def fitted_model():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(80, 3))
    x[rng.random(x.shape) < 0.15] = np.nan
    y = (np.nan_to_num(x).sum(axis=1) > 0).astype(int)
    return SSTGAM(
        n_value_bases=5,
        n_time_bases=3,
        basis="bspline",
        outer_bags=1,
        random_state=0,
    ).fit(x, y, feature_names=["lab_0", "lab_1", "lab_2"])


def test_temporal_surface_figure_uses_public_model_surface():
    from sstgam.visualization import temporal_surface_figure

    figure = temporal_surface_figure(fitted_model(), "lab", n_value_points=8)

    assert len(figure.data) == 1
    assert figure.data[0].type == "surface"
    assert np.asarray(figure.data[0].z).shape == (8, 3)


def test_save_model_surfaces_writes_standalone_html(tmp_path):
    from sstgam.visualization import save_model_surfaces

    output = save_model_surfaces(
        fitted_model(), tmp_path / "surfaces.html", features=["lab"], n_value_points=8
    )

    assert output.exists()
    assert "plotly" in output.read_text(encoding="utf-8").lower()
