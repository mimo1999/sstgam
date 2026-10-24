"""Interactive Plotly visualisations for fitted SSTGAM shape functions.

Plotly is an optional dependency.  The functions in this module deliberately
use the public fitted-model API (``temporal_surface`` and
``value_time_table``), so visualisation can evolve independently of the
training implementation.
"""

from __future__ import annotations

from collections.abc import Iterable
from html import escape
from pathlib import Path

import numpy as np
from sklearn.utils.validation import check_is_fitted


def _plotly():
    """Import Plotly lazily and provide an actionable optional-dependency error."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Plotly visualisation requires the optional dependency 'plotly'. "
            "Install it with `pip install plotly`."
        ) from exc
    return go, pio


def _default_values(model, feature: str, n_points: int) -> np.ndarray:
    """Choose a plotting range from the model's fitted value normalisation."""
    backend = model.model_
    index = model._temporal_index(feature)
    if backend.value_transform == "quantile":
        source_values, _ = backend._quantile_tables[index]
        lo, hi = float(source_values[0]), float(source_values[-1])
    else:
        centre = float(backend._feature_center[index])
        scale = float(backend._feature_scale[index])
        lo, hi = centre - 3.0 * scale, centre + 3.0 * scale
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = -1.0, 1.0
    return np.linspace(float(lo), float(hi), max(int(n_points), 2))


def temporal_surface_figure(
    model,
    feature: str,
    *,
    values: Iterable[float] | None = None,
    time_coordinates: Iterable[float] | None = None,
    n_value_points: int = 40,
    title: str | None = None,
):
    """Return a Plotly 3D figure for one fitted temporal shape function.

    Parameters
    ----------
    model:
        A fitted ``SSTGAM`` or ``SSTGAMRegressor`` using ``method="spline"``
        or ``method="tree"``.
    feature:
        Name of a fitted temporal feature.
    values, time_coordinates:
        Plotting coordinates in the model's original units.  Defaults are
        derived from the fitted model.
    n_value_points:
        Number of value coordinates for a spline surface.
    title:
        Optional Plotly title.

    Returns
    -------
    plotly.graph_objects.Figure
        A rotatable Plotly ``Surface`` figure.
    """
    go, _ = _plotly()
    check_is_fitted(model, "model_")
    if feature not in model.temporal_features_:
        raise ValueError(f"Unknown temporal feature {feature!r}; fitted features: {model.temporal_features_}")

    times = (np.asarray(model.time_ids_, dtype=float) if time_coordinates is None
             else np.asarray(list(time_coordinates), dtype=float))
    if times.ndim != 1 or times.size == 0:
        raise ValueError("time_coordinates must be a non-empty one-dimensional sequence.")

    if model.method == "spline":
        grid_values = (_default_values(model, feature, n_value_points)
                       if values is None else np.asarray(list(values), dtype=float))
        if grid_values.ndim != 1 or grid_values.size == 0:
            raise ValueError("values must be a non-empty one-dimensional sequence.")
        z = np.asarray(model.temporal_surface(feature, grid_values, times, center_time=True), dtype=float)
        y = grid_values
    elif model.method == "tree":
        edges, table = model.value_time_table(feature)
        edges = np.asarray(edges, dtype=float)
        table = np.asarray(table, dtype=float)
        y = (edges[:-1] + edges[1:]) / 2.0
        # Tree tables are fitted only at the model's time ids.
        if not np.array_equal(times, np.asarray(model.time_ids_, dtype=float)):
            raise ValueError("Tree visualisations require time_coordinates to equal model.time_ids_.")
        z = table
    else:
        raise ValueError(f"Unsupported model method {model.method!r}.")

    scale_limit = float(np.nanmax(np.abs(z))) if np.size(z) else 1.0
    scale_limit = max(scale_limit, 1e-9)
    label = "Link-scale contribution"
    trace = go.Surface(
        x=times,
        y=y,
        z=z,
        name=feature,
        colorscale="RdBu",
        reversescale=True,
        cmin=-scale_limit,
        cmax=scale_limit,
        colorbar={"title": label},
        hovertemplate=(f"<b>{escape(feature)}</b><br>Time: %{{x}}<br>Value: %{{y:.4g}}<br>"
                      f"{escape(label)}: %{{z:+.4g}}<extra></extra>"),
    )
    figure = go.Figure(data=[trace])
    figure.update_layout(
        title=title or f"{feature}: value × time shape",
        height=520,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene={
            "xaxis_title": "Time",
            "yaxis_title": "Value",
            "zaxis_title": label,
            "zaxis": {"range": [-1.15 * scale_limit, 1.15 * scale_limit]},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.2, "y": 1.0, "z": 0.65},
        },
        showlegend=False,
    )
    return figure


def save_model_surfaces(
    model,
    output_path: str | Path,
    *,
    features: Iterable[str] | None = None,
    values: Iterable[float] | None = None,
    time_coordinates: Iterable[float] | None = None,
    n_value_points: int = 40,
    title: str = "SSTGAM shape functions",
) -> Path:
    """Save all selected fitted temporal shape functions as a standalone HTML page."""
    _, pio = _plotly()
    selected = list(model.temporal_features_ if features is None else features)
    if not selected:
        raise ValueError("features must contain at least one temporal feature.")
    figures = [temporal_surface_figure(
        model,
        feature,
        values=values,
        time_coordinates=time_coordinates,
        n_value_points=n_value_points,
    ) for feature in selected]

    blocks = [pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs="cdn" if index == 0 else False,
        div_id=f"sstgam-surface-{index}",
        config={"displaylogo": False, "responsive": True},
    ) for index, figure in enumerate(figures)]
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{escape(title)}</title></head>'
            f'<body><h1>{escape(title)}</h1>' + "".join(blocks) + "</body></html>")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")
    return destination


__all__ = ["save_model_surfaces", "temporal_surface_figure"]
