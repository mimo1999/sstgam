"""Interactive 3D views of SSTGAM's value x time surfaces (plotly).

Every plot keeps time as an axis: the models learn one surface over value and
day, not a separate curve per day. Run from the repo root:

    python docs/figures/make_3d_figures.py

Writes sstgam_3d.html next to this script (plotly.js loaded from jsdelivr).
Needs plotly (not a package dependency).
"""

import html

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from synthetic import (
    AXIS,
    DAYS,
    DIVERGING_STOPS,
    EBM,
    GRID,
    INK,
    INK_2,
    MUTED,
    N_DAYS,
    OUT,
    SPLINE,
    SURFACE,
    TREE,
    VALUE_GRID,
    Xtr,
    ebm_cells,
    ebm_surface,
    ebm_surface_centred,
    fit_ebm,
    fit_spline,
    fit_tree,
    spline_backend,
    spline_surface,
    test_loss,
    test_values,
    train_values,
    tree_lookup,
    tree_surface,
    true_effect,
    truth_offset,
    truth_surface,
    y_test,
    y_train,
)

from sstgam import SSTGAM

PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@3.5.0/plotly.min.js"
FINE_DAYS = np.linspace(0, N_DAYS - 1, 57)  # continuous time for the spline and the truth
COLORSCALE = [[i / (len(DIVERGING_STOPS) - 1), color] for i, color in enumerate(DIVERGING_STOPS)]
FONT = dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK_2, size=12)


def stepped(surface):
    """Day-discrete surface -> flat steps per day (two columns per day), so plotly doesn't ramp between days."""
    days = np.repeat(DAYS, 2) + np.tile([-0.45, 0.45], N_DAYS)
    return days, np.repeat(surface, 2, axis=1)


def spline_fine_surface(model):
    backend = model.model_
    return backend.temporal_surface_normalized(0, backend.normalize_values(0, VALUE_GRID),
                                               backend.normalize_time(FINE_DAYS))


def axis(title, **extra):
    return dict(title=dict(text=title, font=dict(color=INK_2, size=11)), backgroundcolor=SURFACE,
                gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=INK_2, size=10),
                showbackground=True, **extra)


def surface(days, z, name, limit, show_scale=False, opacity=1.0):
    return go.Surface(
        x=days, y=VALUE_GRID, z=z, name=name, colorscale=COLORSCALE, cmin=-limit, cmax=limit,
        opacity=opacity, showscale=show_scale,
        colorbar=dict(title=dict(text="Log-odds", font=FONT), tickfont=FONT, outlinewidth=0, len=0.7,
                      thickness=12),
        contours=dict(z=dict(show=False)), lighting=dict(ambient=0.75, diffuse=0.5, specular=0.05),
        hovertemplate=f"<b>{html.escape(name)}</b><br>Day %{{customdata:.0f}}<br>Lab value %{{y:.2f}}"
                      "<br>Contribution %{z:+.2f}<extra></extra>",
        customdata=np.broadcast_to(np.asarray(days) + 1, np.shape(z)))


def panel(traces, title, limit, height=430):
    """One 3D scene as its own figure, so the page grid can reflow panels on narrow screens."""
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(color=INK, size=14), x=0.02, y=0.97),
        height=height, margin=dict(l=0, r=0, t=56, b=0), paper_bgcolor=SURFACE, font=FONT, showlegend=False,
        scene=dict(xaxis=axis("Day", tickvals=[0, 4, 9, 14], ticktext=["1", "5", "10", "15"]),
                   yaxis=axis("Lab value"), zaxis=axis("Log-odds", range=[-1.15 * limit, 1.15 * limit]),
                   aspectmode="manual", aspectratio=dict(x=1.2, y=1, z=0.6),
                   camera=dict(eye=dict(x=1.5, y=-1.5, z=0.85))))
    return fig


def per_day_centred(surface_on_grid):
    """Remove each day's mean over that day's observed training cells (what a per-day model can identify)."""
    centred = surface_on_grid.copy()
    for day in DAYS:
        cells = train_values[:, day]
        cells = cells[np.isfinite(cells)]
        centred[:, day] -= np.interp(cells, VALUE_GRID, surface_on_grid[:, day]).mean()
    return centred


def shape_error(surface_per_day):
    """RMS error vs the truth over observed training cells, after per-day centring of both."""
    truth = per_day_centred(true_effect(VALUE_GRID[:, None], DAYS[None, :]))
    errors = []
    for day in DAYS:
        cells = train_values[:, day]
        cells = cells[np.isfinite(cells)]
        errors.append(np.interp(cells, VALUE_GRID, surface_per_day[:, day])
                      - np.interp(cells, VALUE_GRID, truth[:, day]))
    return float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))


def animated(title_prefix, steps, surfaces, step_losses, days, name, limit):
    # Open on the final model; Play replays every step from the start.
    fig = panel([surface(days, surfaces[-1], name, limit, show_scale=True)],
                f"{title_prefix} {steps[-1]} | test log loss {step_losses[-1]:.4f}", limit, height=520)
    fig.frames = [go.Frame(data=[surface(days, z, name, limit, show_scale=True)], name=str(step),
                           layout=dict(title_text=f"{title_prefix} {step} | test log loss {loss:.4f}"))
                  for step, z, loss in zip(steps, surfaces, step_losses)]
    play = dict(frame=dict(duration=700, redraw=True), fromcurrent=False, transition=dict(duration=0))
    fig.update_layout(
        margin=dict(l=0, r=0, t=56, b=70),
        sliders=[dict(active=len(steps) - 1, x=0.12, len=0.86, y=0.02, pad=dict(t=10), currentvalue=dict(visible=False),
                      font=FONT, bgcolor=GRID, bordercolor=AXIS, activebgcolor=MUTED,
                      steps=[dict(label=str(step), method="animate",
                                  args=[[str(step)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                          transition=dict(duration=0))]) for step in steps])],
        updatemenus=[dict(type="buttons", x=0.0, y=0.02, xanchor="left", yanchor="bottom", pad=dict(t=10),
                          bgcolor=SURFACE, bordercolor=AXIS, font=FONT, showactive=False,
                          buttons=[dict(label="Play", method="animate", args=[None, play])])])
    return fig


# ---------------------------------------------------------------- fits
spline = fit_spline()
penalty = spline.model_.temporal_penalty_
tree = fit_tree(600)
ebm = fit_ebm()
per_day_names = [f"labday{day + 1}_0" for day in DAYS]  # 15 single-time features: no sharing across days
per_day = SSTGAM(method="spline", n_value_bases=8, n_time_bases=5, outer_bags=1, random_state=0).fit(
    Xtr, y_train, feature_names=per_day_names)
per_day_grid = np.column_stack([per_day.temporal_surface(f"labday{day + 1}", VALUE_GRID, [0])[:, 0]
                                for day in DAYS])
spline_fine = spline_fine_surface(spline)
truth_fine = true_effect(VALUE_GRID[:, None], FINE_DAYS[None, :]) - truth_offset
limit = float(np.abs(truth_surface()).max())
losses = {"Per-day independent spline": test_loss(per_day), "EBM (per-day)": test_loss(ebm),
          "SSTGAM spline": test_loss(spline), "SSTGAM tree": test_loss(tree)}
errors = {"Per-day independent spline": shape_error(per_day_grid),
          "EBM (per-day)": shape_error(per_day_centred(ebm_surface(ebm))),
          "SSTGAM spline": shape_error(per_day_centred(spline_surface(spline))),
          "SSTGAM tree": shape_error(per_day_centred(tree_surface(tree)))}
for name in losses:
    print(f"{name:20s} test log loss {losses[name]:.4f}  shape RMSE vs truth {errors[name]:.3f}")

sections = []

# ---------------------------------------------------------------- 1: the whole surface
sections.append((
    "One surface over value and time",
    "Each lab gets a single effect surface over (value, day), learned from all days at once. Drag to rotate, "
    "scroll to zoom, hover for values. The true effect is monotone early and U-shaped late; both models "
    "recover that change over time, the spline as a smooth surface, the tree as a table of value bins by day.",
    [panel([surface(FINE_DAYS, truth_fine, "True effect", limit)], "True effect", limit),
     panel([surface(FINE_DAYS, spline_fine, "SSTGAM spline", limit)], "SSTGAM spline", limit),
     panel([surface(*stepped(tree_surface(tree)), "SSTGAM tree", limit, show_scale=True)], "SSTGAM tree",
           limit)]))

# ---------------------------------------------------------------- 2: temporal consistency
centred_limit = float(np.abs(per_day_centred(truth_surface())).max())
centred = {"Per-day independent spline": stepped(per_day_grid),
           "EBM (per-day)": stepped(per_day_centred(ebm_surface(ebm))),
           "SSTGAM spline": (DAYS, per_day_centred(spline_surface(spline))),
           "SSTGAM tree": stepped(per_day_centred(tree_surface(tree)))}
sections.append((
    "Why sharing across time matters",
    "The first two treat each day as an unrelated feature, so day 7 learns nothing from days 6 and 8: the "
    "same spline GAM fitted per day, and EBM (interactions=0), which fits one boosted shape function per "
    "column. Their 15 curves come out noisy and jump from day to day. SSTGAM pools every day into one "
    "surface, so neighbouring days inform each other. Shape error is the RMS gap to the true effect over the "
    "observed training cells (lower is better); all four are centred per day, the only level a per-day "
    "model can identify.",
    [panel([surface(*centred[name], name, centred_limit, show_scale=name == "SSTGAM tree")],
           f"{name}<br><span style='font-size:12px;color:{INK_2}'>shape error {errors[name]:.3f} · "
           f"test log loss {losses[name]:.4f}</span>", centred_limit) for name in losses]))

# ---------------------------------------------------------------- 3 & 4: training over time
default_iterations = spline_backend._MAX_NEWTON_ITERATIONS
iterations, spline_steps, spline_step_losses = [0, 1, 2, 3, 4, 6, 8], [], []
base_rate = y_train.mean()
for count in iterations:
    if count == 0:  # the base-rate model: a flat surface
        spline_steps.append(np.zeros_like(spline_fine))
        spline_step_losses.append(float(-np.mean(y_test * np.log(base_rate) + (1 - y_test) * np.log(1 - base_rate))))
        continue
    spline_backend._MAX_NEWTON_ITERATIONS = count
    model = fit_spline(temporal_penalty=penalty, static_penalty=penalty)
    spline_steps.append(spline_fine_surface(model))
    spline_step_losses.append(test_loss(model))
spline_backend._MAX_NEWTON_ITERATIONS = default_iterations
sections.append((
    "Spline training: penalized Newton steps on the whole surface",
    "Each Newton step updates every coefficient of the surface at once, across all days. Press Play or drag "
    "the slider: the full value x time shape appears within two steps; later steps only polish it. "
    "Iteration 0 is the base-rate model (a flat surface).",
    [animated("Spline after Newton iteration", iterations, spline_steps, spline_step_losses, FINE_DAYS,
              "SSTGAM spline", limit)]))

rounds = [1, 5, 10, 20, 40, 80, 150, 300, 600]
tree_models = [fit_tree(count) for count in rounds]
sections.append((
    "Tree training: hundreds of small boosting updates",
    "Each round fits a shallow tree to the current residual statistics of the whole value-bin x day table "
    "and adds a damped step. The surface builds up gradually: coarse value splits first, then splits along "
    "time, then refinements.",
    [animated("Tree after boosting round", rounds, [stepped(tree_surface(model))[1] for model in tree_models],
              [test_loss(model) for model in tree_models], stepped(tree_surface(tree))[0], "SSTGAM tree",
              limit)]))

# ---------------------------------------------------------------- 5: one patient in 3D
missing_days = np.isnan(test_values).sum(axis=1)
patient = int(np.flatnonzero((missing_days >= 3) & (missing_days <= 5) & (y_test == 1))[0])
values = test_values[patient]
seen = np.isfinite(values)
ebm_all_days = ebm_cells(ebm, values)  # EBM also scores the missing days
contributions = {
    "SSTGAM spline": np.array([spline.temporal_surface("lab", [v], [d])[0, 0]
                               for v, d in zip(values[seen], DAYS[seen])]),
    "SSTGAM tree": tree_lookup(tree, values[seen], DAYS[seen]),  # the model's own contributions
    "EBM": ebm_all_days[seen],
}
patient_panels = []
floor = -1.15 * limit  # the trajectory is drawn as a shadow on the floor of the box
for (name, cells), color, (days, z) in zip(contributions.items(), [SPLINE, TREE, EBM],
                                           [(FINE_DAYS, spline_fine), stepped(tree_surface(tree)),
                                            stepped(ebm_surface_centred(ebm))]):
    stems, guides = [[], [], []], [[], [], []]  # stems: 0 -> contribution; guides: floor -> 0
    for day, value, cell in zip(DAYS[seen], values[seen], cells):
        for coordinate, (start, end) in zip(stems, [(day, day), (value, value), (0.0, cell)]):
            coordinate += [start, end, None]
        for coordinate, (start, end) in zip(guides, [(day, day), (value, value), (floor, 0.0)]):
            coordinate += [start, end, None]
    patient_panels.append(panel([
        surface(days, z, name, limit, show_scale=name == "EBM", opacity=0.5),
        go.Scatter3d(x=guides[0], y=guides[1], z=guides[2], mode="lines",
                     line=dict(color=MUTED, width=2, dash="dot"), hoverinfo="skip"),
        go.Scatter3d(x=stems[0], y=stems[1], z=stems[2], mode="lines", line=dict(color=color, width=7),
                     hoverinfo="skip"),
        go.Scatter3d(x=DAYS, y=values, z=np.full(N_DAYS, floor), mode="lines+markers", connectgaps=False,
                     line=dict(color=INK_2, width=4), marker=dict(size=4, color=INK_2),
                     hovertemplate="Day %{customdata}<br>Lab value %{y:.2f}<extra>trajectory</extra>",
                     customdata=DAYS + 1),
        go.Scatter3d(x=DAYS[seen], y=values[seen], z=cells, mode="markers",
                     marker=dict(size=5, color=color, line=dict(color=SURFACE, width=2)),
                     hovertemplate="Day %{customdata}<br>Lab value %{y:.2f}<br>Adds %{z:+.2f} log-odds"
                                   "<extra></extra>", customdata=DAYS[seen] + 1),
    ], f"{name}: {cells.sum():+.2f} log-odds from {int(seen.sum())} observed days" + (
        f"<br><span style='font-size:12px;color:{INK_2}'>plus {ebm_all_days[~seen].sum():+.2f} from its "
        f"{int((~seen).sum())} missing days (SSTGAM: 0)</span>" if name == "EBM" else ""), limit, height=500))
sections.append((
    "Scoring one patient: the trajectory reads the surface day by day",
    f"The grey path on the floor of each box is one patient's lab over 15 days; it breaks on the "
    f"{int((~seen).sum())} missing days. Each observed day looks up the model at (value, day); the coloured "
    "stems rise or fall from 0 by what that day adds to the log-odds. In SSTGAM missing days add nothing; "
    "EBM learns a separate score for a missing value on each day, shown in its panel title. The same low "
    "value lowers risk early and raises it late.",
    patient_panels))

# ---------------------------------------------------------------- page
body, plot_index = [], 0
for title, text, figures in sections:
    divs = []
    for figure in figures:
        divs.append(pio.to_html(figure, full_html=False, include_plotlyjs=False, div_id=f"plot{plot_index}",
                                default_width="100%", default_height=f"{figure.layout.height}px",
                                config=dict(displaylogo=False, responsive=True)))
        plot_index += 1
    body.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p>'
                f'<div class="grid grid-{len(figures)}">{"".join(divs)}</div></section>')
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SSTGAM in 3D</title>
<script src="{PLOTLY_JS}"></script>
<style>
  :root {{ --surface: {SURFACE}; --ink: {INK}; --ink-2: {INK_2}; --rule: {GRID}; }}
  body {{ margin: 0; background: var(--surface); color: var(--ink);
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.5; }}
  main {{ max-width: 1500px; margin: 0 auto; padding: 32px 16px 64px; }}
  h1 {{ font-size: 26px; margin: 0 0 8px; }}
  h2 {{ font-size: 18px; margin: 40px 0 6px; padding-top: 24px; border-top: 1px solid var(--rule); }}
  p {{ color: var(--ink-2); max-width: 900px; margin: 0 0 12px; }}
  .grid {{ display: grid; gap: 8px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); }}
  .grid > div {{ min-width: 0; }}
  .grid-1 {{ max-width: 900px; }}
</style>
</head>
<body>
<main>
<h1>SSTGAM in 3D: one surface over value and time</h1>
<p>Synthetic data with a known effect: 4,000 patients, one lab measured daily for 15 days, about 25% of days
missing. Every surface plots lab value x day x contribution to the log-odds, centred so 0 is an average
observation. Spline penalty chosen by <code>"auto"</code>: {penalty:.3g}.</p>
{''.join(body)}
</main>
</body>
</html>
"""
(OUT / "sstgam_3d.html").write_text(page, encoding="utf-8")
print("wrote sstgam_3d.html")
