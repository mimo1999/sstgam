"""Figures contrasting how SSTGAM's spline and tree methods learn and optimize.

Uses synthetic longitudinal data with a known value-time effect, so each plot
can show what the model recovers against the truth. Run from the repo root:

    python docs/figures/make_figures.py

Writes fig1-fig4 PNGs next to this script; the data and fits live in
synthetic.py. Needs matplotlib (not a package dependency).
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import log_loss
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
    SMOOTH,
    SPLINE,
    SURFACE,
    TREE,
    VALUE_GRID,
    WIGGLY,
    ebm_cells,
    ebm_surface_centred,
    fit_ebm,
    fit_spline,
    fit_tree,
    spline_backend,
    spline_surface,
    test_loss,
    test_values,
    train_cells,
    tree_lookup,
    tree_surface,
    truth_surface,
    y_test,
    y_train,
)

DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", DIVERGING_STOPS)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_2,
})


def style_heatmap(ax, image, title):
    ax.grid(False)
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_xticks([0, 7, 14], ["1", "8", "15"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def heatmap(ax, surface, title, limit):
    image = ax.imshow(surface, origin="lower", aspect="auto", cmap=DIVERGING, vmin=-limit, vmax=limit,
                      extent=(-0.5, N_DAYS - 0.5, VALUE_GRID[0], VALUE_GRID[-1]), interpolation="nearest")
    return style_heatmap(ax, image, title)


# ---------------------------------------------------------------- fits
start = time.time()
spline = fit_spline()
spline_seconds = time.time() - start
chosen_penalty = spline.model_.temporal_penalty_
start = time.time()
tree = fit_tree(600)
tree_seconds = time.time() - start
tree_best = tree.model_.best_rounds_[0]
ebm = fit_ebm()
truth = truth_surface()
limit = float(np.abs(truth).max())
print(f"spline: penalty={chosen_penalty:.3g}, {spline_seconds:.1f}s, test log loss {test_loss(spline):.4f}")
print(f"tree:   best round {tree_best}, {tree_seconds:.1f}s, test log loss {test_loss(tree):.4f}")
print(f"ebm:    test log loss {test_loss(ebm):.4f}")

# ---------------------------------------------------------------- fig 1: what each learns
fig = plt.figure(figsize=(15, 7.2))
grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.05], height_ratios=[1.15, 1], hspace=0.45, wspace=0.25)
axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
for ax, surface, title in zip(axes, [truth, spline_surface(spline), tree_surface(tree), ebm_surface_centred(ebm)],
                              ["True effect", "SSTGAM spline: smooth surface", "SSTGAM tree: value-bin x day table",
                               "EBM: 15 separate per-day functions"]):
    image = heatmap(ax, surface, title, limit)
axes[0].set_ylabel("Lab value")
for ax in axes[1:]:
    ax.set_yticklabels([])
colorbar = fig.colorbar(image, cax=fig.add_subplot(grid[0, 4]))
colorbar.set_label("Contribution to log-odds", color=INK_2)
colorbar.outline.set_visible(False)

for column, day in enumerate([0, 7, 14]):
    ax = fig.add_subplot(grid[1, column])
    ax.plot(VALUE_GRID, truth[:, day], color=INK, linewidth=1.5, linestyle=(0, (4, 3)), label="True effect")
    ax.plot(VALUE_GRID, spline_surface(spline)[:, day], color=SPLINE, label="SSTGAM spline")
    ax.step(VALUE_GRID, tree_surface(tree)[:, day], where="mid", color=TREE, label="SSTGAM tree")
    ax.plot(VALUE_GRID, ebm_surface_centred(ebm)[:, day], color=EBM, label="EBM")
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=0)
    ax.set_title(f"Day {day + 1}: effect of the lab value")
    ax.set_xlabel("Lab value")
    if column == 0:
        ax.set_ylabel("Contribution to log-odds")
        handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.45, -0.03))
ax = fig.add_subplot(grid[1, 3])
density = np.histogram(train_cells, bins=40, range=(VALUE_GRID[0], VALUE_GRID[-1]))
ax.bar(density[1][:-1], density[0], width=np.diff(density[1]) * 0.85, align="edge", color="#9ec5f4")
ax.set_title("Where the training data is")
ax.set_xlabel("Lab value (observed cells, all days)")
ax.set_ylabel("Cells")
fig.suptitle("Same data, three models: the spline is smooth, the tree is piecewise-constant, "
             "EBM fits every day on its own", x=0.06, ha="left", fontsize=13, fontweight="semibold", color=INK)
fig.text(0.06, 0.925, "Each heatmap is one lab's value x day effect; colour is the log-odds a single observation "
         "adds. EBM has no shared surface: its 15 per-day shape functions are stacked into one for comparison. "
         "All are centred over the observed training cells.", color=INK_2, fontsize=9.5)
fig.savefig(OUT / "fig1_what_each_model_learns.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 2: how each optimizes
default_iterations = spline_backend._MAX_NEWTON_ITERATIONS
spline_steps, spline_losses, spline_snapshots = [], [], {}
for iterations in range(1, 9):
    spline_backend._MAX_NEWTON_ITERATIONS = iterations
    model = fit_spline(temporal_penalty=chosen_penalty, static_penalty=chosen_penalty)
    spline_steps.append(iterations)
    spline_losses.append(test_loss(model))
    if iterations in (1, 2, 8):
        spline_snapshots[iterations] = spline_surface(model)
spline_backend._MAX_NEWTON_ITERATIONS = default_iterations

tree_rounds = [1, 2, 5, 10, 20, 40, 80, 150, 250, 400, 600]
tree_losses, tree_snapshots = [], {}
for rounds in tree_rounds:
    model = fit_tree(rounds)
    tree_losses.append(test_loss(model))
    if rounds in (10, 80, 600):
        tree_snapshots[rounds] = tree_surface(model)
base_rate_loss = log_loss(y_test, np.full(y_test.shape, y_train.mean()))

fig = plt.figure(figsize=(12, 7.6))
grid = fig.add_gridspec(2, 4, width_ratios=[1.35, 1, 1, 1], hspace=0.55, wspace=0.3, top=0.84)
low = min(min(spline_losses), min(tree_losses))
for row, (steps, losses, color, xlabel, title, scale) in enumerate([
        (spline_steps, spline_losses, SPLINE, "Newton iteration (all coefficients at once)",
         "Spline: penalized Newton (P-IRLS)", "linear"),
        (tree_rounds, tree_losses, TREE, "Boosting round (one shallow tree per term)",
         "Tree: cyclic Newton boosting", "log")]):
    ax = fig.add_subplot(grid[row, 0])
    ax.plot([0] + list(steps) if scale == "linear" else steps,
            [base_rate_loss] + list(losses) if scale == "linear" else losses,
            color=color, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xscale(scale)
    ax.set_ylim(low - 0.01, base_rate_loss + 0.01)
    ax.axhline(low, color=MUTED, linewidth=1.0, linestyle=(0, (2, 2)), zorder=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test log loss")
    ax.annotate(f"{losses[-1]:.4f}", (steps[-1], losses[-1]), textcoords="offset points", xytext=(0, 8),
                ha="right", color=INK_2, fontsize=9)
    snapshots = spline_snapshots if row == 0 else tree_snapshots
    label = "iteration" if row == 0 else "round"
    for column, (step, surface) in enumerate(snapshots.items(), start=1):
        ax = fig.add_subplot(grid[row, column])
        heatmap(ax, surface, f"After {label} {step}", limit)
        if column == 1:
            ax.set_ylabel("Lab value")
        else:
            ax.set_yticklabels([])
fig.suptitle("The spline solves its fit in a few Newton steps; the tree builds its table up over hundreds of rounds",
             x=0.06, ha="left", fontsize=13, fontweight="semibold", color=INK)
fig.text(0.06, 0.925, f"Left: held-out log loss as each optimizer runs (dashed line: best reached; the spline "
         f"curve starts at the base-rate model). Right: the learned surface along the way, on figure 1's colour "
         f"scale.\nFull fits: spline {spline_seconds:.1f}s including the penalty search, tree {tree_seconds:.1f}s "
         f"for 600 rounds.", color=INK_2, fontsize=9.5, va="top")
fig.savefig(OUT / "fig2_how_each_model_optimizes.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 3: spline smoothness choice
search = spline.model_.penalty_search_
penalties = {"Too smooth (x1000)": chosen_penalty * 1000, "Chosen by \"auto\"": chosen_penalty,
             "Too wiggly (/1000)": chosen_penalty / 1000}
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"wspace": 0.25})
left.plot(list(search), list(search.values()), color=SPLINE, marker="o", markersize=6,
          markeredgecolor=SURFACE, markeredgewidth=1.5)
left.plot([chosen_penalty], [search[chosen_penalty]], marker="o", markersize=11, color=SPLINE,
          markeredgecolor=SURFACE, markeredgewidth=2)
left.annotate(f"chosen: {chosen_penalty:.3g}", (chosen_penalty, search[chosen_penalty]),
              textcoords="offset points", xytext=(10, 10), color=INK_2, fontsize=9)
left.set_xscale("log")
left.set_title("Penalty search (held-out log loss)")
left.set_xlabel("Penalty (temporal = static)")
left.set_ylabel("Held-out log loss")
right.plot(VALUE_GRID, truth[:, 14], color=INK, linewidth=1.5, linestyle=(0, (4, 3)), label="True effect")
for color, (label, penalty) in zip([SMOOTH, SPLINE, WIGGLY], penalties.items()):
    model = fit_spline(temporal_penalty=penalty, static_penalty=penalty)
    right.plot(VALUE_GRID, spline_surface(model)[:, 14], color=color, label=label)
right.axhline(0, color=AXIS, linewidth=1.0, zorder=0)
right.set_title("Day 15 effect at three penalties")
right.set_xlabel("Lab value")
right.set_ylabel("Contribution to log-odds")
right.legend(loc="upper center")
fig.suptitle("The spline's only smoothness knob is its penalty, chosen from the data",
             x=0.06, y=1.02, ha="left", fontsize=13, fontweight="semibold", color=INK)
fig.savefig(OUT / "fig3_spline_penalty_choice.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 4: scoring one patient
missing_days = np.isnan(test_values).sum(axis=1)
patient = int(np.flatnonzero((missing_days >= 3) & (missing_days <= 5) & (y_test == 1))[0])
values = test_values[patient]
seen = np.isfinite(values)
spline_cells = np.array([spline.temporal_surface("lab", [v], [d])[0, 0] if s else 0.0
                         for v, d, s in zip(values, DAYS, seen)])
tree_cells = np.where(seen, tree_lookup(tree, np.nan_to_num(values), DAYS), 0.0)
ebm_day_cells = ebm_cells(ebm, values)  # EBM scores missing days too

fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 1.3], "hspace": 0.3})
top.plot(DAYS, values, color=INK_2, linewidth=1.5, marker="o", markersize=6,  # NaN days break the line
         markerfacecolor=INK_2, markeredgecolor=SURFACE, markeredgewidth=1.5)
for day in DAYS[~seen]:
    top.axvspan(day - 0.45, day + 0.45, color=GRID, alpha=0.6, linewidth=0)
    top.text(day, top.get_ylim()[1], "missing", ha="center", va="top", color=MUTED, fontsize=8)
top.set_title("One patient's 15 days of the lab (shaded: not measured)")
top.set_ylabel("Lab value")
width, gap = 0.24, 0.03  # gap: the surface-coloured spacer between grouped bars
bottom.bar(DAYS[seen] - (width + gap), spline_cells[seen], width=width, color=SPLINE,
           label=f"SSTGAM spline (sum {spline_cells.sum():+.2f})")
bottom.bar(DAYS[seen], tree_cells[seen], width=width, color=TREE,
           label=f"SSTGAM tree (sum {tree_cells.sum():+.2f})")
bottom.bar(DAYS + (width + gap), ebm_day_cells, width=width, color=EBM,
           label=f"EBM (sum {ebm_day_cells.sum():+.2f}, {ebm_day_cells[~seen].sum():+.2f} from missing days)")
bottom.axhline(0, color=AXIS, linewidth=1.0)
for day in DAYS[~seen]:
    bottom.axvspan(day - 0.45, day + 0.45, color=GRID, alpha=0.6, linewidth=0, zorder=0)
    bottom.text(day - (width + gap) / 2, 0, "0  0", ha="center", va="bottom", color=MUTED, fontsize=7)
bottom.set_title("What each model adds to the log-odds per day")
bottom.set_ylabel("Contribution to log-odds")
bottom.set_xlabel("Day")
bottom.set_xticks(DAYS, [str(day + 1) for day in DAYS])
bottom.legend(loc="upper left", ncol=3)
fig.suptitle("Scoring a patient: SSTGAM reads each observed day off one shared surface and adds 0 for\n"
             "missing days; EBM scores each day with its own function, missing days included",
             x=0.06, y=1.03, ha="left", fontsize=13, fontweight="semibold", color=INK)
fig.savefig(OUT / "fig4_scoring_one_patient.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", *sorted(path.name for path in OUT.glob("fig*.png")))
