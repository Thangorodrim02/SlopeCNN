"""
make_parity_figure.py
=====================
Paper figure: predicted against calculated FoS for all 346 slopes, Soranzo et al. (2023)
and SlopeCNN, with 5 and 7 folds. Each slope is predicted by the models of the fold that
held it out (out of fold), averaged over the 3 seeds.

    python scripts/make_parity_figure.py

Writes output/paper_figures/fig_parity.png/.pdf
The panels show only the pooled R² (as in Table 2) and the failing-slope count; the fold-mean
RMSE of Table 2 is not shown, because the RMSE of the pooled predictions differs from it.
"""

import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--runs5", default=os.path.join(ROOT, "output", "runs"))
ap.add_argument("--runs7", default=os.path.join(ROOT, "output", "runs_k7"))
ap.add_argument("--out", default=os.path.join(ROOT, "output", "paper_figures"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "savefig.dpi": 300,
    "figure.dpi": 150, "pdf.fonttype": 42,
})
INK, GREY = "#0b0b0b", "#9a9993"
MODELS = [("soranzo_published", "Soranzo et al. (2023)", "#BA7517"),
          ("slopecnn_all6_pp", "SlopeCNN", "#185FA5")]
RUNS = {5: args.runs5, 7: args.runs7}


def oof(k, run):
    """Out-of-fold prediction per slope, averaged over the seeds."""
    files = glob.glob(os.path.join(RUNS[k], run, "pred_fold*_seed*.csv"))
    return pd.concat(pd.read_csv(p) for p in files).groupby("geometry_id").mean()


fig, axes = plt.subplots(2, 2, figsize=(5.6, 5.6), sharex=True, sharey=True)
lo, hi = 0.6, 2.8
xx = np.array([lo, hi])
for r, k in enumerate((5, 7)):
    for c, (run, label, col) in enumerate(MODELS):
        ax = axes[r, c]
        o = oof(k, run)
        y, p = o.fos_true.to_numpy(), o.fos_pred.to_numpy()
        err = p - y
        r2 = 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)
        rmse = np.sqrt(np.mean(err ** 2))
        missed = int(np.sum((y < 1) & (p >= 1)))
        ax.fill_between(xx, xx * 0.9, xx * 1.1, color="#e6e5e0", lw=0, zorder=0, label="±10 %")
        ax.plot(xx, xx, color="#52514e", lw=0.8, zorder=1, label="1:1")
        ax.axvline(1, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.axhline(1, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.scatter(y, p, s=9, color=col, alpha=0.75, lw=0.3, edgecolor="white", zorder=2)
        ax.set_title(f"{label}, {k} folds", color=INK)
        ax.text(0.04, 0.96, f"Pooled R² {r2:.3f}\nFoS < 1 predicted ≥ 1: {missed} of {int(np.sum(y < 1))}",
                transform=ax.transAxes, va="top", fontsize=6.8, color=INK, linespacing=1.4)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        print(f"{label}, k={k}: R2 {r2:.3f}, RMSE {rmse:.3f}, failing predicted stable {missed}")
for ax in axes[1]:
    ax.set_xlabel("FoS label (simplified Bishop)")
for ax in axes[:, 0]:
    ax.set_ylabel("FoS, predicted (out of fold)")
axes[1, 1].legend(loc="lower right", frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(args.out, f"fig_parity.{ext}"), bbox_inches="tight", facecolor="white")
print(os.path.join(args.out, "fig_parity.png"))
