"""
make_lr_figure.py
=================
Paper figure: learning rate during training. SlopeCNN: cosine schedule from 1e-3 to 1e-6 over
200 epochs (CosineAnnealingLR, stepped once per epoch, as in train_cv.py); Soranzo et al. (2023):
constant 1e-3. The grey band is the range of epochs at which the SlopeCNN runs stopped (early
stopping), read from the finished runs.

    python scripts/make_lr_figure.py

Writes output/paper_figures/fig_lr_schedule.png/.pdf
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=[os.path.join(ROOT, "output", "runs"), os.path.join(ROOT, "output", "runs_k7")])
ap.add_argument("--out", default=os.path.join(ROOT, "output", "paper_figures"))
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5,
                     "savefig.dpi": 300, "pdf.fonttype": 42})
LR0, LRMIN, T = 1e-3, 1e-6, 200
ep = np.arange(1, T + 1)
cos = LRMIN + 0.5 * (LR0 - LRMIN) * (1 + np.cos(np.pi * (ep - 1) / T))
stops = [json.load(open(p))["epochs_run"] for d in args.runs
         for p in glob.glob(os.path.join(d, "slopecnn_all6_pp", "fold*_seed*.json"))]
fig, ax = plt.subplots(figsize=(3.5, 2.2))
if stops:
    ax.axvspan(min(stops), max(stops), color="#e6e5e0", lw=0,
               label=f"SlopeCNN runs stopped (epochs {min(stops)}–{max(stops)})")
ax.plot(ep, cos, color="#185FA5", lw=1.6, label="SlopeCNN: cosine schedule")
ax.plot(ep, np.full_like(ep, LR0, dtype=float), color="#BA7517", lw=1.6, ls=(0, (4, 2)),
        label="Soranzo et al. (2023): constant")
ax.set_yscale("log"); ax.set_ylim(5e-7, 2e-3); ax.set_xlim(0, T)
ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate η")
ax.legend(loc="lower left", frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(args.out, f"fig_lr_schedule.{ext}"), bbox_inches="tight", facecolor="white")
print(os.path.join(args.out, "fig_lr_schedule.png"), "stops", min(stops), max(stops), len(stops))
