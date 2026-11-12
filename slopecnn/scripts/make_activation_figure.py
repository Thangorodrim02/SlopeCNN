"""
make_activation_figure.py
=========================
Paper figure: the activation functions of the two networks, ReLU (SlopeCNN, Eq. 8) and
leaky ReLU with alpha = 0.1 (Soranzo et al. 2023, Eq. 9).

    python scripts/make_activation_figure.py

Writes output/paper_figures/fig_activation.png/.pdf
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(ROOT, "output", "paper_figures"))
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5,
                     "savefig.dpi": 300, "pdf.fonttype": 42})
z = np.linspace(-3, 3, 601)
ALPHA = 0.1
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
for ax, (a, title, col) in zip(axes, [(np.maximum(0, z), "ReLU (SlopeCNN)", "#185FA5"),
                                       (np.maximum(ALPHA * z, z), f"Leaky ReLU, α = {ALPHA} (Soranzo et al.)", "#BA7517")]):
    ax.axhline(0, color="#9a9993", lw=0.6); ax.axvline(0, color="#9a9993", lw=0.6)
    ax.plot(z, a, color=col, lw=1.6)
    ax.set_title(title, fontsize=7.5); ax.set_xlabel("z"); ax.set_xlim(-3, 3); ax.set_ylim(-0.6, 3)
axes[0].set_ylabel("a")
fig.tight_layout(w_pad=0.8)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(args.out, f"fig_activation.{ext}"), bbox_inches="tight", facecolor="white")
print(os.path.join(args.out, "fig_activation.png"))
