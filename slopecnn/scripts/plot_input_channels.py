"""
plot_input_channels.py
======================
Paper figure of the CNN input for one slope: the six property channels and the
pore-pressure channel on the 42 x 243 grid (10 m cells).
Replaces v1.6 Fig. 3 (six channels on the old 42 x 127 grid).

    python scripts/plot_input_channels.py              (model 7, same slope as fig_section_example)
    python scripts/plot_input_channels.py --id 12

Writes output/paper_figures/fig_input_channels.png/.pdf
"""

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from slope_cnn.config import CNN_CHANNELS, PORE_P_MAX

ap = argparse.ArgumentParser()
ap.add_argument("--id", type=int, default=7)
ap.add_argument("--tensors", default=os.path.join(ROOT, "output", "tensors_v3", "tensors.npz"))
ap.add_argument("--out", default=os.path.join(ROOT, "output", "paper_figures"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "axes.labelsize": 7, "axes.edgecolor": "#52514e", "axes.linewidth": 0.5,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "savefig.dpi": 300, "pdf.fonttype": 42})

d = np.load(args.tensors)
dom = np.load(os.path.join(os.path.dirname(args.tensors), "domain_info.npz"))
n = int(np.where(d["ids"] == args.id)[0][0])
X, fos = d["X"][n], float(d["fos"][n])
extent = [float(dom["x_min"]), float(dom["x_max"]), float(dom["y_min"]), float(dom["y_max"])]
soil = X[0] > 0
cell = (extent[1] - extent[0]) / X.shape[2]
cols = np.flatnonzero(soil.any(axis=0))            # grid columns that hold soil for this slope
x_left = extent[0] + cols[0] * cell                 # plot from the model's left edge, not the grid's

names = {"gamma_sat": "Saturated γ$_{sat}$", "gamma_unsat": "Unsaturated γ$_{unsat}$",
         "c_peak": "Peak c′", "c_res": "Residual c$_r$",
         "phi_peak": "Peak φ′", "phi_res": "Residual φ$_r$"}
panels = [(c["idx"], f"{names[c['prop']]} ({c['unit']})") for c in CNN_CHANNELS]
panels.append((6, "Pore pressure u (kPa)"))

fig, axes = plt.subplots(4, 2, figsize=(6.9, 4.6), sharex=True, sharey=True,
                         gridspec_kw=dict(hspace=0.45, wspace=0.08))
for i, (ax, (ch, title)) in enumerate(zip(axes.ravel(), panels)):
    im = ax.imshow(np.ma.masked_where(~soil, X[ch]), origin="lower", extent=extent, cmap="viridis",
                   vmin=0, vmax=1, aspect="equal", interpolation="nearest")
    # the 10 m x 10 m cell grid the CNN sees: a thin line on every cell edge
    ax.set_xticks(np.arange(extent[0], extent[1] + cell / 2, cell), minor=True)
    ax.set_yticks(np.arange(extent[2], extent[3] + cell / 2, cell), minor=True)
    ax.grid(which="minor", color="#ffffff", lw=0.15, alpha=0.6)
    ax.tick_params(which="minor", length=0)
    ax.set_xlim(x_left, extent[1]); ax.set_ylim(extent[2], extent[3])   # after the ticks, which widen the view
    ax.set_title(f"({'abcdefg'[i]}) Channel {ch + 1}: {title}", loc="left")
axes[-1, -1].axis("off")
print(f"shown: grid columns {cols[0] + 1}-{X.shape[2]} of {X.shape[2]} (model starts at x = {x_left:.0f} m)")
cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.55, pad=0.015, aspect=25)
cb.set_label("normalised value")
for ax in axes[-1]:
    ax.set_xlabel("x (m)")
axes[-2, 1].xaxis.set_tick_params(labelbottom=True)
for ax in axes[:, 0]:
    ax.set_ylabel("y (m)")
fig.suptitle(f"CNN input for model {args.id} (FoS {fos:.2f}): 7 channels on a 42 × 243 grid of 10 m cells",
             fontsize=8, y=0.97)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(args.out, f"fig_input_channels.{ext}"), bbox_inches="tight", facecolor="white")
print(os.path.join(args.out, "fig_input_channels.png"))
