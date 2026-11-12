"""
make_dataset_figures.py
=======================
Dataset figures for the paper, redrawn for dataset v3 (346 Slide2 models). They
replace the v1.6 figures made from the 300 generated slopes.

    python scripts/make_dataset_figures.py
    python scripts/make_dataset_figures.py --example 7      (slope shown in the section figure)

Needs slopelem (reads the .slim files). Default paths are the repository layout:
../data/slide2_models, ../data/soil_properties_346geom_v3_cnn_input.xlsx, ../slopelem.

Writes output/paper_figures/ (PNG 300 dpi + PDF):
    fig_section_example     one model: six layers, five boundaries, water table, slope angle, height
                            (replaces v1.6 "Random slope generation", Fig. 1)
    fig_property_boxplots   the six properties of the six layers over the 346 models (v1.6 Fig. 2)
    fig_fos_vs_angle        FoS label vs slope angle, by stability class and water-table type
                            (v1.6 "FOS values for the generated data set")
    fig_geometry_envelope   all 346 slope surfaces (a) and water tables (b), coloured by FoS
                            (v1.6 "Failure mode envelope")
The input-tensor figure (v1.6 Fig. 3) is scripts/plot_input_channels.py.
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)                # repository root
sys.path.insert(0, ROOT)
from slope_cnn.config import LAYERS, LAYER_COLORS

ap = argparse.ArgumentParser()
ap.add_argument("--models", default=os.path.join(REPO, "data", "slide2_models"))
ap.add_argument("--excel", default=os.path.join(REPO, "data", "soil_properties_346geom_v3_cnn_input.xlsx"))
ap.add_argument("--wt", default=os.path.join(ROOT, "output", "tensors_v3", "wt_depth.csv"))
ap.add_argument("--slopelem", default=os.path.join(REPO, "slopelem", "src"))
ap.add_argument("--example", type=int, default=7)
ap.add_argument("--out", default=os.path.join(ROOT, "output", "paper_figures"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

try:
    from slopelem import SlopeModel
except ImportError:
    sys.path.insert(0, args.slopelem)
    from slopelem import SlopeModel

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "savefig.dpi": 300,
    "figure.dpi": 150, "pdf.fonttype": 42,
})
INK, MUTED, WATER = "#0b0b0b", "#52514e", "#2a78d6"
NAMES = [l["name"] for l in LAYERS]            # index 0 = Green Clay (bottom) ... 5 = Loam (top)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", name)


def load(gid):
    return SlopeModel.from_file(os.path.join(args.models, f"Feed{gid}.slim"))


def layer_bands(m):
    """x, lo[6, n], hi[6, n]: bottom and top of each layer in every 0.5 m column (nan if absent)."""
    x = np.asarray(m.column_x)
    lo = np.full((6, len(x)), np.nan); hi = np.full((6, len(x)), np.nan)
    for i, segs in enumerate(m.columns):
        for a, b, soil in segs:
            k = 7 - soil - 1                       # Slide2 soil 1 = Loam (top) -> index 5
            lo[k, i] = a if np.isnan(lo[k, i]) else min(lo[k, i], a)
            hi[k, i] = b if np.isnan(hi[k, i]) else max(hi[k, i], b)
    return x, lo, hi


ex = pd.read_excel(args.excel)
ex["fos"] = pd.to_numeric(ex["Factor of Safety (FS)"], errors="coerce")
wt = pd.read_csv(args.wt).rename(columns={"gid": "geometry_id"})
ex = ex.merge(wt[["geometry_id", "toe_depth", "meandepth"]], on="geometry_id", how="left")
ex["drawn_down"] = ex.toe_depth > 5.0            # water table well below the toe (41 models)
print(f"{len(ex)} models, FoS {ex.fos.min():.2f}-{ex.fos.max():.2f}, "
      f"{int(ex.drawn_down.sum())} with the water table drawn down below the toe")

# ================================================================ example section
m = load(args.example)
x, lo, hi = layer_bands(m)
row = ex[ex.geometry_id == args.example].iloc[0]
from matplotlib.patches import Patch
fig, ax = plt.subplots(figsize=(6.9, 2.8))
for k in range(6):
    ok = ~np.isnan(lo[k])
    ax.fill_between(x, lo[k], hi[k], where=ok, color=LAYER_COLORS[k], lw=0, alpha=0.9)
ax.plot(x, m.ground(x), color=INK, lw=0.9)
xw = np.linspace(min(m.water_x), max(m.water_x), 400)
ax.plot(xw, m.water_table(xw), color=WATER, lw=1.2, ls=(0, (5, 2)))
# overall slope angle (crest to toe) and height
y = m.ground(x); top, bot = y.max(), y.min()
xc, xt = x[y >= top - 0.01].max(), x[y <= bot + 0.01].min()
ax.plot([xc, xt], [top, bot], color=INK, lw=0.7, ls=(0, (2, 2)))
ax.text(xt + 15, bot + 8, f"α = {row.slope_angle_deg:.1f}°", fontsize=7, color=INK)
ax.annotate("", xy=(xt + 170, top), xytext=(xt + 170, bot),
            arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8))
ax.text(xt + 185, (top + bot) / 2, f"H = {top - bot:.0f} m", fontsize=7, color=MUTED, va="center")
ax.set_xlabel("x (m)"); ax.set_ylabel("Elevation (m)")
ax.set_aspect("equal"); ax.grid(False)
handles = [Patch(color=LAYER_COLORS[k], label=NAMES[k]) for k in range(5, -1, -1)]
handles += [Line2D([], [], color=WATER, lw=1.2, ls=(0, (5, 2)), label="water table"),
            Line2D([], [], color=INK, lw=0.7, ls=(0, (2, 2)), label="overall slope (crest to toe)")]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
ax.set_title(f"Model {args.example}: FoS {row.fos:.2f} (simplified Bishop, residual strength)", color=INK)
fig.tight_layout()
save(fig, "fig_section_example")

# ================================================================ property boxplots
# style of the v1.6 figure: bold panel titles, symbol + unit on the y axis, earth-tone boxes with a
# dark outline and a white median, small grey outliers, light horizontal grid only
props = [("c_peak", "Peak cohesion", "c′ (kPa)"), ("phi_peak", "Peak friction angle", "φ′ (°)"),
         ("gamma_sat", "Saturated unit weight", "γ$_{sat}$ (kN/m³)"),
         ("c_res", "Residual cohesion", "c$_r$ (kPa)"), ("phi_res", "Residual friction angle", "φ$_r$ (°)"),
         ("gamma_unsat", "Unsaturated unit weight", "γ$_{unsat}$ (kN/m³)")]
BOX_COLORS = ["#4a6741", "#6f8f55", "#9a7b2f", "#5f8f65", "#9c8468", "#cdb38f"]   # Green Clay ... Loam
with plt.rc_context({"axes.grid": False, "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
                     "xtick.labelsize": 10, "ytick.labelsize": 9}):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8))
    for ax, (p, title, ylab) in zip(axes.ravel(), props):
        data = [ex[f"layer{k + 1}_{NAMES[k].replace(' ', '_')}_{p}"].to_numpy() for k in range(6)]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                        medianprops=dict(color="white", lw=2),
                        whiskerprops=dict(color="#3d3d3d", lw=1.1), capprops=dict(color="#3d3d3d", lw=1.1),
                        boxprops=dict(edgecolor="#1f1f1f", lw=1.2),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor="#9a9993",
                                        markeredgecolor="#555555", markeredgewidth=0.5))
        for patch, col in zip(bp["boxes"], BOX_COLORS):
            patch.set_facecolor(col)
        ax.set_title(title, fontweight="bold", color=INK)
        ax.set_ylabel(ylab)
        ax.set_xticks(range(1, 7)); ax.set_xticklabels(NAMES, rotation=35, ha="right")
        ax.yaxis.grid(True, color="#e6e5e0", lw=0.7); ax.set_axisbelow(True)
    fig.suptitle(f"{len(ex)} Slide2 Models — Soil Property Distributions by Layer, Çöllolar Section 4",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(h_pad=2.5, w_pad=3)
    save(fig, "fig_property_boxplots")

# ================================================================ FoS vs slope angle
cls = [("FoS < 1.0", ex.fos < 1.0, "#e34948"),
       ("1.0 ≤ FoS < 1.3", (ex.fos >= 1.0) & (ex.fos < 1.3), "#eda100"),
       ("FoS ≥ 1.3", ex.fos >= 1.3, "#1baf7a")]
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for label, mask, col in cls:
    for dd, face in ((False, col), (True, "white")):
        mm = mask & (ex.drawn_down == dd)
        ax.scatter(ex.slope_angle_deg[mm], ex.fos[mm], s=14, color=col, facecolor=face, lw=0.8,
                   label=f"{label} (n={int(mask.sum())})" if not dd else None, zorder=3)
ax.axhline(1.0, color="#e34948", lw=0.7, ls=(0, (4, 2)))
ax.axhline(1.3, color="#eda100", lw=0.7, ls=(0, (4, 2)))
h, l = ax.get_legend_handles_labels()
h.append(Line2D([], [], marker="o", ls="", color=MUTED, markerfacecolor="white", markersize=4))
l.append(f"open: water table below the toe (n={int(ex.drawn_down.sum())})")
ax.legend(h, l, frameon=False, loc="upper right")
ax.set_xlabel("Overall slope angle α (°)"); ax.set_ylabel("FoS label (simplified Bishop)")
fig.tight_layout()
save(fig, "fig_fos_vs_angle")

# ================================================================ geometry + water-table envelope
norm = TwoSlopeNorm(vmin=ex.fos.min(), vcenter=1.0, vmax=ex.fos.max())
cmap = plt.get_cmap("RdYlGn")
fig, axes = plt.subplots(2, 1, figsize=(6.9, 4.6), sharex=True)
order = ex.sort_values("fos", ascending=False)        # draw the failing slopes last (on top)
for gid, f in zip(order.geometry_id, order.fos):
    mm = load(gid)
    xx = np.asarray(mm.column_x)
    axes[0].plot(xx, mm.ground(xx), color=cmap(norm(f)), lw=0.5, alpha=0.7)
    xw = np.linspace(min(mm.water_x), max(mm.water_x), 300)
    axes[1].plot(xw, mm.water_table(xw), color=cmap(norm(f)), lw=0.5, alpha=0.7)
axes[0].set_title("(a) Slope surfaces", loc="left", color=INK)
axes[1].set_title("(b) Water tables", loc="left", color=INK)
for ax in axes:
    ax.set_ylabel("Elevation (m)")
axes[1].set_xlabel("x (m)")
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
cb.set_label("FoS label (simplified Bishop)")
cb.set_ticks([t for t in (0.75, 1.0, 1.25, 1.5, 2.0, 2.5) if norm.vmin <= t <= norm.vmax])
cb.ax.axhline(1.0, color=INK, lw=0.8)
save(fig, "fig_geometry_envelope")

print(f"written to {args.out}")
