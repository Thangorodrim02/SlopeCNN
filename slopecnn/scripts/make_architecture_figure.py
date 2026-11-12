"""
make_architecture_figure.py
===========================
Block diagram of the two networks in the v1.6 style (dashed boxes, one column per network),
updated to v3: Soranzo et al. (2023) as published (3 inputs) on our 42 x 243 grid, and
SlopeCNN with the 7 site inputs. Shapes and parameter counts are computed from
slope_cnn/model.py, so the figure always matches the code.

    python scripts/make_architecture_figure.py

Writes output/paper_figures/fig_architectures.png/.pdf
"""

import os, sys
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from slope_cnn.model import SlopeCNN, SoranzoCNN, count_params

OUT = os.path.join(ROOT, "output", "paper_figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "savefig.dpi": 300, "font.family": "DejaVu Sans"})
H, W = 42, 243


def shp(t):
    return " × ".join(str(v) for v in t.shape[1:])


# ---- Soranzo et al. (2023) as published: 3 inputs (rho, c', phi')
r = SoranzoCNN(3, (H, W)).eval()
x = torch.zeros(1, 3, H, W); pools = []
with torch.no_grad():
    for layer in r.features:
        x = layer(x)
        if isinstance(layer, torch.nn.MaxPool2d):
            pools.append(shp(x))
SOR = [
    ("Input  3 × 42 × 243", "ρ, c′, φ′"),
    ("Conv 3×3 (32)\nLeakyReLU + MaxPool 2×2", pools[0]),
    ("Conv 3×3 (64)\nLeakyReLU + MaxPool 2×2", pools[1]),
    ("Conv 3×3 (128)\nLeakyReLU + MaxPool 2×2", pools[2]),
    ("Flatten", f"{x.numel():,}"),
    ("Fully connected\nFC(128) → Linear(1)", ""),
    ("FoS prediction", ""),
]
n_sor = count_params(r)

# ---- SlopeCNN, 7 site inputs
s = SlopeCNN(7, (H, W)).eval()
x = torch.zeros(1, 7, H, W)
with torch.no_grad():
    a = s.stem(x); b = s.pool1(a); c = s.res1(b); d = s.pool2(c); e = s.res2_conv(d)
SLO = [
    ("Input  7 × 42 × 243", "γ$_{sat}$, γ$_{unsat}$, c′, c$_r$, φ′, φ$_r$, u"),
    ("Stem\nConv 1×7 → Conv 3×3  |  BN + ReLU", shp(a)),
    ("MaxPool 2×6\nResBlock 1 + SE attention", shp(c)),
    ("MaxPool 3×4\nResBlock 2 + SE attention", shp(e)),
    ("Flatten + fill ratio", f"{e[0].numel():,} + 1"),
    ("Fully connected\nFC(64) + Dropout → Linear(1)", ""),
    ("FoS prediction", ""),
]
n_slo = count_params(s)

# ---- drawing (v1.6 layout)
FW, FH = 9.4, 7.2
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")
XC, BW, BH, GAP, TOP = [2.35, 7.05], 3.5, 0.58, 0.2, 6.55
SHOW_DETAILS = False   # True adds the grey italics: shapes, input channels, parameter counts


def column(x, title, blocks, params):
    ax.text(x, 6.95, title, ha="center", va="center", fontsize=8.8)
    ax.annotate("", xy=(x, TOP + 0.02), xytext=(x, TOP + 0.24),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black", mutation_scale=7))
    for i, (main, sub) in enumerate(blocks):
        y_top = TOP - i * (BH + GAP)
        edge = 1.4 if i in (0, len(blocks) - 1) else 0.9
        ax.add_patch(FancyBboxPatch((x - BW / 2, y_top - BH), BW, BH,
                                    boxstyle="round,pad=0,rounding_size=0.1",
                                    fc="white", ec="black", lw=edge, ls=(0, (4, 2))))
        y_mid = y_top - BH / 2
        if sub and SHOW_DETAILS:
            ax.text(x, y_mid + 0.09, main, ha="center", va="center", fontsize=7.5,
                    linespacing=1.15)
            ax.text(x, y_top - BH + 0.07, sub, ha="center", va="bottom", fontsize=6.5, color="#52514e",
                    style="italic")
        else:
            ax.text(x, y_mid, main, ha="center", va="center", fontsize=7.5,
                    linespacing=1.15)
        if i < len(blocks) - 1:
            ax.annotate("", xy=(x, y_top - BH - GAP + 0.02), xytext=(x, y_top - BH - 0.02),
                        arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black", mutation_scale=7))
    y_bot = TOP - len(blocks) * BH - (len(blocks) - 1) * GAP
    ax.text(x, y_bot - 0.25, params, ha="center", va="center", fontsize=8.5)
    return y_bot


column(XC[0], "Soranzo et al. (2023) — Baseline", SOR, f"Parameters: {n_sor:,}")
y_bot = column(XC[1], "SlopeCNN — Proposed", SLO, f"Parameters: {n_slo:,}")
ax.plot([FW / 2, FW / 2], [y_bot - 0.45, 7.15],
        color="#9a9993", lw=0.7, ls=(0, (3, 3)))
if SHOW_DETAILS:
    ax.text(FW / 2, y_bot - 0.75, "grey italics: output shape (channels × rows × columns) or input channels",
            ha="center", va="center", fontsize=6.5, color="#52514e", style="italic")

y_low = y_bot - (1.0 if SHOW_DETAILS else 0.5)
ax.set_ylim(y_low, FH); fig.set_size_inches(FW, FH - y_low)   # crop unused space below the boxes
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_architectures.{ext}"), bbox_inches="tight", facecolor="white")
print(f"Soranzo et al. (2023) {n_sor:,} params, SlopeCNN {n_slo:,} params ->",
      os.path.join(OUT, "fig_architectures.png"))
