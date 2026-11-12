"""
make_pipeline_flowchart.py
==========================
Pipeline flowchart for the v3 SlopeCNN study, drawn in the same layout as the
v1.x figure "Complete training and validation pipeline" (three columns,
START / END ovals, decision diamond, fold loop). Edit the TEXT dict and rerun.

    python scripts/make_pipeline_flowchart.py

Writes output/paper_figures/fig_pipeline_flowchart.png / .pdf / .svg
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "paper_figures")
if not os.path.isdir(os.path.join(ROOT, "output")):      # run outside the repo
    OUT = os.path.join(os.getcwd(), "paper_figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
                     "font.family": "DejaVu Sans"})

# ------------------------------------------------------------------ text
TEXT = dict(
    head1="①  Data generation",
    head2="②  Training pipeline",
    head3="③  Results",
    geom="Geometry generation\nSection 4, α = 5–25°, varied layer dips\nwater table drawn in each model",
    soil="Soil property sampling\n6 properties × 6 layers, ±20 % of base\n(γ$_{sat}$ ≥ γ$_{unsat}$, residual ≤ peak)",
    fos="FoS computation (simplified Bishop)\nresidual along bedding, peak across\nslopelem solver, checked against Slide2",
    decide="346\nmodels?",
    tensor="Tensor construction\n7 × 42 × 243\n6 soil properties + pore pressure",
    split="Stratified k-fold split  k ∈ {5, 7}",
    train_set="Training\nslopes",
    val_set="Validation\n(15 %)",
    test_set="Test\nfold",
    aug="Data augmentation ×18\n(training slopes only)",
    train="Model training\nSlopeCNN and Soranzo et al. (2023)\nearly stopping on validation",
    pred="FoS prediction\n(test fold)",
    evalf="Fold evaluation\nR²   RMSE   MAE",
    loop="repeat\nnext fold",
    held="held out",
    agg="Aggregate across folds × seeds\nmean ± std",
    kcmp="k ∈ {5, 7}\ncomparison",
    mcmp="Model comparison\nSlopeCNN vs Soranzo et al. (2023)",
)

# ------------------------------------------------------------------ canvas (units ≈ pixels, y down)
W, H = 1380, 900
fig = plt.figure(figsize=(W / 100, H / 100))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
LW, FS, FS_HEAD = 1.0, 8.6, 10.5
BW = 350
X1, X2, X3 = 233, 690, 1147
SEP1, SEP2 = 461, 918


def box(xc, yc, text, h=68, w=BW, dashed=False, fs=FS):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=4", fc="white", ec="black",
                                lw=LW, ls=(0, (3, 2)) if dashed else "-"))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, linespacing=1.3)
    return dict(l=xc - w / 2, r=xc + w / 2, t=yc - h / 2, b=yc + h / 2, x=xc, y=yc)


def oval(xc, yc, text, w=185, h=50):
    ax.add_patch(Ellipse((xc, yc), w, h, fc="white", ec="black", lw=1.6))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=9.5)
    return dict(t=yc - h / 2, b=yc + h / 2, x=xc, y=yc)


def diamond(xc, yc, text, w=272, h=80):
    ax.add_patch(Polygon([(xc, yc - h / 2), (xc + w / 2, yc), (xc, yc + h / 2), (xc - w / 2, yc)],
                         closed=True, fc="white", ec="black", lw=LW))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=FS, linespacing=1.2)
    return dict(l=xc - w / 2, r=xc + w / 2, t=yc - h / 2, b=yc + h / 2, x=xc, y=yc)


def arrow(p, q, dashed=False):
    ax.annotate("", xy=q, xytext=p,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=LW, mutation_scale=9,
                                shrinkA=0, shrinkB=0, ls=(0, (3, 2)) if dashed else "-"))


def path(points, dashed=False):
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color="black", lw=LW, ls=(0, (3, 2)) if dashed else "-",
            solid_capstyle="butt")
    arrow(points[-2], points[-1], dashed)


# headers and separators
for x, t in ((X1, "head1"), (X2, "head2"), (X3, "head3")):
    ax.text(x, 48, TEXT[t], ha="center", va="center", fontsize=FS_HEAD, fontweight="bold")
for x in (SEP1, SEP2):
    ax.plot([x, x], [35, 860], color="#9a9a9a", lw=0.8, ls=(0, (4, 3)))

# ---------------- column 1: data generation
start = oval(X1, 125, "START")
geom = box(X1, 208, TEXT["geom"], h=74)
soil = box(X1, 304, TEXT["soil"], h=74)
fos = box(X1, 400, TEXT["fos"], h=74)
dec = diamond(X1, 506, TEXT["decide"])
ten = box(X1, 622, TEXT["tensor"], h=74)
for a, b in ((start, geom), (geom, soil), (soil, fos), (fos, dec), (dec, ten)):
    arrow((X1, a["b"]), (X1, b["t"]))
ax.text(X1 + 10, (dec["b"] + ten["t"]) / 2, "Yes", ha="left", va="center", fontsize=8)
path([(dec["l"], dec["y"]), (40, dec["y"]), (40, geom["y"]), (geom["l"], geom["y"])])
ax.text(22, (geom["y"] + dec["y"]) / 2, "No", ha="center", va="center", fontsize=8)

# ---------------- column 2: training pipeline
split = box(X2, 125, TEXT["split"], dashed=True)
sw, sg = 108, 13
row_y, row_h = 222, 58
trs = box(X2 - sw - sg, row_y, TEXT["train_set"], h=row_h, w=sw, dashed=True, fs=8.2)
vas = box(X2, row_y, TEXT["val_set"], h=row_h, w=sw, dashed=True, fs=8.2)
tes = box(X2 + sw + sg, row_y, TEXT["test_set"], h=row_h, w=sw, dashed=True, fs=8.2)
aug = box(515 + 105, 322, TEXT["aug"], w=210)
trn = box(X2, 420, TEXT["train"], h=74)
prd = box(X2, 520, TEXT["pred"])
evl = box(X2, 616, TEXT["evalf"])
arrow((X2, split["b"]), (X2, vas["t"]))
arrow((trs["x"], split["b"]), (trs["x"], trs["t"]))
arrow((tes["x"], split["b"]), (tes["x"], tes["t"]))
arrow((trs["x"], trs["b"]), (trs["x"], aug["t"]))
# validation slopes skip augmentation and go straight to training (early stopping)
vx = vas["r"] - 12
arrow((vx, vas["b"]), (vx, trn["t"]))
arrow((aug["x"], aug["b"]), (aug["x"], trn["t"]))
for a, b in ((trn, prd), (prd, evl)):
    arrow((X2, a["b"]), (X2, b["t"]))
# test fold bypasses augmentation and training
rx = 890
path([(tes["r"], tes["y"]), (rx, tes["y"]), (rx, prd["y"]), (prd["r"], prd["y"])], dashed=True)
ax.text(rx + 5, (aug["y"] + trn["y"]) / 2, TEXT["held"], rotation=90, ha="left", va="center",
        fontsize=7.4, style="italic")
# tensor -> split ; fold loop
path([(ten["r"], ten["y"]), (SEP1, ten["y"]), (SEP1, split["y"]), (split["l"], split["y"])])
lx = 503
ax.plot([evl["l"], lx, lx], [evl["y"], evl["y"], split["y"]], color="black", lw=LW)
ax.text((SEP1 + lx) / 2, 372, TEXT["loop"], ha="center", va="center", fontsize=7.8,
        style="italic", linespacing=1.15,
        bbox=dict(fc="white", ec="none", pad=1.5))

# ---------------- column 3: results
agg = box(X3, 125, TEXT["agg"], h=78)
kc = box(X3, 290, TEXT["kcmp"], h=78)
mc = box(X3, 455, TEXT["mcmp"], h=78)
end = oval(X3, 618, "END")
for a, b in ((agg, kc), (kc, mc), (mc, end)):
    arrow((X3, a["b"]), (X3, b["t"]))
path([(evl["r"], evl["y"]), (SEP2, evl["y"]), (SEP2, agg["y"]), (agg["l"], agg["y"])])

ax.set_ylim(700, 10)                                   # crop unused space at the bottom
fig.set_size_inches(W / 100, 690 / 100)
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(OUT, f"fig_pipeline_flowchart.{ext}"), facecolor="white")
print(os.path.join(OUT, "fig_pipeline_flowchart.png"))
