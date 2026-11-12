"""
make_geometry_schematic.py
==========================
The v1.6 "Random Slope Generation" figure, redrawn from the v3 data. Everything in it is read
from the 346 Slide2 models, not from the old generator:

  * one real model is drawn (benched face, six layers, its water table)
  * the five layer boundaries B1-B5 are straight lines in Slide2; each one is fitted in every
    model and the yellow wedge shows the section dip +/- 3 deg (the range used for the 346 models)
  * the slope angle range, L = H / tan(alpha) range and H are taken from the models

    python scripts/make_geometry_schematic.py                (picks a typical model, alpha ~ 18 deg)
    python scripts/make_geometry_schematic.py --example 7

Writes output/paper_figures/fig_geometry_schematic.png/.pdf
The boundary fits are cached in output/tensors_v3/boundary_fits.csv (delete it to refit).
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)                # repository root
ap = argparse.ArgumentParser()
ap.add_argument("--models", default=os.path.join(REPO, "data", "slide2_models"))
ap.add_argument("--excel", default=os.path.join(REPO, "data", "soil_properties_346geom_v3_cnn_input.xlsx"))
ap.add_argument("--wt", default=os.path.join(ROOT, "output", "tensors_v3", "wt_depth.csv"))
ap.add_argument("--slopelem", default=os.path.join(REPO, "slopelem", "src"))
ap.add_argument("--example", type=int, default=None)
ap.add_argument("--out", default=os.path.join(ROOT, "output", "paper_figures"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
try:
    from slopelem import SlopeModel
except ImportError:
    sys.path.insert(0, args.slopelem)
    from slopelem import SlopeModel

# Slide2 soil 1 = Loam (top) ... soil 6 = Green Clay (bottom); boundary Bj lies between soil j and j+1
NAMES = {1: "Loam", 2: "Blue Clay", 3: "Gyttja", 4: "Lignite", 5: "Black Clay", 6: "Green Clay"}
FILL = {1: "#e8ddb4", 2: "#d3c47c", 3: "#adc47b", 4: "#86aa55", 5: "#678c44", 6: "#90a978"}
BAND, BLINE = "#e2c51c", "#4a4a4a"
C_ALPHA, C_H, C_L, C_WT = "#b8731e", "#1f5fa8", "#4b3fb0", "#2a78d6"


def load(gid):
    return SlopeModel.from_file(os.path.join(args.models, f"Feed{gid}.slim"))


def frame(m):
    """x relative to the toe, ground relative to the toe, toe x, toe elevation, crest x."""
    x = np.asarray(m.column_x); y = m.ground(x)
    bot, top = y.min(), y.max()
    xt = x[y <= bot + 0.01].min(); xc = x[y >= top - 0.01].max()
    return x, y, xt, bot, xc


def merged(segs):
    out = []
    for a, b, s in sorted(segs):
        if out and out[-1][2] == s:
            out[-1][1] = b
        else:
            out.append([a, b, s])
    return out


def boundary_points(m, x, xt, bot, j):
    """Points on boundary Bj (top of soil j+1 under soil j), relative to the toe."""
    X, Y = [], []
    for i, segs in enumerate(m.columns):
        mg = merged(segs)
        for (a1, b1, s1), (a2, b2, s2) in zip(mg[:-1], mg[1:]):
            if s1 == j + 1 and s2 == j and b1 - a1 > 0.3 and b2 - a2 > 0.3:
                X.append(x[i] - xt); Y.append(b1 - bot)
    return np.array(X), np.array(Y)


# ---------------------------------------------------------------- fit B1-B5 in all 346 models
ex = pd.read_excel(args.excel)
wt = pd.read_csv(args.wt).rename(columns={"gid": "geometry_id"})
ex = ex.merge(wt[["geometry_id", "toe_depth"]], on="geometry_id", how="left")
cache = os.path.join(ROOT, "output", "tensors_v3", "boundary_fits.csv")
if os.path.exists(cache):
    fits = pd.read_csv(cache)
else:
    rows = []
    for n, gid in enumerate(ex.geometry_id.astype(int)):
        m = load(gid); x, y, xt, bot, xc = frame(m)
        for j in range(1, 6):
            X, Y = boundary_points(m, x, xt, bot, j)
            if len(X) > 10:
                p = np.polyfit(X, Y, 1)
                rows.append(dict(gid=gid, b=j, slope=p[0], icpt=p[1], x0=X.min(), x1=X.max(),
                                 dip=np.degrees(np.arctan(p[0])),
                                 resid=np.abs(Y - np.polyval(p, X)).max()))
        if n % 50 == 0:
            print(f"  fitted boundaries in {n}/{len(ex)} models")
    fits = pd.DataFrame(rows); fits.to_csv(cache, index=False)

# ---------------------------------------------------------------- example model
if args.example is None:          # typical: alpha 17-20 deg, crest-to-toe water table, clean boundaries
    clean = fits.groupby("gid").agg(n=("b", "size"), r=("resid", "max"))
    ok = clean[(clean.n == 5) & (clean.r < 0.5)].index
    cand = ex[ex.geometry_id.isin(ok) & ex.slope_angle_deg.between(17, 20) & (ex.toe_depth <= 5)].copy()
    med = fits.groupby("b").dip.median()
    cand["dev"] = [sum(abs(fits[(fits.gid == g) & (fits.b == j)].dip.iloc[0] - med[j]) for j in range(1, 6))
                   for g in cand.geometry_id]
    args.example = int(cand.sort_values("dev").geometry_id.iloc[0])
row = ex[ex.geometry_id == args.example].iloc[0]
alpha = float(row.slope_angle_deg)
m = load(args.example); x, y, xt, bot, xc = frame(m)
X = x - xt; G = y - bot
H = G.max(); xcr = xc - xt; ybase = m.y_min - bot
a_lo, a_hi = ex.slope_angle_deg.min(), ex.slope_angle_deg.max()
L_lo, L_hi = H / np.tan(np.radians(a_hi)), H / np.tan(np.radians(a_lo))
print(f"example model {args.example}: alpha {alpha:.1f} deg, FoS {row['Factor of Safety (FS)']}")

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "savefig.dpi": 300})
fig, ax = plt.subplots(figsize=(13, 6.6))

# layers of the example model (0.5 m columns)
lo = {s: np.full(len(X), np.nan) for s in NAMES}; hi = {s: np.full(len(X), np.nan) for s in NAMES}
for i, segs in enumerate(m.columns):
    for a, b, s in segs:
        lo[s][i] = np.nanmin([lo[s][i], a - bot]); hi[s][i] = np.nanmax([hi[s][i], b - bot])
for s in NAMES:
    ax.fill_between(X, lo[s], hi[s], where=~np.isnan(lo[s]), color=FILL[s], lw=0, zorder=1)

# each boundary: this model's line (dashed) and the +/-3 deg range about the section dip (yellow wedge,
# pivoting at the left edge as in the v1.6 figure; section dip = median dip over the 346 models)
sec = fits.groupby("b").dip.median()
_xx = X[X <= 0.0]
own = {int(r.b): r.icpt + r.slope * _xx for r in fits[fits.gid == args.example].itertuples()}
LAB = {1: (-545, 3), 2: (-470, -3), 3: (-450, -3), 4: (-330, 3), 5: (120, -4)}   # label x, offset (m)
for j in range(1, 6):
    me = fits[(fits.gid == args.example) & (fits.b == j)]
    if not len(me):
        continue
    me = me.iloc[0]
    xx = X[X <= 0.0]; y0 = me.icpt + me.slope * X[0]     # wedge runs to the face / toe
    up = y0 + np.tan(np.radians(sec[j] + 3)) * (xx - X[0])
    dn = y0 + np.tan(np.radians(sec[j] - 3)) * (xx - X[0])
    gg = G[X <= 0.0]
    # no overlap: each wedge stops at 40 % of the distance to the neighbouring boundary of this
    # model, leaving a visible gap between neighbouring wedges
    if j - 1 in own:
        up = np.minimum(up, own[j] + 0.4 * (own[j - 1] - own[j]))
    if j + 1 in own:
        dn = np.maximum(dn, own[j] - 0.4 * (own[j] - own[j + 1]))
    w_lo, w_hi = np.minimum(dn, gg), np.minimum(up, gg)
    ax.fill_between(xx, w_lo, w_hi, where=w_hi > w_lo + 1e-6, color=BAND, alpha=0.45, lw=0, zorder=2)
    xb = X[X <= me.x1]
    ax.plot(xb, me.icpt + me.slope * xb, color=BLINE, lw=1.3, ls=(0, (4, 2)), zorder=4)
    lx, dy = LAB[j]
    ax.text(lx, me.icpt + me.slope * lx + dy, f"B{j} ({me.dip:+.1f}°)", ha="center",
            va="bottom" if dy > 0 else "top", color="white", fontweight="bold", fontsize=10.5, zorder=6)

# layer names at the left edge, mid-thickness
for s in NAMES:
    k = np.where(~np.isnan(lo[s]))[0]
    if len(k):
        k0 = k[0] + 40      # 20 m in from the left edge
        ax.text(X[k0], (lo[s][k0] + hi[s][k0]) / 2, NAMES[s], color="white", fontweight="bold",
                fontsize=10.5, va="center", ha="left", zorder=6)

# ground outline and model boundary
ax.plot(np.r_[X[0], X, X[-1], X[0], X[0]], np.r_[ybase, G, ybase, ybase, ybase],
        color="#1a1a1a", lw=2.0, zorder=5)

# water table of this model
xw = np.linspace(m.water_x.min(), m.water_x.max(), 600)
ax.plot(xw - xt, m.water_table(xw) - bot, color=C_WT, lw=1.6, ls=(0, (6, 2)), zorder=5)
xwl = xcr + 0.25 * (0 - xcr)
ax.text(xwl, m.water_table(xwl + xt) - bot - 6, "water table", color=C_WT, fontsize=10,
        ha="center", va="top", fontweight="bold", zorder=6,
        rotation=np.degrees(np.arctan(-H / -xcr)) * 0.0)

# overall slope (crest to toe) and alpha at the toe
ax.plot([xcr, 0], [H, 0], color=C_ALPHA, lw=1.1, ls=(0, (2, 2)), zorder=5)
ax.plot([-0.16 * abs(xcr) * 1.4, 0], [0, 0], color=C_ALPHA, lw=1.2, ls=":", zorder=5)
r = 0.16 * abs(xcr)
t = np.linspace(np.pi - np.radians(alpha), np.pi, 30)
ax.plot(r * np.cos(t), r * np.sin(t) * (1.0), color=C_ALPHA, lw=2, zorder=6)
ax.text(-r * 0.95, r * np.sin(np.radians(alpha)) + 10, f"α = {alpha:.1f}°", color=C_ALPHA,
        fontsize=12, ha="center", zorder=6)
ax.text(X[-1] - 20, ybase + 25, f"α ∈ U({a_lo:.0f}°, {a_hi:.0f}°)", color="white", fontsize=11,
        ha="right", zorder=6)

# H arrow (right of the toe)
xh = 0.55 * X[-1]
ax.annotate("", xy=(xh, H), xytext=(xh, 0),
            arrowprops=dict(arrowstyle="<|-|>", color=C_H, lw=1.8, mutation_scale=12), zorder=6)
ax.plot([xh - 12, xh + 12], [H, H], color=C_H, lw=1.2, zorder=6)
ax.text(xh + 18, H / 2, f"H = {H:.0f} m\nfive ~30 m faces,\nfour benches", color=C_H, fontsize=11,
        fontweight="bold", va="center", ha="left", zorder=6)

# L = H / tan(alpha) below the model
yL = ybase - 18
for xv in (xcr, 0):
    ax.plot([xv, xv], [ybase - 8, yL - 6], color=C_L, lw=1.2, zorder=6, clip_on=False)
ax.annotate("", xy=(0, yL), xytext=(xcr, yL),
            arrowprops=dict(arrowstyle="<|-|>", color=C_L, lw=1.6, mutation_scale=10), zorder=6,
            annotation_clip=False)
ax.text(xcr / 2, yL - 8, f"$L = H/\\tan\\alpha$ = {abs(xcr):.0f} m here  "
        f"({L_lo:.0f}–{L_hi:.0f} m for $\\alpha \\in$ {a_lo:.0f}°–{a_hi:.0f}°)",
        color=C_L, fontsize=11, ha="center", va="top", zorder=6, clip_on=False)

ax.set_xlim(X[0] - 25, X[-1] + 25); ax.set_ylim(ybase - 55, H + 25)
ax.set_xlabel("Horizontal distance from the toe (m)", fontsize=12)
ax.set_ylabel("Elevation above the toe (m)", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#e6e5e0", lw=0.5, zorder=0)
dmin, dmax = fits.dip.min(), fits.dip.max()
ax.set_title(f"Slope Geometry of Model {args.example} — Çöllolar Open-Pit Mine, Section 4\n"
             f"6 soil layers · 5 boundaries (B1–B5) · α ∈ U({a_lo:.0f}°, {a_hi:.0f}°) · "
             "boundary dip = section dip ± 3°",
             fontsize=12.5, fontweight="bold")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(args.out, f"fig_geometry_schematic.{ext}"), bbox_inches="tight",
                facecolor="white")
print(os.path.join(args.out, "fig_geometry_schematic.png"))
