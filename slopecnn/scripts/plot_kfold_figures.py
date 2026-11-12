"""
plot_kfold_figures.py
=====================
The two v1.6 k-fold figures, redrawn from the v3 runs with the same layout and
settings as the original scripts, for 5-fold AND 7-fold:

  1. fold_comparison_<run>_<k>fold.png   one per model and k
     left: train / validation MSE of the best fold and of the worst-converging fold
     right: validation R2 and RMSE of every fold; best fold bold with a star
     (paper figures: no test scores on the validation curves; test scores are in Table 2)
     smoothing: moving average, window 7, edge padding
  2. cv_comparison.png                   
     top: best-fold validation loss, R2 and RMSE of every model and k (rolling mean, window 15;
          colour = model, solid = 5-fold, dashed = 7-fold)
     bottom: best-fold scatter of every model and k, 1:1 line and +/-30 % band, train/val/test counts

What changed because v3 has 3 seeds per fold and a separate test fold:
  * a fold's curve is the mean of its 3 seeds (epoch by epoch, while at least 2 seeds run);
    its R2 / RMSE / MAE are the mean of the 3 seeds' TEST-fold scores;
  * "Val" curves are the inner validation split (15 %), which chose the checkpoint;
    the test fold is scored once, after training;
  * best fold = highest mean test R2; worst fold = earliest best epoch (as in the original).

    python scripts/plot_kfold_figures.py               (5- and 7-fold, whatever has finished)
    python scripts/plot_kfold_figures.py --k 5         (one k only)
"""

import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from slope_cnn.cv import make_folds

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, nargs="+", default=[5, 7])
ap.add_argument("--runs5", default=os.path.join(ROOT, "output", "runs"))
ap.add_argument("--runs7", default=os.path.join(ROOT, "output", "runs_k7"))
ap.add_argument("--window", type=int, default=7, help="smoothing of the fold-comparison figures")
ap.add_argument("--window_cv", type=int, default=15, help="smoothing of the CV-comparison figure")
ap.add_argument("--out", default=os.path.join(ROOT, "output", "paper_figures"))
args = ap.parse_args()
RUNS = {5: args.runs5, 7: args.runs7}
os.makedirs(args.out, exist_ok=True)

MODELS = [  # run folder, label, colour (colours of the original plot_comparison.py)
    ("soranzo_published", "Soranzo et al. (2023)", "#BA7517"),
    ("slopecnn_all6_pp", "SlopeCNN", "#185FA5"),
]
LINESTYLE = {5: "-", 7: "--"}
FOLD_COLORS = ['#c62828', '#185FA5', '#534AB7', '#0F6E56', '#FF6B35',
               '#CC7A00', '#D81B60', '#1D9E75', '#7B3F9E', '#888780']   # F7 pink, not a second red

fos_all = np.load(os.path.join(ROOT, "output", "tensors_v3", "tensors.npz"))["fos"].astype(float)
N = len(fos_all)


def smooth(y, window=7):                       # plot_fold_comparison.py
    y = np.asarray(y, float)
    if len(y) < window:
        return y
    pad = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(pad, np.ones(window) / window, mode="valid")[:len(y)]


def smooth_cv(arr, w):                         # plot_comparison.py
    return pd.Series(np.asarray(arr, float)).rolling(window=w, center=True, min_periods=1).mean().to_numpy()


def score(y, p):
    e = p - y
    return {"r2": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2),
            "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e)))}


def load(run, k):
    """{fold: {'hist': DataFrame(epoch, train_mse, val_mse, val_r2, val_rmse), 'res': {...}, 'pred': DataFrame}}"""
    d = os.path.join(RUNS[k], run)
    folds = make_folds(fos_all, k)
    out = {}
    for f in range(1, k + 1):
        js = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, f"fold{f}_seed*.json")))]
        hs = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(d, f"history_fold{f}_seed*.csv")))]
        ps = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(d, f"pred_fold{f}_seed*.csv")))]
        if not (js and hs and ps):
            continue
        h = pd.concat(hs).groupby("epoch")
        keep = h.size().index[h.size() >= min(2, len(hs))]
        hist = h[["train_mse", "val_mse"]].mean().loc[keep].reset_index()
        yv = fos_all[folds[f - 1][1]]
        hist["val_r2"] = 1 - hist.val_mse / np.mean((yv - yv.mean()) ** 2)
        hist["val_rmse"] = np.sqrt(hist.val_mse)
        pred = pd.concat(ps).groupby("geometry_id").mean().reset_index()   # mean of the seeds per slope
        res = {m: float(np.mean([j[m] for j in js])) for m in ("r2", "rmse", "mae")}
        res["best_ep"] = float(np.mean([j["best_epoch"] for j in js]))
        res.update(n_train=js[0]["n_train"], n_val=js[0]["n_val"], n_test=js[0]["n_test"], seeds=len(js))
        out[f] = {"hist": hist, "res": res, "pred": pred}
    return out


DATA = {}
for k in args.k:
    for run, _, _ in MODELS:
        F = load(run, k)
        if F:
            DATA[(run, k)] = F
        else:
            print(f"  no finished runs for {run}, k={k} in {RUNS[k]}: skipped")
if not DATA:
    sys.exit("nothing to plot")

# ===================================================== 1. fold comparison, one figure per model and k
for (run, k), F in DATA.items():
    label = dict((m[0], m[1]) for m in MODELS)[run]
    r2s = {i: F[i]["res"]["r2"] for i in F}
    beps = {i: F[i]["res"]["best_ep"] for i in F}
    best = max(r2s, key=r2s.get)
    # worst = earliest best epoch (as in the original); never the best fold itself, or the two panels repeat
    worst = min((i for i in beps if i != best), key=beps.get) if len(beps) > 1 else best
    W = args.window
    fig = plt.figure(figsize=(13, 10))
    fig.suptitle(f"{k}-Fold Cross Validation — {label} | {N} geometries, 3 seeds per fold\n"
                 f"Best Fold {best} (R²={r2s[best]:.3f}) vs Worst Fold {worst} (R²={r2s[worst]:.3f})",
                 fontsize=12, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)
    for row, (fid, name) in enumerate([(best, "Best Fold"), (worst, "Worst Fold")]):
        h, r = F[fid]["hist"], F[fid]["res"]
        ax = fig.add_subplot(gs[row, 0])
        ax.plot(h.epoch, smooth(h.train_mse, W), color="#185FA5", lw=2.0, label="Train")
        ax.plot(h.epoch, smooth(h.val_mse, W), color="#c62828", lw=2.0, ls="--", label="Val")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
        ax.set_title(f"Fold {fid} ({name}) — MSE\n"
                     f"Test R²={r['r2']:.4f}  RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}")
        ax.legend(fontsize=9, loc="upper right", frameon=False); ax.grid(alpha=0.3)
    for row, (key, ylab, metric) in enumerate([("val_r2", "Val R²", "r2"), ("val_rmse", "Val RMSE", "rmse")]):
        ax = fig.add_subplot(gs[row, 1])
        vals = [F[i]["res"][metric] for i in sorted(F)]
        tag = "R²" if metric == "r2" else "RMSE"
        for i in sorted(F):
            h = F[i]["hist"]
            ax.plot(h.epoch, smooth(h[key], W), color=FOLD_COLORS[(i - 1) % len(FOLD_COLORS)],
                    lw=2.5 if i == best else 1.2, alpha=1.0 if i == best else 0.45,
                    label=f"F{i}  {F[i]['res'][metric]:.3f}" + (" ★" if i == best else ""))
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylab)
        ax.set_title(f"{ylab} — All {k} Folds  (Mean test {tag}={np.mean(vals):.3f}±{np.std(vals):.3f})")
        # legend outside the axes, so it never covers a curve
        ax.legend(fontsize=8, title=f"test {tag}", title_fontsize=8, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), frameon=False, borderaxespad=0)
        ax.grid(alpha=0.3)
    path = os.path.join(args.out, f"fold_comparison_{run}_{k}fold.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"{label}, k={k}: best fold {best} (R2 {r2s[best]:.4f}), worst-converging fold {worst} -> {path}")

# ===================================================== 1b. 5-fold and 7-fold together, best fold only
# one row per k: best-fold train/val MSE | val R2 of every fold | val RMSE of every fold;
# one fold legend per row, outside the panels on the right (test R2 and RMSE of each fold)
ks = [k for k in args.k if any((run, k) in DATA for run, _, _ in MODELS)]
if len(ks) > 1:
    for run, label, _ in MODELS:
        rows = [k for k in ks if (run, k) in DATA]
        if len(rows) < 2:
            continue
        W = args.window
        fig = plt.figure(figsize=(17, 4.6 * len(rows)))
        heads = []
        for k in rows:
            F = DATA[(run, k)]
            b = max(F, key=lambda i: F[i]["res"]["r2"])
            heads.append(f"{k}-fold: best fold {b}")
        fig.suptitle(f"5- and 7-Fold Cross Validation — {label} | {N} geometries, 3 seeds per fold\n"
                     + "   |   ".join(heads), fontsize=12, fontweight="bold")
        gs = gridspec.GridSpec(len(rows), 3, figure=fig, hspace=0.5, wspace=0.28)
        for r_i, k in enumerate(rows):
            F = DATA[(run, k)]
            best = max(F, key=lambda i: F[i]["res"]["r2"])
            h, r = F[best]["hist"], F[best]["res"]
            ax = fig.add_subplot(gs[r_i, 0])
            ax.plot(h.epoch, smooth(h.train_mse, W), color="#185FA5", lw=2.0, label="Train")
            ax.plot(h.epoch, smooth(h.val_mse, W), color="#c62828", lw=2.0, ls="--", label="Val")
            ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
            ax.set_title(f"{k}-fold, Fold {best} (Best Fold) — MSE")
            ax.legend(fontsize=9, loc="upper right", frameon=False); ax.grid(alpha=0.3)
            for c_i, (key, ylab, metric) in enumerate([("val_r2", "Val R²", "r2"), ("val_rmse", "Val RMSE", "rmse")]):
                ax = fig.add_subplot(gs[r_i, 1 + c_i])
                vals = [F[i]["res"][metric] for i in sorted(F)]
                tag = "R²" if metric == "r2" else "RMSE"
                for i in sorted(F):
                    hh = F[i]["hist"]
                    ax.plot(hh.epoch, smooth(hh[key], W), color=FOLD_COLORS[(i - 1) % len(FOLD_COLORS)],
                            lw=2.5 if i == best else 1.2, alpha=1.0 if i == best else 0.45,
                            label=f"F{i}" + (" ★ best" if i == best else ""))
                ax.set_xlabel("Epoch"); ax.set_ylabel(ylab)
                ax.set_title(f"{k}-fold, {ylab} — All {k} Folds")
                ax.grid(alpha=0.3)
                if c_i == 1:                       # one legend per row, right of the last panel
                    ax.legend(fontsize=8, title=f"{k}-fold", title_fontsize=8, loc="upper left",
                              bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0)
        path = os.path.join(args.out, f"fold_comparison_{run}_5and7fold.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
        print(f"{label}: 5- and 7-fold best folds -> {path}")

# ===================================================== 2. CV comparison of every model and k
order = [(run, k) for k in args.k for run, _, _ in MODELS if (run, k) in DATA]
n = len(order)
W = args.window_cv
fig = plt.figure(figsize=(max(18, 4.6 * n), 10), facecolor="white")
fig.suptitle("Cross-Validation Analysis - Soranzo et al. (2023) vs SlopeCNN",
             fontsize=13, fontweight="bold", y=1.05)
cols = 3 * n                                   # top: 3 panels of n columns; bottom: n panels of 3 columns
gs = gridspec.GridSpec(2, cols, figure=fig, hspace=0.48, wspace=1.2)
tops = [fig.add_subplot(gs[0, i * n:(i + 1) * n]) for i in range(3)]
meta = {m[0]: (m[1], m[2]) for m in MODELS}
for run, k in order:
    F = DATA[(run, k)]
    label, col = meta[run]
    bf = max(F, key=lambda i: F[i]["res"]["r2"])
    h = F[bf]["hist"]
    lbl = f"{label}, k={k}  (best fold {bf})"
    for ax, key in zip(tops, ("val_mse", "val_r2", "val_rmse")):
        ax.plot(h.epoch, smooth_cv(h[key], W), color=col, lw=2, ls=LINESTYLE.get(k, "-"), label=lbl)
for ax, ylabel, title in zip(tops, ("Validation MSE", "Validation R2", "Validation RMSE"),
                             ("Best Fold - Validation Loss", "Best Fold - Validation R2",
                              "Best Fold - Validation RMSE")):
    ax.set_xlabel("Epoch", fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
# one legend for the whole top row, above the panels, so no curve is covered
h_, l_ = tops[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))

for j, (run, k) in enumerate(order):
    F = DATA[(run, k)]
    label, col = meta[run]
    bf = max(F, key=lambda i: F[i]["res"]["r2"])
    ax = fig.add_subplot(gs[1, 3 * j:3 * j + 3])
    p, r = F[bf]["pred"], F[bf]["res"]
    yt, yp = p.fos_true.to_numpy(), p.fos_pred.to_numpy()
    s = score(yt, yp)
    lim = [min(yt.min(), yp.min()) - 0.1, max(yt.max(), yp.max()) + 0.1]
    xx = np.linspace(lim[0], lim[1], 200)
    ax.fill_between(xx, xx * 0.70, xx * 1.30, color="gray", alpha=0.12, label="+/-30%")
    ax.scatter(yt, yp, s=22, alpha=0.65, color=col, edgecolors="none",
               marker="o" if k == 5 else "D")
    ax.plot(xx, xx, "k--", lw=1.2, label="1:1")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"{label}  k={k} - Best Fold {bf}",
                 fontsize=10, fontweight="bold")
    ax.text(0.98, 0.04, f"Train: {r['n_train']}\nVal: {r['n_val']}\nTest: {r['n_test']}",
            transform=ax.transAxes, fontsize=8, color="#555555", ha="right", va="bottom",
            bbox=dict(fc="white", ec="#cccccc", boxstyle="round,pad=0.3", lw=0.8))
    ax.set_xlabel("Actual FoS", fontsize=9)
    ax.set_ylabel("Predicted FoS (mean of 3 seeds)", fontsize=9)
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

name = "cv_comparison.png" if len(args.k) > 1 else f"cv_comparison_{args.k[0]}fold.png"
path = os.path.join(args.out, name)
plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white"); plt.close()
print(f"saved {path}")

# ===================================================== 3. table of the average results
PARAMS = {"soranzo_published": "2,551,105", "slopecnn_all6_pp": "124,343"}
rows = []
for run, k in order:
    F = DATA[(run, k)]
    fr = pd.DataFrame([F[i]["res"] for i in sorted(F)])
    # every slope is in exactly one test fold: pool the (seed-averaged) predictions of all folds
    pooled = pd.concat([F[i]["pred"] for i in F])
    y, p = pooled.fos_true.to_numpy(), pooled.fos_pred.to_numpy()
    o = score(y, p)
    rows.append({
        "Model": meta[run][0], "k": k,
        "Trainings": int(fr.seeds.sum()),
        "R² (mean ± sd of folds)": f"{fr.r2.mean():.3f} ± {fr.r2.std(ddof=0):.3f}",
        "RMSE": f"{fr.rmse.mean():.3f} ± {fr.rmse.std(ddof=0):.3f}",
        "MAE": f"{fr.mae.mean():.3f} ± {fr.mae.std(ddof=0):.3f}",
        "Best fold R²": f"{fr.r2.max():.3f}", "Worst fold R²": f"{fr.r2.min():.3f}",
        "Pooled R² (all 346)": f"{o['r2']:.3f}",
    })
tab = pd.DataFrame(rows)
# fold-paired difference, SlopeCNN minus Soranzo, same k
notes = []
for k in ks:
    a, b = ("slopecnn_all6_pp", k), ("soranzo_published", k)
    if a in DATA and b in DATA:
        d = np.array([DATA[a][i]["res"]["rmse"] - DATA[b][i]["res"]["rmse"] for i in sorted(DATA[a]) if i in DATA[b]])
        notes.append(f"{k}-fold: SlopeCNN − Soranzo RMSE = {d.mean():+.3f} ± {d.std(ddof=0):.3f}, "
                     f"SlopeCNN better in {int((d < 0).sum())}/{len(d)} folds")
tab.to_csv(os.path.join(args.out, "table_kfold_comparison.csv"), index=False)
md = ["| " + " | ".join(tab.columns) + " |", "|" + "---|" * len(tab.columns)]
md += ["| " + " | ".join(str(v) for v in r) + " |" for r in tab.itertuples(index=False)]
md += ["", "Test-fold scores, mean ± sd over folds (each fold = mean of 3 seeds). Pooled R²: the "
       "seed-averaged prediction of every slope from the fold that held it out.", ""]
md += [f"- {n}" for n in notes]
open(os.path.join(args.out, "table_kfold_comparison.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
print(tab.to_string(index=False)); print("\n".join(notes))
