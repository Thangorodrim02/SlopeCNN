"""
summarize_runs.py
=================
Collects the finished runs of one cross-validation (5-fold or 7-fold) and writes the
numbers reported in the paper (no training, takes a second).

    python scripts/summarize_runs.py                                      (5-fold)
    python scripts/summarize_runs.py --runs output/runs_k7 --out output/summary_k7

Output (--out):
    results.csv   one row per run
                  r2 / rmse / mae      mean and sd (population, ddof = 0) over the test folds;
                                       the score of a fold is the mean of its seeds (Table 2)
                  pooled_r2 / _rmse / _mae
                                       out-of-fold scores: every slope predicted by the models
                                       that did not see it in training, mean over the seeds
                  failing_called_stable   label FoS < 1.0, predicted >= 1.0
                  stable_called_failing   label FoS >= 1.0, predicted < 1.0
                  params, best_epoch_median (CNN runs)
    results.md    the same as a Markdown table
    paired.csv    fold-paired RMSE differences (a - b; negative = a better):
                  SlopeCNN vs Soranzo et al. (2023), and SlopeCNN with vs without the
                  residual-strength channels
"""

import argparse, glob, json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=os.path.join(ROOT, "output", "runs"))
ap.add_argument("--out", default=os.path.join(ROOT, "output", "summary"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

NAMES = {
    "slopecnn_all6_pp": "SlopeCNN (7 channels)",
    "slopecnn_peak4_pp": "SlopeCNN without residual channels (5 channels)",
    "soranzo_published": "Soranzo et al. (2023)",
    "tabular_rf_wt": "Random forest + water-table depth",
    "tabular_rf": "Random forest",
    "tabular_rf_angle_only": "Random forest, slope angle only",
    "tabular_linear_wt": "Linear regression + water-table depth",
    "tabular_linear": "Linear regression",
}

rows = [json.load(open(p)) for p in glob.glob(os.path.join(args.runs, "*", "fold*_seed*.json"))]
if not rows:
    raise SystemExit(f"no finished runs in {args.runs}")
df = pd.DataFrame(rows)
fold = df.groupby(["run", "fold"])[["r2", "rmse", "mae"]].mean()      # a fold = mean of its seeds


def scores(y, p):
    e = p - y
    return {"r2": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2),
            "rmse": np.sqrt(np.mean(e ** 2)), "mae": np.mean(np.abs(e))}


out = []
for run, g in fold.groupby(level=0):
    r = {"run": run, "model": NAMES.get(run, run), "folds": len(g),
         "seeds": df[df.run == run].seed.nunique()}
    for m in ("r2", "rmse", "mae"):
        r[f"{m}_mean"], r[f"{m}_sd"] = g[m].mean(), g[m].std(ddof=0)
    files = glob.glob(os.path.join(args.runs, run, "pred_fold*_seed*.csv"))
    p = pd.concat(pd.read_csv(f) for f in files).groupby("geometry_id").mean()
    y, yp = p.fos_true.to_numpy(), p.fos_pred.to_numpy()
    r.update({f"pooled_{k}": v for k, v in scores(y, yp).items()})
    r["failing_called_stable"] = int(np.sum((y < 1) & (yp >= 1)))
    r["stable_called_failing"] = int(np.sum((y >= 1) & (yp < 1)))
    r["n_failing"] = int(np.sum(y < 1))
    d = df[df.run == run]
    if "params" in d:
        r["params"] = d.params.max()
    if "best_epoch" in d:
        r["best_epoch_median"] = d.best_epoch.median()
    out.append(r)

res = pd.DataFrame(out).sort_values("rmse_mean")
res.round(4).to_csv(os.path.join(args.out, "results.csv"), index=False)

lines = ["| Model | Folds | R² | RMSE | MAE | Pooled R² | FoS < 1 called ≥ 1 |", "|---|---|---|---|---|---|---|"]
for _, r in res.iterrows():
    lines.append(f"| {r.model} | {r.folds} | {r.r2_mean:.3f} ± {r.r2_sd:.3f} | {r.rmse_mean:.3f} ± {r.rmse_sd:.3f} "
                 f"| {r.mae_mean:.3f} ± {r.mae_sd:.3f} | {r.pooled_r2:.3f} | {r.failing_called_stable} of {r.n_failing} |")
open(os.path.join(args.out, "results.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))

pairs = []
rm = fold.rmse.unstack(0)
for a, b, what in [("slopecnn_all6_pp", "soranzo_published", "SlopeCNN vs Soranzo et al. (2023)"),
                   ("slopecnn_all6_pp", "slopecnn_peak4_pp", "SlopeCNN with vs without residual channels")]:
    if a in rm and b in rm:
        d = (rm[a] - rm[b]).dropna()
        pairs.append({"comparison": what, "a": a, "b": b, "folds": len(d),
                      "rmse_diff_mean": d.mean(), "rmse_diff_sd": d.std(),
                      "folds_a_better": int((d < 0).sum())})
if pairs:
    pd.DataFrame(pairs).round(4).to_csv(os.path.join(args.out, "paired.csv"), index=False)
    print("\n" + pd.DataFrame(pairs).round(4).to_string(index=False))
print(f"\nwritten to {args.out}")
