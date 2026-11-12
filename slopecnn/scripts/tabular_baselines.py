"""
tabular_baselines.py
====================
Non-image baselines on exactly the same test folds as the CNNs.

    python scripts/tabular_baselines.py --excel soil_properties_346geom_v3_cnn_input.xlsx
           [--wt_depth wt_depth.csv]

Inputs: the 47 numbers that define each slope in the Excel sheet (slope angle,
5 boundary angles and offsets, 6 properties x 6 layers), optionally + the mean
and toe water-table depth. Models are fitted on the same training slopes as the
CNNs (the inner validation slopes are not used: these models need no early
stopping). Random forest is run with the same seeds as the CNNs.

Output: output/runs/tabular_<model>[_wt]/fold<f>_seed<s>.json and pred_*.csv
"""

import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from slope_cnn.cv import make_folds

ap = argparse.ArgumentParser()
ap.add_argument("--excel", required=True)
ap.add_argument("--wt_depth", default=None, help="csv with gid, meandepth, toe_depth")
ap.add_argument("--tensors", default=os.path.join(ROOT, "output", "tensors_v3", "tensors.npz"))
ap.add_argument("--out", default=os.path.join(ROOT, "output", "runs"))
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
args = ap.parse_args()

d = np.load(args.tensors)
ids, fos = d["ids"], d["fos"].astype(np.float32)
g = pd.read_excel(args.excel).set_index("geometry_id").loc[ids]
cols = [c for c in g.columns if c != "Factor of Safety (FS)"]
assert np.allclose(g["Factor of Safety (FS)"].to_numpy(), fos), "Excel FoS differs from tensors"
F = g[cols].copy()
if args.wt_depth:
    w = pd.read_csv(args.wt_depth).set_index("gid").loc[ids]
    F["wt_mean_depth"], F["wt_toe_depth"] = w.meandepth.to_numpy(), w.toe_depth.to_numpy()
F = F.to_numpy(float)
suffix = "_wt" if args.wt_depth else ""

def metrics(y, p):
    e = p - y
    return {"r2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
            "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e)))}

alpha = g[["slope_angle_deg"]].to_numpy(float)
for name, make, feats, seeds in [
        ("linear", lambda s: LinearRegression(), F, [0]),
        ("rf", lambda s: RandomForestRegressor(500, random_state=s, n_jobs=-1), F, args.seeds),
        ("rf_angle_only", lambda s: RandomForestRegressor(500, min_samples_leaf=5, random_state=s, n_jobs=-1), alpha, [0])]:
    if name == "rf_angle_only" and suffix:
        continue
    out = os.path.join(args.out, f"tabular_{name}{suffix}")
    os.makedirs(out, exist_ok=True)
    r2 = []
    for f, (tr, va, te) in enumerate(make_folds(fos, args.k), start=1):
        for s in seeds:
            p = make(s).fit(feats[tr], fos[tr]).predict(feats[te])
            m = {"run": f"tabular_{name}{suffix}", "fold": f, "seed": s, "n_train": len(tr),
                 "n_test": len(te), **metrics(fos[te], p)}
            r2.append(m["r2"])
            with open(os.path.join(out, f"fold{f}_seed{s}.json"), "w") as fh:
                json.dump(m, fh, indent=2)
            pd.DataFrame({"geometry_id": ids[te], "fos_true": fos[te], "fos_pred": p}).to_csv(
                os.path.join(out, f"pred_fold{f}_seed{s}.csv"), index=False)
    print(f"tabular_{name}{suffix}: mean test R2 {np.mean(r2):.3f} (sd {np.std(r2):.3f})")
