"""
pick_settings.py
================
Chooses each model's learning rate from the tuning runs (output/runs_tune),
by the lowest mean inner-validation MSE. Test folds are never looked at.
Writes output/settings.json, which train_cv.py reads.

    python scripts/pick_settings.py
"""

import glob, json, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = [json.load(open(p)) for p in glob.glob(os.path.join(ROOT, "output", "runs_tune", "*", "fold*_seed*.json"))]
if not rows:
    raise SystemExit("no tuning runs found in output/runs_tune")
df = pd.DataFrame(rows)
tab = df.groupby(["model", "lr"]).agg(val_mse=("val_mse", "mean"), runs=("val_mse", "size"),
                                      best_epoch=("best_epoch", "median")).reset_index()
print(tab.to_string(index=False))
settings = {m: {"lr": float(g.loc[g.val_mse.idxmin(), "lr"])} for m, g in tab.groupby("model")}
with open(os.path.join(ROOT, "output", "settings.json"), "w") as fh:
    json.dump(settings, fh, indent=2)
print("chosen:", settings)
