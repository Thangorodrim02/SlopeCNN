"""
build_tensors_from_slim.py
==========================
Builds the CNN input tensors for the v3 dataset straight from the Slide2 models.

    python scripts/build_tensors_from_slim.py --models ../data/slide2_models
           --labels ../slopelem/data/labels_v3_346.csv

Output (repo root):
    output/tensors_v3/tensors.npz   X (N, 7, 42, 243) float32, ids (N,), fos (N,)
                                    channels 0-5 = CNN_CHANNELS, 6 = pore pressure
    output/tensors_v3/domain_info.npz
Every tensor is built with the pore-pressure channel; the training script
chooses which channels to use (--channels).
"""

import argparse, os, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from slope_cnn.config import Config
from slope_cnn.grid import build_global_grid
from slope_cnn.slim_raster import slim_to_tensor

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True, help="folder with FeedN.slim")
ap.add_argument("--labels", required=True, help="labels_v3_346.csv (geometry_id, FoS)")
ap.add_argument("--out", default=os.path.join(ROOT, "output", "tensors_v3"))
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
grid = build_global_grid(Config(angle_min=5.0))
lab = pd.read_csv(args.labels).sort_values("geometry_id")
print(f"Grid {grid.ny} x {grid.nx}, toe at row {grid.toe_row}, col {grid.toe_col}; {len(lab)} models")

X = np.zeros((len(lab), 7, grid.ny, grid.nx), dtype=np.float32)
t0 = time.time()
for n, gid in enumerate(lab.geometry_id):
    X[n], _ = slim_to_tensor(os.path.join(args.models, f"Feed{gid}.slim"), grid, pore_pressure=True)
    if (n + 1) % 50 == 0:
        print(f"  {n + 1}/{len(lab)}  {time.time() - t0:.0f} s")

assert X.min() >= 0 and X.max() <= 1, f"values outside [0, 1]: {X.min()} {X.max()}"
np.savez_compressed(os.path.join(args.out, "tensors.npz"), X=X,
                    ids=lab.geometry_id.to_numpy(), fos=lab.FoS.to_numpy(np.float32))
np.savez(os.path.join(args.out, "domain_info.npz"), **grid.to_dict())
print(f"Saved {len(lab)} tensors, value range [{X.min():.3f}, {X.max():.3f}] -> {args.out}")
