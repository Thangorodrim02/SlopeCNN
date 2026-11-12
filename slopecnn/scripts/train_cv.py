"""
train_cv.py
===========
Nested k-fold cross-validation for one model / input combination.

The runs of the paper (run_paper.bat / run_paper.sh):
    python scripts/train_cv.py --model slopecnn --channels all6  --pore_pressure          SlopeCNN
    python scripts/train_cv.py --model slopecnn --channels peak4 --pore_pressure          without residual channels
    python scripts/train_cv.py --model soranzo  --as_published                            Soranzo et al. (2023)
    (add --k 7 --out output/runs_k7 for 7-fold cross-validation)

Protocol (shared by every run; --as_published changes only what Soranzo et al. did differently)
  * same outer folds and inner validation split (slope_cnn/cv.py)
  * same 18 augmentations on the training slopes only (slope_cnn/augmentation.py)
  * MSE loss, Adam (lr 1e-3, weight decay 1e-4), cosine schedule, batch 32,
    gradient clipping 1.0
  * checkpoint = lowest inner-validation MSE from epoch 30 on (the validation
    loss jumps around in the first epochs); stop after 40 epochs without improvement;
    the test fold is touched once, after training, to score the best model
  * several training seeds per fold (model initialisation and batch order)

Output: output/runs/<model>_<channels>[_pp]/
    fold<f>_seed<s>.json   metrics on the test fold, best epoch, parameters, time
    pred_fold<f>_seed<s>.csv  geometry_id, fos_true, fos_pred (test fold)
    history_fold<f>_seed<s>.csv  per-epoch train / val loss
A finished (fold, seed) is skipped on re-run, so the script can be stopped and resumed.

Tuning (run before the main runs; uses only the training folds)
    python scripts/train_cv.py --model slopecnn --channels all6 --pore_pressure --tune --lr 3e-4 --folds 1 2 --seeds 0
    ... for each learning rate in the grid, then  python scripts/pick_settings.py
    -> output/settings.json, read automatically by the main runs.
"""

import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from slope_cnn.augmentation import AUGMENTATIONS, augment
from slope_cnn.cv import make_folds
from slope_cnn.model import build_model, count_params

CHANNEL_SETS = {
    "all6":     [0, 1, 2, 3, 4, 5],   # gamma_sat, gamma_unsat, c_peak, c_res, phi_peak, phi_res
    "peak4":    [0, 1, 2, 4],         # the same without the residual channels
    "soranzo3": [0, 2, 4],            # gamma_sat, c_peak, phi_peak (Soranzo's three properties)
}
PP_CHANNEL = 6

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["slopecnn", "soranzo"], required=True)
ap.add_argument("--channels", choices=list(CHANNEL_SETS), default="all6")
ap.add_argument("--pore_pressure", action="store_true")
ap.add_argument("--tensors", default=os.path.join(ROOT, "output", "tensors_v3", "tensors.npz"))
ap.add_argument("--out", default=os.path.join(ROOT, "output", "runs"))
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--folds", type=int, nargs="+", default=None, help="run only these folds (1-based)")
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--patience", type=int, default=40)
ap.add_argument("--min_epochs", type=int, default=30,
                help="no checkpoint is selected and early stopping cannot trigger before this epoch")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=None,
                help="learning rate; default = the value tune picked (output/settings.json), else 1e-3")
ap.add_argument("--tune", action="store_true",
                help="tuning mode: score on the inner validation split only, never on the test fold")
ap.add_argument("--weight_decay", type=float, default=1e-4)
ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
ap.add_argument("--as_published", action="store_true",
                help="Soranzo et al. (2023) baseline as published, retrained on our data: "
                     "3 inputs (density, c', phi') with their scaling (2400 kg/m3, 300 kPa, 45 deg), "
                     "no water, Adam lr 1e-3 without weight decay or schedule, "
                     "6x augmentation (mirror x vertical shift 0/-1/+1), batch 32 (as in v1.6)")
args = ap.parse_args()

if args.threads:
    torch.set_num_threads(args.threads)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

aug_list = list(AUGMENTATIONS)
if args.as_published:
    if args.model != "soranzo":
        sys.exit("--as_published is only for --model soranzo")
    args.channels, args.pore_pressure = "soranzo3", False
    args.weight_decay, args.lr = 0.0, (args.lr or 1e-3)
    aug_list = [a for a in AUGMENTATIONS if a[0] == 0]          # mirror x vertical shift = 6
chans = CHANNEL_SETS[args.channels] + ([PP_CHANNEL] if args.pore_pressure else [])
run = f"{args.model}_{args.channels}" + ("_pp" if args.pore_pressure else "")
if args.as_published:
    run = "soranzo_published"

settings_path = os.path.join(ROOT, "output", "settings.json")
if args.lr is None:
    args.lr = 1e-3
    if not args.tune and os.path.exists(settings_path):
        args.lr = json.load(open(settings_path)).get(args.model, {}).get("lr", 1e-3)
if args.tune:
    run += f"_lr{args.lr:g}"
    if args.out == os.path.join(ROOT, "output", "runs"):
        args.out = os.path.join(ROOT, "output", "runs_tune")
print(f"learning rate {args.lr:g}" + ("  [TUNING: test folds are not used]" if args.tune else ""))
out_dir = os.path.join(args.out, run)
os.makedirs(out_dir, exist_ok=True)

data = np.load(args.tensors)
X = data["X"][:, chans]                       # (N, C, H, W)
if args.as_published:                         # Soranzo's own scaling (their Supplementary Material 1)
    from slope_cnn.config import GAMMA_SAT_MAX, C_PEAK_MAX, PHI_PEAK_MAX
    X = X.copy()
    X[:, 0] = X[:, 0] * GAMMA_SAT_MAX * 1000 / 9.81 / 2400   # rho = gamma_sat / g, / 2400 kg/m3
    X[:, 1] = X[:, 1] * C_PEAK_MAX / 300                       # c' / 300 kPa
    X[:, 2] = X[:, 2] * PHI_PEAK_MAX / 45                      # phi' / 45 deg
    X = X.astype(np.float32)
ids, fos = data["ids"], data["fos"].astype(np.float32)
H, W = X.shape[2:]
folds = make_folds(fos, args.k)
print(f"{run}: {len(ids)} slopes, channels {chans}, grid {H}x{W}, device {device}")


def predict(model, idx):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            out.append(model(torch.from_numpy(X[idx[s:s + 64]]).to(device)).cpu().numpy())
    return np.concatenate(out)


def metrics(y, p):
    err = p - y
    return {"r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err)))}


for f, (tr, va, te) in enumerate(folds, start=1):
    if args.folds and f not in args.folds:
        continue
    for seed in args.seeds:
        tag = f"fold{f}_seed{seed}"
        if os.path.exists(os.path.join(out_dir, tag + ".json")):
            print(f"  {tag}: done, skipped")
            continue
        torch.manual_seed(1000 * seed + f)
        rng = np.random.default_rng(1000 * seed + f)
        model = build_model(args.model, len(chans), (H, W)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = (None if args.as_published else
                 torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6))
        loss_fn = nn.MSELoss()
        pairs = np.array([(i, a) for i in tr for a in range(len(aug_list))])

        best, best_ep, best_state, wait, hist = np.inf, 0, None, 0, []
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            model.train()
            order = rng.permutation(len(pairs))
            tl, nb = 0.0, 0
            for s in range(0, len(order), args.batch):
                b = pairs[order[s:s + args.batch]]
                xb = np.stack([augment(X[i], *aug_list[a]) for i, a in b])
                xb, yb = torch.from_numpy(xb).to(device), torch.from_numpy(fos[b[:, 0]]).to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl += loss.item(); nb += 1
            if sched is not None:
                sched.step()
            vl = float(np.mean((predict(model, va) - fos[va]) ** 2))
            hist.append((ep, tl / nb, vl))
            if ep >= args.min_epochs:          # the validation loss is very noisy early on
                if vl < best:
                    best, best_ep, wait = vl, ep, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    wait += 1
            if ep % 20 == 0:
                print(f"  {tag} ep {ep:3d}  train {tl / nb:.4f}  val {vl:.4f}  best ep {best_ep}")
            if wait >= args.patience:
                break

        if best_state is None:                 # only if --epochs < --min_epochs
            best_state, best_ep = {k: v.detach().clone() for k, v in model.state_dict().items()}, ep
            best = float(np.mean((predict(model, va) - fos[va]) ** 2))
        model.load_state_dict(best_state)
        if args.tune:
            res = {"run": run, "model": args.model, "lr": args.lr, "fold": f, "seed": seed,
                   "best_epoch": best_ep, "epochs_run": ep, "val_mse": best,
                   "seconds": round(time.time() - t0)}
            with open(os.path.join(out_dir, tag + ".json"), "w") as fh:
                json.dump(res, fh, indent=2)
            print(f"  {tag}: inner-validation MSE {best:.4f} (best ep {best_ep}, {res['seconds']} s)")
            continue
        p = predict(model, te)
        res = {"run": run, "lr": args.lr, "model": args.model, "channels": chans, "fold": f, "seed": seed,
               "k": args.k, "augmentations": len(aug_list), "weight_decay": args.weight_decay,
               "schedule": "none" if sched is None else "cosine",
               "n_train": len(tr), "n_val": len(va), "n_test": len(te),
               "best_epoch": best_ep, "epochs_run": ep, "val_mse": best,
               "params": count_params(model), "seconds": round(time.time() - t0),
               **metrics(fos[te], p)}
        pd.DataFrame({"geometry_id": ids[te], "fos_true": fos[te], "fos_pred": p}).to_csv(
            os.path.join(out_dir, f"pred_{tag}.csv"), index=False)
        pd.DataFrame(hist, columns=["epoch", "train_mse", "val_mse"]).to_csv(
            os.path.join(out_dir, f"history_{tag}.csv"), index=False)
        torch.save(best_state, os.path.join(out_dir, f"model_{tag}.pt"))
        with open(os.path.join(out_dir, tag + ".json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"  {tag}: test R2 {res['r2']:.3f}  RMSE {res['rmse']:.3f}  MAE {res['mae']:.3f}  "
              f"(best ep {best_ep}, {res['seconds']} s)")

print("finished")
