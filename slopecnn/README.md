# slopecnn

The CNN part of the repository: SlopeCNN, the Soranzo et al. (2023) baseline, the
training and cross-validation code, the finished runs of the paper and the scripts
that make its tables and figures.

## Results (test folds; a fold's score is the mean of 3 seeds; mean ± sd over folds)

| Model | k | R² | RMSE | MAE | Pooled R² | FoS < 1 predicted ≥ 1 (of 88) |
|---|---|---|---|---|---|---|
| Soranzo et al. (2023) | 5 | 0.840 ± 0.016 | 0.170 ± 0.009 | 0.128 ± 0.007 | 0.850 | 19 |
| SlopeCNN | 5 | 0.946 ± 0.011 | 0.098 ± 0.009 | 0.068 ± 0.006 | 0.953 | 10 |
| Soranzo et al. (2023) | 7 | 0.830 ± 0.029 | 0.174 ± 0.019 | 0.135 ± 0.016 | 0.837 | 27 |
| SlopeCNN | 7 | 0.939 ± 0.015 | 0.104 ± 0.018 | 0.073 ± 0.012 | 0.945 | 9 |

SlopeCNN without the residual-strength channels (5-fold): R² 0.910, RMSE 0.127,
16 failing slopes predicted stable. Linear regression and random forest on the 47
parameters of each model (5-fold): 0.811 and 0.886; with the water-table depth 0.855 and
0.905; random forest on the slope angle alone 0.869. Full tables:
`output/summary/results.md` (5-fold) and `output/summary_k7/results.md` (7-fold).

## Layout

```
slope_cnn/
  config.py          site constants, layer base values, normalisation maxima
  slim_raster.py     Slide2 .slim -> (7, 42, 243) input tensor   [needs slopelem]
  grid.py            the fixed grid of 10 m cells (toe always on the same cell)
  geometry.py        geometry container used by grid.py
  augmentation.py    the 18 label-preserving transformations (mirror x vertical x horizontal shift)
  cv.py              the shared folds: stratified test folds + 15 % inner validation
  model.py           SlopeCNN and SoranzoCNN
scripts/
  build_tensors_from_slim.py   output/tensors_v3/tensors.npz from the 346 Slide2 models
  train_cv.py                  nested cross-validation of one model (--as_published, --k, --out)
  tabular_baselines.py         linear regression and random forest on the same folds
  pick_settings.py             learning rate from the tuning runs -> output/settings.json (optional)
  summarize_runs.py            results.csv / results.md / paired.csv  (Table 2 and the paired differences)
  make_pipeline_flowchart.py   Fig. 2   pipeline
  make_geometry_schematic.py   Fig. 3   geometry of model 92 and the ranges of the 346 models
  make_dataset_figures.py      Fig. 4, 5 property box plots, FoS vs slope angle (+ two extra dataset figures)
  plot_input_channels.py       Fig. 7   the seven input channels of model 7
  make_activation_figure.py    Fig. 8   ReLU (SlopeCNN) and leaky ReLU (Soranzo et al., 2023)
  make_architecture_figure.py  Fig. 9   both architectures with their parameter counts
  make_lr_figure.py            Fig. 10  learning rate during training (cosine vs constant)
  plot_kfold_figures.py        Fig. 11, 12 training curves and best-fold comparison, 5- and 7-fold
  make_parity_figure.py        Fig. 13  out-of-fold predicted vs calculated FoS
check_setup.py                 checks the packages and the GPU, runs one test epoch
run_paper.bat / run_paper.sh   every run, table and figure of the paper
output/
  tensors_v3/                  tensors.npz (346 x 7 x 42 x 243), water-table depths, boundary fits
  settings.json                learning rates (1e-3 for both models)
  runs/, runs_k7/              finished runs, 5-fold and 7-fold (see below)
  summary/, summary_k7/        tables made from them
  paper_figures/               figures made from them
```

Each run folder holds, for every fold `f` and seed `s`:
`fold<f>_seed<s>.json` (test-fold R², RMSE, MAE, best epoch, parameters, settings),
`pred_fold<f>_seed<s>.csv` (geometry_id, FoS label, prediction) and, for the CNNs,
`history_fold<f>_seed<s>.csv` (training and validation loss per epoch). Trained weights
are not included; `train_cv.py` writes them as `model_fold<f>_seed<s>.pt`.

| Run folder | Model |
|---|---|
| `slopecnn_all6_pp` | SlopeCNN, 7 channels (5- and 7-fold) |
| `slopecnn_peak4_pp` | SlopeCNN without the residual channels, 5 channels (5-fold) |
| `soranzo_published` | Soranzo et al. (2023) (5- and 7-fold) |
| `tabular_*` | linear regression / random forest, with `_wt` the water-table depth |

## Running

From the repository root:

```bash
pip install -r requirements.txt          # includes slopelem from ./slopelem
cd slopecnn
python check_setup.py                    # packages, GPU, one test epoch
```

Tables and figures from the included runs (seconds, no training):

```bash
python scripts/summarize_runs.py
python scripts/summarize_runs.py --runs output/runs_k7 --out output/summary_k7
python scripts/make_parity_figure.py     # and the other figure scripts
```

Everything again, including training: `run_paper.bat` (Windows) or `bash run_paper.sh`.
The CNN runs take about 4–5 h on an RTX 4060; they are resumable, and a (fold, seed)
whose `.json` already exists is skipped, so delete `output/runs*/<run>` first to retrain it.
The results of a retrained run can differ slightly from the included ones (GPU
non-determinism), but the folds, inner validation splits and seeds are fixed.

Single runs:

```bash
python scripts/train_cv.py --model slopecnn --channels all6 --pore_pressure                 # SlopeCNN, 5-fold
python scripts/train_cv.py --model soranzo --as_published --k 7 --out output/runs_k7         # Soranzo, 7-fold
python scripts/train_cv.py --model slopecnn --channels all6 --pore_pressure --folds 1 --seeds 0 --out output/test   # quick test
```

Channel sets: `all6` = γ_sat, γ_unsat, c′, c_r, φ′, φ_r; `peak4` = γ_sat, γ_unsat, c′, φ′;
`--pore_pressure` adds the pore pressure u as the last channel.

## Protocol

Shared by both models: stratified folds on FoS quintiles (seed 42), 15 % of the training
slopes as inner validation, 3 seeds per fold, MSE loss, Adam (lr 1e-3), batch 32,
gradient clipping 1.0, at most 200 epochs, checkpoint = lowest validation MSE from
epoch 30 on, stop after 40 epochs without improvement; the test fold is used once,
after training.

SlopeCNN only: weight decay 1e-4, cosine learning-rate schedule (1e-3 → 1e-6 over 200
epochs), 18 augmentations (mirror × vertical shift 0/±1 × horizontal shift 0/2/4 cells),
7 channels scaled by fixed maxima (23 and 17 kN/m³, 71 and 58 kPa, 40° and 36°, 1600 kPa).

Soranzo et al. (2023), `--as_published`: their 3 inputs (ρ from γ_sat, c′, φ′) with their
scaling (2400 kg/m³, 300 kPa, 45°), no water, no weight decay, constant learning rate,
6 augmentations (mirror × vertical shift), on the same 42 × 243 grid.

## Parameter counts (42 × 243 input)

SlopeCNN 124,343; Soranzo et al. (2023) 2,551,105 (96 % in the first fully connected layer).
Soranzo et al. report 334,433 parameters for their 100 × 100 input. Their published
Keras file (`11440_2022_1783_MOESM9_ESM.h5`) declares the input `[None, 3, 100, 100]`
with `channels_last`, so Keras reads it as a 3 × 100 image with 100 channels, which
gives 334,433; built as their paper describes it, the network is larger.
