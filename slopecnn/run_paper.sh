#!/usr/bin/env bash
# Reproduces every result of the paper (Linux / macOS / Git Bash):  bash run_paper.sh
# Resumable: stop it at any time and run it again; finished (fold, seed) runs are skipped.
# About 4-5 h on an RTX 4060 GPU (the CNN runs); the rest takes minutes.
set -euo pipefail
cd "$(dirname "$0")"
MODELS=../data/slide2_models
LABELS=../slopelem/data/labels_v3_346.csv
EXCEL=../data/soil_properties_346geom_v3_cnn_input.xlsx
PY=${PYTHON:-python}

# 0. Input tensors (included; rebuilt only if missing, needs slopelem)
[ -f output/tensors_v3/tensors.npz ] || $PY scripts/build_tensors_from_slim.py --models "$MODELS" --labels "$LABELS"

# 1. CNNs, 5-fold: SlopeCNN, SlopeCNN without residual channels, Soranzo et al. (2023)
$PY scripts/train_cv.py --model slopecnn --channels all6  --pore_pressure
$PY scripts/train_cv.py --model slopecnn --channels peak4 --pore_pressure
$PY scripts/train_cv.py --model soranzo --as_published

# 2. CNNs, 7-fold
$PY scripts/train_cv.py --model slopecnn --channels all6 --pore_pressure --k 7 --out output/runs_k7
$PY scripts/train_cv.py --model soranzo --as_published --k 7 --out output/runs_k7

# 3. Linear regression and random forest on the parameter table (same folds, seconds)
for K in 5 7; do
  RUNS=output/runs; [ "$K" = 7 ] && RUNS=output/runs_k7
  $PY scripts/tabular_baselines.py --excel "$EXCEL" --k $K --out $RUNS
  $PY scripts/tabular_baselines.py --excel "$EXCEL" --wt_depth output/tensors_v3/wt_depth.csv --k $K --out $RUNS
done

# 4. Tables
$PY scripts/summarize_runs.py
$PY scripts/summarize_runs.py --runs output/runs_k7 --out output/summary_k7

# 5. Figures -> output/paper_figures
for S in make_pipeline_flowchart make_geometry_schematic make_dataset_figures plot_input_channels \
         make_activation_figure make_architecture_figure make_lr_figure plot_kfold_figures make_parity_figure; do
  $PY scripts/$S.py
done
echo "Done. Tables: output/summary, output/summary_k7   Figures: output/paper_figures"
