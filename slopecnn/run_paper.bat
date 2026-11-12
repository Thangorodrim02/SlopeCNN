@echo off
REM Reproduces every result of the paper (Windows). Run from any folder: run_paper.bat
REM Resumable: stop it at any time and run it again; finished (fold, seed) runs are skipped.
REM About 4-5 h on an RTX 4060 GPU (the CNN runs); the rest takes minutes.

cd /d "%~dp0"
set MODELS=..\data\slide2_models
set LABELS=..\slopelem\data\labels_v3_346.csv
set EXCEL=..\data\soil_properties_346geom_v3_cnn_input.xlsx

REM 0. Input tensors (included; rebuilt only if missing, needs slopelem)
if not exist output\tensors_v3\tensors.npz python scripts\build_tensors_from_slim.py --models "%MODELS%" --labels "%LABELS%"
if errorlevel 1 goto :fail

REM 1. CNNs, 5-fold: SlopeCNN, SlopeCNN without residual channels, Soranzo et al. (2023)
python scripts\train_cv.py --model slopecnn --channels all6  --pore_pressure
if errorlevel 1 goto :fail
python scripts\train_cv.py --model slopecnn --channels peak4 --pore_pressure
if errorlevel 1 goto :fail
python scripts\train_cv.py --model soranzo --as_published
if errorlevel 1 goto :fail

REM 2. CNNs, 7-fold
python scripts\train_cv.py --model slopecnn --channels all6 --pore_pressure --k 7 --out output\runs_k7
if errorlevel 1 goto :fail
python scripts\train_cv.py --model soranzo --as_published --k 7 --out output\runs_k7
if errorlevel 1 goto :fail

REM 3. Linear regression and random forest on the parameter table (same folds, seconds)
python scripts\tabular_baselines.py --excel "%EXCEL%" --k 5 --out output\runs
if errorlevel 1 goto :fail
python scripts\tabular_baselines.py --excel "%EXCEL%" --wt_depth output\tensors_v3\wt_depth.csv --k 5 --out output\runs
if errorlevel 1 goto :fail
python scripts\tabular_baselines.py --excel "%EXCEL%" --k 7 --out output\runs_k7
if errorlevel 1 goto :fail
python scripts\tabular_baselines.py --excel "%EXCEL%" --wt_depth output\tensors_v3\wt_depth.csv --k 7 --out output\runs_k7
if errorlevel 1 goto :fail

REM 4. Tables
python scripts\summarize_runs.py
if errorlevel 1 goto :fail
python scripts\summarize_runs.py --runs output\runs_k7 --out output\summary_k7
if errorlevel 1 goto :fail

REM 5. Figures -> output\paper_figures
for %%S in (make_pipeline_flowchart make_geometry_schematic make_dataset_figures plot_input_channels make_activation_figure make_architecture_figure make_lr_figure plot_kfold_figures make_parity_figure) do (
  python scripts\%%S.py
  if errorlevel 1 goto :fail
)

echo.
echo Done. Tables: output\summary, output\summary_k7   Figures: output\paper_figures
pause
exit /b 0

:fail
echo.
echo Something failed - see the error above.
pause
exit /b 1
