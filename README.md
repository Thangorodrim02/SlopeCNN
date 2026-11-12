# SlopeCNN

Code and data of the SlopeCNN study: a convolutional neural network (CNN) surrogate
model that predicts the factor of safety (FoS) of slopes at Section 4 of the Çöllolar
open-pit lignite mine (Elbistan, Türkiye), compared with the CNN of Soranzo et al. (2023).

The repository has three parts, which together reproduce every number and figure of the paper:

| Folder | What it is |
|---|---|
| [`data/`](data) | The dataset: 346 Rocscience Slide2 models of Section 4, their soil properties and FoS labels, and the 50-model validation against Slide2 |
| [`slopelem/`](slopelem) | The LEM model generator: builds the limit equilibrium model of each slope from its Slide2 file and calculates its FoS (simplified Bishop, anisotropic peak / residual strength, Hu pore pressure, critical-circle search) |
| [`slopecnn/`](slopecnn) | SlopeCNN and the Soranzo et al. (2023) baseline: input tensors, training with cross-validation, the finished runs, and the scripts for the tables and figures |

```
Slide2 models (data/slide2_models)
   │  slopelem: simplified Bishop critical-circle search
   ▼
FoS labels (slopelem/data/labels_v3_346.csv)
   │  slopecnn: 7-channel input tensors (6 soil properties + pore pressure) on a 42 × 243 grid of 10 m cells
   ▼
SlopeCNN vs Soranzo et al. (2023), stratified 5- and 7-fold cross-validation, 3 seeds
```

## Main results

| Model | k | R² | RMSE | Pooled R² | FoS < 1 predicted ≥ 1 (of 88) | Parameters |
|---|---|---|---|---|---|---|
| Soranzo et al. (2023) | 5 | 0.840 ± 0.016 | 0.170 ± 0.009 | 0.850 | 19 | 2,551,105 |
| SlopeCNN | 5 | 0.946 ± 0.011 | 0.098 ± 0.009 | 0.953 | 10 | 124,343 |
| Soranzo et al. (2023) | 7 | 0.830 ± 0.029 | 0.174 ± 0.019 | 0.837 | 27 | 2,551,105 |
| SlopeCNN | 7 | 0.939 ± 0.015 | 0.104 ± 0.018 | 0.945 | 9 | 124,343 |

Test folds; a fold's score is the mean of 3 seeds; mean ± sd over folds. SlopeCNN has the
lower RMSE in every fold (5 of 5 and 7 of 7). The FoS labels agree with Slide2 within
0.055 % on average (largest difference 0.147 %) on 50 checked models.

## Installation

Python ≥ 3.9. From the repository root:

```bash
pip install -r requirements.txt     # PyTorch, NumPy, SciPy, pandas, scikit-learn, matplotlib, openpyxl, slopelem
```

For a GPU, install the PyTorch build for your CUDA version first (see pytorch.org).

## Reproducing the paper

```bash
# 1. FoS labels: recompute all 346 from their critical circles (seconds) and run the tests
cd slopelem
python examples/reproduce_labels.py ../data/slide2_models
pytest
python -m slopelem.cli fos ../data/slide2_models/Feed7.slim     # full critical-circle search of one model (30-60 s)

# 2. Tables and figures from the included runs (no training)
cd ../slopecnn
python scripts/summarize_runs.py
python scripts/summarize_runs.py --runs output/runs_k7 --out output/summary_k7
python scripts/make_parity_figure.py

# 3. Everything, including training the CNNs (about 4-5 h on an RTX 4060 GPU)
run_paper.bat            # Windows
bash run_paper.sh        # Linux / macOS
```

| Paper | Script (in `slopecnn/scripts/`) or file |
|---|---|
| Fig. 1 Cross-sections of the mine | from Öge (2008), not made by code |
| Fig. 2 Pipeline | `make_pipeline_flowchart.py` |
| Fig. 3 Slope geometry | `make_geometry_schematic.py` |
| Fig. 4 Soil property distributions | `make_dataset_figures.py` → `fig_property_boxplots` |
| Fig. 5 FoS labels | `make_dataset_figures.py` → `fig_fos_vs_angle` |
| Fig. 6 Example critical circle | screenshot of a Slide2 model, not made by code |
| Fig. 7 Input channels | `plot_input_channels.py` |
| Fig. 8 Activation functions | `make_activation_figure.py` |
| Fig. 9 Architectures | `make_architecture_figure.py` |
| Fig. 10 Learning rate | `make_lr_figure.py` |
| Fig. 11, 12 Cross-validation curves | `plot_kfold_figures.py` → `fold_comparison_slopecnn_all6_pp_5and7fold`, `cv_comparison` |
| Fig. 13 Predicted vs calculated FoS | `make_parity_figure.py` |
| Table 1 Base values of the soil properties | `data/dataset_v3_346_geometries_FoS.xlsx`, sheet *Reference Values* |
| Table 2 and the paired differences | `summarize_runs.py` → `output/summary*/results.md`, `paired.csv` |
| Slide2 validation (50 models) | `slopelem/validation/Slide2_validation_50_models.xlsx` |

All figures are written to `slopecnn/output/paper_figures/`; the versions in the repository
were made from the included runs and are the ones in the paper.

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite the paper when it is published.

## Licence

MIT, see [`LICENSE`](LICENSE). The dataset in `data/` is released under the same terms.

## References

* Bishop, A. W. (1955). The use of the slip circle in the stability analysis of slopes. *Géotechnique, 5*(1), 7–17.
* Öge, İ. F. (2008). *Slope stability analysis and design in Elbistan-Çöllolar open cast mine* (M.Sc. thesis). Middle East Technical University, Ankara.
* Özbay, A., & Cabalar, A. F. (2015). FEM and LEM stability analyses of the fatal landslides at Çöllolar open-cast lignite mine in Elbistan, Turkey. *Landslides, 12*(1), 155–163.
* Soranzo, E., Guardiani, C., Chen, Y., Wang, Y., & Wu, W. (2023). Convolutional neural networks prediction of the factor of safety of random layered slopes by the strength reduction method. *Acta Geotechnica, 18*(6), 3391–3402. https://doi.org/10.1007/s11440-022-01783-3
