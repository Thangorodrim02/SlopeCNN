| Model | k | Trainings | R² (mean ± sd of folds) | RMSE | MAE | Best fold R² | Worst fold R² | Pooled R² (all 346) |
|---|---|---|---|---|---|---|---|---|
| Soranzo et al. (2023) | 5 | 15 | 0.840 ± 0.016 | 0.170 ± 0.009 | 0.128 ± 0.007 | 0.858 | 0.819 | 0.850 |
| SlopeCNN | 5 | 15 | 0.946 ± 0.011 | 0.098 ± 0.009 | 0.068 ± 0.006 | 0.957 | 0.927 | 0.953 |
| Soranzo et al. (2023) | 7 | 21 | 0.830 ± 0.029 | 0.174 ± 0.019 | 0.135 ± 0.016 | 0.873 | 0.784 | 0.837 |
| SlopeCNN | 7 | 21 | 0.939 ± 0.015 | 0.104 ± 0.018 | 0.073 ± 0.012 | 0.965 | 0.915 | 0.945 |

Test-fold scores, mean ± sd over folds (each fold = mean of 3 seeds). Pooled R²: the seed-averaged prediction of every slope from the fold that held it out.

- 5-fold: SlopeCNN − Soranzo RMSE = -0.072 ± 0.016, SlopeCNN better in 5/5 folds
- 7-fold: SlopeCNN − Soranzo RMSE = -0.070 ± 0.012, SlopeCNN better in 7/7 folds
