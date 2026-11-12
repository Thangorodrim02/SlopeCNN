"""
slope_cnn/cv.py
===============
One definition of the cross-validation folds, shared by every model
(SlopeCNN, the Soranzo-type baseline, the tabular baselines).

Outer folds: k stratified folds on FoS (quintile bins), fixed seed. Each
             outer fold is the TEST fold once and is never used for training,
             early stopping or model selection.
Inner split: from the remaining training slopes, VAL_FRAC (15 %) are held out
             for early stopping (stratified, fixed per fold). The same inner
             split is used by every model and every training seed.
"""

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

FOLD_SEED = 42
VAL_FRAC = 0.15


def _bins(fos: np.ndarray, n_bins: int = 5) -> np.ndarray:
    edges = np.quantile(fos, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(fos, edges)


def make_folds(fos: np.ndarray, k: int = 5) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """[(train_idx, val_idx, test_idx), ...] as positions into the dataset arrays."""
    folds = []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=FOLD_SEED)
    for f, (rest, test) in enumerate(skf.split(np.zeros(len(fos)), _bins(fos))):
        tr, va = train_test_split(rest, test_size=VAL_FRAC, random_state=FOLD_SEED + f,
                                  stratify=_bins(fos[rest]))
        folds.append((np.sort(tr), np.sort(va), np.sort(test)))
    return folds
