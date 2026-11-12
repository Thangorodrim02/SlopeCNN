"""
slope_cnn/augmentation.py
=========================
The 18 label-preserving transformations used for training BOTH models.

Each training slope is used 18 times per epoch:
  horizontal shift  0, 2 or 4 columns to the right (20 / 40 m)
  vertical shift    0, -1 (up) or +1 (down) row     (10 m)
  mirror            no / yes
  3 x 3 x 2 = 18 copies, all with the slope's own FoS.

Why the FoS stays the same
  * mirror: a mirrored slope has the same factor of safety;
  * vertical shift: the whole section moves 10 m; the grid has 2.5 rows of
    empty margin below the model base and 4.5 above the crest, so nothing is cut;
  * right shift: the image's right edge is dropped. 2 columns remove only the
    empty margin; 4 columns remove the last 20 m of a floor that runs 464 m
    past the toe, far outside any slip circle.

Augmentation is applied to the training part of each fold only, never to the
validation or test slopes, and identically for every model (fair comparison).
"""

from itertools import product
from typing import Tuple

import numpy as np

H_SHIFTS = (0, 2, 4)
V_SHIFTS = (0, -1, 1)
MIRROR = (False, True)

# (h_shift, v_shift, mirror); index 0 is the identity
AUGMENTATIONS: Tuple[Tuple[int, int, bool], ...] = tuple(product(H_SHIFTS, V_SHIFTS, MIRROR))


def augment(t: np.ndarray, h_shift: int, v_shift: int, mirror: bool) -> np.ndarray:
    """Apply one transformation to a (C, H, W) tensor; vacated cells are 0."""
    out = t
    if h_shift:
        out = np.roll(out, h_shift, axis=2)
        out[:, :, :h_shift] = 0.0
    if v_shift:
        out = np.roll(out, v_shift, axis=1)
        if v_shift > 0:
            out[:, :v_shift, :] = 0.0
        else:
            out[:, v_shift:, :] = 0.0
    if mirror:
        out = out[:, :, ::-1]
    return np.ascontiguousarray(out)


def check_label_preserving(t: np.ndarray) -> dict:
    """Soil cells lost by each transformation (0 = nothing cut)."""
    n = int((t[0] > 0).sum())
    return {a: n - int((augment(t.copy(), *a)[0] > 0).sum()) for a in AUGMENTATIONS}
