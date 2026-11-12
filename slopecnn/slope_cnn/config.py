"""
slope_cnn/config.py
===================
All constants and user-configurable settings in one place.
Edit this file to change run parameters.
"""

from dataclasses import dataclass, field
from typing import List, Dict


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY CONSTANTS  (Section 4 cross-section — do not change)
# ─────────────────────────────────────────────────────────────────────────────

TOE_X   = 793.929    # slope toe x  (m)
TOE_Y   = 1025.389   # slope toe y  (m)
CREST_Y = 1175.354   # top surface elevation  (m)
BASE_Y  = 833.5      # bottom elevation       (m)
RIGHT_X = 1258.165   # right wall x           (m)

H         = CREST_Y - TOE_Y          # slope height    = 149.965 m
L_ORIG    = TOE_X   - 404.186        # original run    = 389.743 m
TOP_WIDTH = 404.186 - 199.015        # flat top width  = 205.171 m
ANCHOR_X  = 199.015                  # boundary anchor x (m)

# Original stepped slope face — relative to toe (x_rel, y_rel)
import numpy as np
SLOPE_FACE_REL = np.array([
    [  0.000,   0.000], [ -3.805,   3.653], [-30.899,  29.664],
    [-60.151,  29.664], [-89.402,  29.664], [-103.616,  45.526],
    [-117.829,  61.387], [-147.081,  61.387], [-176.332,  61.387],
    [-192.194,  76.631], [-208.055,  91.874], [-238.749,  91.874],
    [-269.442,  91.874], [-286.760, 108.961], [-300.341, 122.362],
    [-331.241, 122.362], [-362.140, 122.362], [-379.998, 140.220],
    [-389.743, 149.965],
], dtype=np.float64)

# Layer boundaries — top to bottom
BOUNDARIES = [
    {"id": 1, "key": "bnd1", "name": "Loam / Blue Clay",
     "anchor_y": 1151.6, "orig_angle":  2.48},
    {"id": 2, "key": "bnd2", "name": "Blue Clay / Gyttja",
     "anchor_y": 1138.1, "orig_angle": -1.21},
    {"id": 3, "key": "bnd3", "name": "Gyttja / Lignite",
     "anchor_y": 1099.8, "orig_angle": -1.78},
    {"id": 4, "key": "bnd4", "name": "Lignite / Black Clay",
     "anchor_y": 1043.9, "orig_angle": -1.61},
    {"id": 5, "key": "bnd5", "name": "Black Clay / Green Clay",
     "anchor_y": 1038.8, "orig_angle": -1.41},
]

# Soil layers — bottom (index 0) to top (index 5)
#
# Base values (METU laboratory, Öge 2008 Table 4.2; unit weights Özbay & Cabalar 2015):
#   gamma_sat   : saturated unit weight   (kN/m³)
#   gamma_unsat : unsaturated unit weight (kN/m³)
#   c_peak      : peak cohesion           (kPa)
#   c_res       : residual cohesion       (kPa)
#   phi_peak    : peak friction angle     (°)
#   phi_res     : residual friction angle (°)
#
# Constraint enforced during sampling: c_res <= c_peak, phi_res <= phi_peak
LAYERS = [
    {"id": 1, "name": "Green Clay",
     "gamma_sat": 16.63, "gamma_unsat": 11.39,
     "c_peak": 32.83, "c_res": 15.50,
     "phi_peak": 23.17, "phi_res": 16.00},
    {"id": 2, "name": "Black Clay",
     "gamma_sat": 16.59, "gamma_unsat": 10.50,
     "c_peak": 57.50, "c_res": 36.00,
     "phi_peak": 28.00, "phi_res": 25.00},
    {"id": 3, "name": "Lignite",
     "gamma_sat": 13.31, "gamma_unsat": 6.47,
     "c_peak": 48.00, "c_res": 32.40,
     "phi_peak": 32.80, "phi_res": 30.00},
    {"id": 4, "name": "Gyttja",
     "gamma_sat": 14.84, "gamma_unsat": 8.46,
     "c_peak": 59.00, "c_res": 41.30,
     "phi_peak": 31.80, "phi_res": 28.33},
    {"id": 5, "name": "Blue Clay",
     "gamma_sat": 18.41, "gamma_unsat": 13.79,
     "c_peak": 34.30, "c_res": 21.50,
     "phi_peak": 26.30, "phi_res": 21.00},
    {"id": 6, "name": "Loam",
     "gamma_sat": 18.53, "gamma_unsat": 13.72,
     "c_peak": 56.00, "c_res": 48.00,
     "phi_peak": 15.00, "phi_res": 11.00},
]

# All 6 property keys
PROP_KEYS = ("gamma_sat", "gamma_unsat", "c_peak", "c_res", "phi_peak", "phi_res")

LAYER_NAMES  = [l["name"] for l in LAYERS]
LAYER_COLORS = ["#8B6914", "#6B8E6B", "#4A7A4A",
                "#5B8FA8", "#7B9EC0", "#C8A96E"]
BND_COLORS   = ["#8B0000", "#B8860B", "#006400", "#00008B", "#4B0082"]

# Normalisation constants (fixed, so every channel lies in [0, 1]).
#
# Rule: largest base value across the six layers x 1.20 (the +20 % sampling
# headroom), rounded up. Checked on the v3 dataset (346 models): the largest
# normalised value is 0.999.
#
#   gamma_sat   : 18.53 x 1.2 = 22.24 -> 23.0   (v3 max 22.24)
#   gamma_unsat : 13.79 x 1.2 = 16.55 -> 17.0   (v3 max 16.54)
#   c_peak      : 59.00 x 1.2 = 70.80 -> 71.0   (v3 max 70.80)
#   c_res       : 48.00 x 1.2 = 57.60 -> 58.0   (v3 max 57.58)
#   phi_peak    : 32.80 x 1.2 = 39.36 -> 40.0   (v3 max 39.33)
#   phi_res     : 30.00 x 1.2 = 36.00 -> 36.0   (v3 max 35.95)

GAMMA_SAT_MAX   = 23.0   # kN/m3
GAMMA_UNSAT_MAX = 17.0   # kN/m3
C_PEAK_MAX      = 71.0   # kPa
C_RES_MAX       = 58.0   # kPa
PHI_PEAK_MAX    = 40.0   # deg
PHI_RES_MAX     = 36.0   # deg
PORE_P_MAX      = 1600.0 # kPa, optional channel 7 (pore pressure), see slim_raster.py
GAMMA_W         = 9.81   # kN/m3, as in Slide2

# Aliases used elsewhere in the codebase
GAMMA_MAX = GAMMA_SAT_MAX     # largest unit weight
C_MAX     = C_PEAK_MAX        # largest cohesion
PHI_MAX   = PHI_PEAK_MAX      # largest friction angle
RHO_MAX   = GAMMA_MAX         # backward-compat alias

# CNN channel definitions
CNN_CHANNELS = [
    {"idx": 0, "prop": "gamma_sat",   "norm": GAMMA_SAT_MAX,   "label": "γ_sat / 23",   "unit": "kN/m³"},
    {"idx": 1, "prop": "gamma_unsat", "norm": GAMMA_UNSAT_MAX, "label": "γ_unsat / 17", "unit": "kN/m³"},
    {"idx": 2, "prop": "c_peak",      "norm": C_PEAK_MAX,      "label": "c_peak / 71",  "unit": "kPa"},
    {"idx": 3, "prop": "c_res",       "norm": C_RES_MAX,       "label": "c_res / 58",   "unit": "kPa"},
    {"idx": 4, "prop": "phi_peak",    "norm": PHI_PEAK_MAX,    "label": "φ_peak / 40",  "unit": "°"},
    {"idx": 5, "prop": "phi_res",     "norm": PHI_RES_MAX,     "label": "φ_res / 36",   "unit": "°"},
]


# ─────────────────────────────────────────────────────────────────────────────
# USER SETTINGS  (edit freely)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Run ──────────────────────────────────────────────────────────────────
    n_samples   : int   = 300
    seed        : int   = 42
    output_dir  : str   = "output"

    # ── Randomisation ────────────────────────────────────────────────────────
    angle_min   : float = 5.0    # overall slope angle lower bound (°)
    angle_max   : float = 25.0   # overall slope angle upper bound (°)
    bnd_delta   : float = 1.0    # boundary dip perturbation ±  (°)
    prop_pct    : float = 0.20   # soil property variation ± 20 %

    # ── CNN grid ─────────────────────────────────────────────────────────────
    cell_size   : float = 10.0   # grid cell size (m); square pixels

    # ── Export flags ─────────────────────────────────────────────────────────
    export_dxf  : bool  = True
    export_png  : bool  = True
    export_tensor: bool = True
    png_dpi     : int   = 110
    zip_outputs : bool  = True
