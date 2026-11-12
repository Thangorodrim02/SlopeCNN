"""
slope_cnn/grid.py
=================
Defines the fixed global CNN grid and builds normalised input tensors.

Design choice: Fixed physical domain — RECTANGULAR grid
--------------------------------------------------------
  • One shared coordinate system for ALL geometries
  • 10 m × 10 m SQUARE cells (Config.cell_size, default 10.0) → physical
    scale preserved, no geometric distortion. Confirmed via
    output/tensors/domain_info.npz: cell_size=10.0, shape (6, 42, 127).
  • Domain aspect ratio is such that the grid is RECTANGULAR (42 rows x
    127 cols for the Section 4 cross-section) NOT square — squeezing to
    NxN would distort the slope angle information
  • Toe always at the same pixel (fixed toe_col/toe_row) → spatial anchor
    for the CNN
  • Shallow slopes fill more of the grid; steep slopes leave left zeros
  • Slope angle is implicitly encoded in the polygon shape

Sorted-boundary layer assignment
---------------------------------
  At cell (x, y): compute y_b(x) for each of the 5 boundaries.
  layer_index = count of boundaries with y_b(x) < y  (0-5).
  Sorting by y_b at runtime preserves all 6 layers even when
  two boundaries cross.

Normalisation  (data-tight, per-property maxima)
-------------------------------------------------
  CH0  γ_sat   / 17 kN/m³     CH1  γ_unsat / 23 kN/m³
  CH2  c_peak  / 71 kPa       CH3  c_res   / 58 kPa
  CH4  φ_peak  / 40 °         CH5  φ_res   / 36 °
  cells outside polygon = 0.0

Public API
----------
    build_global_grid(config)  -> GridInfo
    build_tensor(geo, props, grid_info) -> np.ndarray  shape (6, NY, NX)
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from matplotlib.path import Path

from .config import (
    TOE_X, TOE_Y, BASE_Y, CREST_Y, RIGHT_X,
    H, TOP_WIDTH, BOUNDARIES,
    GAMMA_MAX, C_MAX, PHI_MAX, CNN_CHANNELS,
    Config,
)
from .geometry import GeometryResult


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GridInfo:
    """Immutable description of the fixed global CNN raster grid."""
    x_min    : float
    x_max    : float
    y_min    : float
    y_max    : float
    cell     : float          # cell size (m)
    nx       : int            # number of columns
    ny       : int            # number of rows
    toe_col  : int            # column index of slope toe (fixed)
    toe_row  : int            # row index  of slope toe (fixed)
    xs       : np.ndarray     # (nx,) cell-centre x coordinates
    ys       : np.ndarray     # (ny,) cell-centre y coordinates
    X        : np.ndarray     # (ny, nx) meshgrid x
    Y        : np.ndarray     # (ny, nx) meshgrid y

    @property
    def shape(self):
        return (self.ny, self.nx)

    @property
    def tensor_shape(self):
        return (len(CNN_CHANNELS), self.ny, self.nx)

    def summary(self) -> str:
        ch_lines = "  ".join(f"CH{c['idx']}={c['label']}" for c in CNN_CHANNELS)
        lines = [
            "CNN INPUT GRID  (fixed physical domain — RECTANGULAR, square pixels)",
            f"  Grid size  : {self.nx} cols × {self.ny} rows  (W × H)  — aspect ratio {self.nx/self.ny:.1f}:1",
            f"  Tensor     : ({len(CNN_CHANNELS)}, {self.ny}, {self.nx})  float32  [rectangular, NOT square]",
            f"  Domain X   : [{self.x_min:.1f}, {self.x_max:.1f}] m  (span {self.x_max-self.x_min:.0f} m)",
            f"  Domain Y   : [{self.y_min:.1f}, {self.y_max:.1f}] m  (span {self.y_max-self.y_min:.0f} m)",
            f"  Cell size  : {self.cell} m × {self.cell} m  (square pixels — no geometric distortion)",
            f"  Toe pixel  : col={self.toe_col}, row={self.toe_row}"
            f"  (fixed for ALL geometries)",
            f"  Channels   : {ch_lines}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialisable metadata for domain_info.npz."""
        return dict(
            x_min=self.x_min, x_max=self.x_max,
            y_min=self.y_min, y_max=self.y_max,
            cell_size=self.cell, nx=self.nx, ny=self.ny,
            toe_col=self.toe_col, toe_row=self.toe_row,
            n_channels=len(CNN_CHANNELS),
            gamma_max=GAMMA_MAX, c_max=C_MAX, phi_max=PHI_MAX,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def build_global_grid(config: Config) -> GridInfo:
    """
    Compute the fixed global grid that covers every possible geometry.

    The x-extent is determined by the shallowest slope (angle_min),
    which produces the widest left extension.  The y-extent is fixed
    (BASE_Y to CREST_Y with margins). Cell size comes from
    config.cell_size (default 10.0 m).
    """
    lx_widest = TOE_X - H / np.tan(np.radians(config.angle_min)) - TOP_WIDTH
    x_min = lx_widest - 20
    x_max = RIGHT_X   + 20
    y_min = BASE_Y    - 25
    y_max = CREST_Y   + 45

    cell = config.cell_size
    nx   = int(np.ceil((x_max - x_min) / cell))
    ny   = int(np.ceil((y_max - y_min) / cell))

    xs = x_min + (np.arange(nx) + 0.5) * cell
    ys = y_min + (np.arange(ny) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)

    toe_col = int((TOE_X - x_min) / cell)
    toe_row = int((TOE_Y - y_min) / cell)

    return GridInfo(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        cell=cell, nx=nx, ny=ny,
        toe_col=toe_col, toe_row=toe_row,
        xs=xs, ys=ys, X=X, Y=Y,
    )


def build_tensor(
    geo   : GeometryResult,
    props : Dict[int, Dict[str, float]],
    grid  : GridInfo,
) -> np.ndarray:
    """
    Build a normalised (6, NY, NX) float32 tensor for one geometry.

    Channels (CNN_CHANNELS order)
    ──────────────────────────────
      CH 0 : γ_sat   / 17    saturated unit weight
      CH 1 : γ_unsat / 23    unsaturated unit weight
      CH 2 : c_peak  / 71    peak cohesion
      CH 3 : c_res   / 58    residual cohesion
      CH 4 : φ_peak  / 40    peak friction angle
      CH 5 : φ_res   / 36    residual friction angle

    Cells outside the slope polygon = 0.0
    """
    X, Y   = grid.X, grid.Y
    NY, NX = grid.ny, grid.nx
    N_CH   = len(CNN_CHANNELS)

    # ── Inside mask ──────────────────────────────────────────────────────
    pts_flat = np.column_stack([X.ravel(), Y.ravel()])
    inside   = Path(geo.perimeter).contains_points(pts_flat).reshape(NY, NX)

    # ── Sorted-boundary layer index  (0=Green Clay … 5=Loam) ─────────────
    bnd_y = np.empty((5, NY, NX), dtype=np.float32)
    for i, bnd in enumerate(BOUNDARIES):
        bl = geo.bnd_lines[bnd["key"]]
        x0, y0 = bl[0];  x1, y1 = bl[1]
        m = float((y1-y0)/(x1-x0)) if abs(x1-x0) > 1e-6 else 0.0
        bnd_y[i] = (y0 + m * (X - x0)).astype(np.float32)

    layer = np.sum(
        bnd_y < Y[np.newaxis].astype(np.float32), axis=0
    ).astype(np.int8)   # 0-5

    # ── Build one channel per property ───────────────────────────────────
    channels = np.zeros((N_CH, NY, NX), dtype=np.float32)

    for mat in range(6):           # mat 0..5 → layer_id 1..6
        mask = inside & (layer == mat)
        lid  = mat + 1
        for ch_def in CNN_CHANNELS:
            val = props[lid][ch_def["prop"]]
            channels[ch_def["idx"]][mask] = val / ch_def["norm"]

    return channels   # (6, NY, NX)


def get_layer_grid(geo: GeometryResult, grid: GridInfo) -> np.ndarray:
    """
    Return the raw integer layer index grid (0-5, -1 outside).
    Useful for visualisation without needing property values.
    """
    X, Y   = grid.X, grid.Y
    NY, NX = grid.ny, grid.nx

    pts_flat = np.column_stack([X.ravel(), Y.ravel()])
    inside   = Path(geo.perimeter).contains_points(pts_flat).reshape(NY, NX)

    bnd_y = np.empty((5, NY, NX), dtype=np.float32)
    for i, bnd in enumerate(BOUNDARIES):
        bl = geo.bnd_lines[bnd["key"]]; x0,y0=bl[0]; x1,y1=bl[1]
        m = float((y1-y0)/(x1-x0)) if abs(x1-x0) > 1e-6 else 0.
        bnd_y[i] = (y0 + m*(X-x0)).astype(np.float32)

    layer = np.sum(bnd_y < Y[np.newaxis].astype(np.float32),
                   axis=0).astype(np.int8)
    return np.where(inside, layer, -1)
