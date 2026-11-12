"""
slope_cnn/slim_raster.py
========================
Builds the CNN input tensor directly from a Slide2 model file (.slim).

Why read the Slide2 file instead of rebuilding the slope from the Excel row
---------------------------------------------------------------------------
The FoS label of every v3 geometry was computed from its .slim file. Reading
the same file makes the image show exactly what the label was computed from:

  * the layer of every cell comes from the model's own triangulated material
    cells (no re-drawn boundaries, so no mismatch where boundaries cross);
  * the soil properties are the values stored in the model's materials;
  * the water table is the model's own piezometric line (optional channel).

Grid
----
The same fixed global grid as grid.build_global_grid (10 m square cells, toe
always on the same pixel). With angle_min = 5 deg it is 42 rows x 243 cols and
holds every v3 slope without cropping.

Channels (CNN_CHANNELS order, each divided by its fixed maximum in config.py)
------------------------------------------------------------------------------
  0  gamma_sat   (Slide2 'uwbwt')          3  c_res    (Slide2 'c',  slot 1)
  1  gamma_unsat (Slide2 'uw')             4  phi_peak (Slide2 'phi2', slot 2)
  2  c_peak      (Slide2 'c2', slot 2)     5  phi_res  (Slide2 'phi', slot 1)
  6  pore pressure (optional): u = Hu * gamma_w * max(0, y_wt - y),
     the same rule Slide2 uses, divided by PORE_P_MAX.
Cells outside the soil are 0 in every channel.

Material mapping in the v3 models: soil1 Loam, soil2 Blue Clay, soil3 Gyttja,
soil4 Lignite, soil5 Black Clay, soil6 Green Clay, i.e. layer id = 7 - soilK
(layer ids as in config.LAYERS: 1 Green Clay ... 6 Loam).

Public API
----------
    slim_to_tensor(path, grid, pore_pressure=False) -> (tensor, layer_grid)
"""

from typing import Tuple

import numpy as np

from .config import CNN_CHANNELS, GAMMA_W, PORE_P_MAX
from .grid import GridInfo

try:
    from slopelem import SlopeModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "slim_raster needs the slopelem package: "
        "pip install -e ../slopelem  (from the slopecnn folder)") from exc


def _layer_props(model: "SlopeModel") -> dict:
    """{layer_id: {prop: value}} read from the Slide2 materials."""
    props = {}
    for k, mat in model.materials.items():
        props[7 - k] = {
            "gamma_sat":   mat.gamma_sat,
            "gamma_unsat": mat.gamma,
            "c_peak":      mat.c2,
            "c_res":       mat.c1,
            "phi_peak":    mat.phi2,
            "phi_res":     mat.phi1,
            "hu":          mat.hu,
        }
    return props


def slim_to_tensor(path: str, grid: GridInfo,
                   pore_pressure: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rasterise one Slide2 model onto the fixed grid.

    Returns
    -------
    tensor     : float32 array (6, NY, NX), or (7, NY, NX) with pore_pressure
    layer_grid : int8 array (NY, NX), layer id 1..6 (1 Green Clay ... 6 Loam),
                 0 outside the soil
    """
    model = SlopeModel.from_file(path)
    props = _layer_props(model)
    ny, nx = grid.ny, grid.nx

    layer = np.zeros((ny, nx), dtype=np.int8)
    for j, x in enumerate(grid.xs):
        if x <= model.x_min or x >= model.x_max:
            continue
        i = int(np.clip(round((x - model.column_x[0]) / model.column_dx),
                        0, len(model.column_x) - 1))
        for lo, hi, soil in model.columns[i]:
            rows = (grid.ys > lo) & (grid.ys <= hi)   # a centre on the model base is outside
            layer[rows, j] = 7 - soil

    n_ch = len(CNN_CHANNELS) + (1 if pore_pressure else 0)
    tensor = np.zeros((n_ch, ny, nx), dtype=np.float32)
    for lid in range(1, 7):
        mask = layer == lid
        if not mask.any():
            continue
        for ch in CNN_CHANNELS:
            tensor[ch["idx"]][mask] = props[lid][ch["prop"]] / ch["norm"]

    if pore_pressure:
        y_wt = model.water_table(grid.xs)[None, :]             # (1, NX)
        depth = np.clip(y_wt - grid.ys[:, None], 0.0, None)    # (NY, NX)
        hu = np.zeros((ny, nx), dtype=np.float32)
        for lid in range(1, 7):
            hu[layer == lid] = props[lid]["hu"]
        tensor[len(CNN_CHANNELS)] = hu * GAMMA_W * depth / PORE_P_MAX

    return tensor, layer
