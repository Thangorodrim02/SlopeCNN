"""Slope model: geometry, materials and water table read from a Slide2 file.

Conventions
-----------
* Units: metres, kN/m3, kPa, degrees.
* The slope faces to the right (crest on the left, toe on the right), as in the
  Slide2 models of the SlopeCNN dataset ("direction: left to right").
* Materials are the six soil types ``soil1`` ... ``soil6`` of the Slide2 file.
"""

from __future__ import annotations

import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

GAMMA_W: float = 9.81
"""Unit weight of water (kN/m3), as used by Slide2 (``gammaw: 9.81``)."""


@dataclass(frozen=True)
class Material:
    """Anisotropic Mohr-Coulomb material (Slide2 "Anisotropic Strength").

    Strength depends on the angle ``delta`` between the slice base and the
    1-direction of the material (Slide2 "Angle (ccw to 1)")::

        c(delta)   = c1 * cos^2(delta) + c2 * sin^2(delta)
        phi(delta) = phi1 * cos^2(delta) + phi2 * sin^2(delta)

    Attributes
    ----------
    c1, phi1 : strength along the 1-direction (kPa, deg); residual in SlopeCNN.
    c2, phi2 : strength perpendicular to it (kPa, deg); peak in SlopeCNN.
    angle    : orientation of the 1-direction, counter-clockwise from +x (deg).
    gamma    : unit weight above the water table (kN/m3), Slide2 ``uw``.
    gamma_sat: unit weight below the water table (kN/m3), Slide2 ``uwbwt``.
    hu       : Hu pore-pressure coefficient (-).
    """

    c1: float
    phi1: float
    c2: float
    phi2: float
    angle: float
    gamma: float
    gamma_sat: float
    hu: float

    def strength(self, base_angle_rad: float) -> Tuple[float, float]:
        """Return ``(c, phi_deg)`` for a slice base inclined at ``base_angle_rad``.

        ``base_angle_rad`` is the angle of the base line measured from +x with x
        increasing (radians).
        """
        delta = base_angle_rad - math.radians(self.angle)
        cos2, sin2 = math.cos(delta) ** 2, math.sin(delta) ** 2
        return self.c1 * cos2 + self.c2 * sin2, self.phi1 * cos2 + self.phi2 * sin2


def read_sli_text(path: str | Path) -> str:
    """Return the text of the ``.sli`` model inside a ``.slim`` archive (or a bare ``.sli``)."""
    path = str(path)
    if path.lower().endswith(".slim"):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".sli"))
            return z.read(name).decode("latin-1")
    with open(path, encoding="latin-1") as fh:
        return fh.read()


class SlopeModel:
    """Slope geometry, materials and water table, prepared for slice integration.

    The triangulated material cells of the Slide2 model are pre-sampled into
    vertical columns every ``dx`` metres; each column stores the material
    intervals ``(y_low, y_high, material_id)`` met by a vertical line.
    """

    def __init__(self, sli_text: str, dx: float = 0.5, n_materials: int = 6):
        body = sli_text.split("\nvertices:")[1]
        self.vertices: Dict[int, Tuple[float, float]] = {
            int(i): (float(x), float(y))
            for i, x, y in re.findall(
                r"^\s+(\d+) x: (\S+)\s+y: (\S+)", body.split("profile_vertices:")[0], re.M
            )
        }
        cells = re.findall(
            r"vertices: \[([\d,]+)\] material: soil(\d+)",
            sli_text.split("\ncells:")[1].split("profile_cells:")[0],
        )
        self.cells: List[Tuple[np.ndarray, int]] = [
            (np.array([self.vertices[int(i)] for i in s.split(",")]), int(m)) for s, m in cells
        ]
        self.materials: Dict[int, Material] = {
            k: self._parse_material(sli_text, k) for k in range(1, n_materials + 1)
        }

        wt = re.search(r"water table:\s*\n.*?vertices: \[([\d,]+)\]", sli_text).group(1)
        pts = np.array([self.vertices[int(i)] for i in wt.split(",")])
        pts = pts[np.argsort(pts[:, 0])]
        self.water_x, self.water_y = pts[:, 0], pts[:, 1]

        sl = re.search(r"slope limits:\s*\n\s*x1: (\S+) y1: (\S+) x2: (\S+) y2: (\S+)", sli_text)
        self.slope_limits: Tuple[float, float] = (float(sl.group(1)), float(sl.group(3)))

        xy = np.array(list(self.vertices.values()))
        self.x_min, self.x_max = float(xy[:, 0].min()), float(xy[:, 0].max())
        self.y_min = float(xy[:, 1].min())
        self._build_columns(dx)

    # ------------------------------------------------------------------ factories
    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "SlopeModel":
        """Read a Slide2 ``.slim`` (or ``.sli``) file."""
        return cls(read_sli_text(path), **kwargs)

    # ------------------------------------------------------------------ parsing
    @staticmethod
    def _parse_material(sli_text: str, k: int) -> Material:
        line = re.search(rf"^  soil{k} = (.*)$", sli_text, re.M).group(1).split()
        d = dict(zip((t.rstrip(":") for t in line[0::2]), line[1::2]))
        return Material(
            c1=float(d["c"]),
            phi1=float(d["phi"]),
            c2=float(d.get("c2", d["c"])),
            phi2=float(d.get("phi2", d["phi"])),
            angle=float(d.get("ang", 0.0)),
            gamma=float(d["uw"]),
            gamma_sat=float(d.get("uwbwt", d["uw"])),
            hu=float(d["water"]),
        )

    def _build_columns(self, dx: float) -> None:
        xs = np.arange(self.x_min + dx / 2, self.x_max, dx)
        self.column_x = xs
        self.column_dx = dx
        cols: List[list] = [[] for _ in xs]
        for tri, mat in self.cells:
            x0, x1 = tri[:, 0].min(), tri[:, 0].max()
            idx = np.where((xs > x0) & (xs < x1))[0]
            if not len(idx):
                continue
            X = xs[idx]
            ys = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                (xa, ya), (xb, yb) = tri[a], tri[b]
                if abs(xb - xa) < 1e-12:
                    ys.append(np.full(len(X), np.nan))
                    continue
                t = (X - xa) / (xb - xa)
                y = ya + t * (yb - ya)
                y[(t < -1e-9) | (t > 1 + 1e-9)] = np.nan
                ys.append(y)
            ys = np.array(ys)
            lo, hi = np.nanmin(ys, 0), np.nanmax(ys, 0)
            for j, i in enumerate(idx):
                if hi[j] - lo[j] > 1e-9:
                    cols[i].append((lo[j], hi[j], mat))
        self.columns = [sorted(c) for c in cols]
        self.column_top = np.array([c[-1][1] if c else np.nan for c in self.columns])

    # ------------------------------------------------------------------ queries
    def ground(self, x):
        """Ground-surface elevation at ``x`` (scalar or array)."""
        return np.interp(x, self.column_x, self.column_top)

    def water_table(self, x):
        """Water-table elevation at ``x`` (scalar or array)."""
        return np.interp(x, self.water_x, self.water_y)

    def column(self, x: float, y_base: float) -> Tuple[float, int]:
        """Weight per unit width of the soil column above ``y_base`` at ``x``, and the
        material at ``y_base``.

        Soil above the water table uses ``gamma``; soil below it uses ``gamma_sat``.
        """
        i = int(np.clip(round((x - self.column_x[0]) / self.column_dx), 0, len(self.column_x) - 1))
        y_wt = self.water_table(x)
        weight, base_mat = 0.0, None
        for lo, hi, m in self.columns[i]:
            if hi <= y_base:
                continue
            if base_mat is None and lo <= y_base + 1e-9:
                base_mat = m
            a = max(lo, y_base)
            mat = self.materials[m]
            below = max(0.0, min(hi, y_wt) - a)
            above = (hi - a) - below
            weight += below * mat.gamma_sat + above * mat.gamma
        if base_mat is None and self.columns[i]:
            base_mat = self.columns[i][0][2]
        return weight, base_mat
