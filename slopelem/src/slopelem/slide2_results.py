"""Reader for Slide2 analysis results stored in a computed ``.slim`` file (``.s01``).

Used to verify the solver against Slide2: the global-minimum surface of each
method, and the per-slice data of the Bishop critical surface.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .bishop import Circle


@dataclass(frozen=True)
class Slide2Result:
    global_minimum: Dict[str, Circle]      # method name -> circle
    global_fos: Dict[str, float]           # method name -> FoS
    bishop_slice_x: np.ndarray             # Bishop slice boundaries (x)
    slice_data: Dict[str, np.ndarray]      # e.g. "Slice Weight", "Base Cohesion" (first column = Bishop)


def read_slide2_result(path: str | Path) -> Optional[Slide2Result]:
    """Read Slide2 results from a computed ``.slim``; ``None`` if it holds no results."""
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".s01")]
        if not names:
            return None
        lines = z.read(names[0]).decode("latin-1").splitlines()

    i = lines.index("* Global Minimum FS (xc,yc,r,x1,y1,x2,y2,fs,name)") + 1
    circles, fos = {}, {}
    while not lines[i].startswith("*"):
        p = lines[i].split()
        name = " ".join(p[8:])
        xc, yc, r, x1, _y1, x2, _y2, fs = map(float, p[:8])
        circles[name] = Circle(xc, yc, r, x1, x2)
        fos[name] = fs
        i += 1

    k = next(j for j, l in enumerate(lines) if l.startswith("* minimum slice info") and "bishop" in l) + 1
    xb = []
    while not lines[k].startswith("*"):
        xb.append(float(lines[k].split()[0]))
        k += 1

    data = {}
    for j, l in enumerate(lines):
        if l == "* name":
            m, rows = j + 5, []
            while m < len(lines) and not lines[m].startswith("*"):
                rows.append([float(v) for v in lines[m].split()])
                m += 1
            data[lines[j + 1]] = np.array(rows)
    return Slide2Result(circles, fos, np.array(xb), data)
