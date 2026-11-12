"""Command-line interface.

    slopelem fos    MODEL.slim                       critical circle and FoS of one model
    slopelem circle MODEL.slim XC YC R               FoS of a given circle
    slopelem batch  FOLDER --out results.csv [--workers 2]
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .bishop import Circle, bishop_fos
from .model import SlopeModel
from .search import find_critical_circle

FIELDS = ["model", "fos", "found_by", "xc", "yc", "radius", "x_entry", "x_exit"]


def _ground_x(model: SlopeModel, xc: float, yc: float, r: float):
    """Entry/exit x of a circle with the ground (outermost crossings)."""
    xs = np.linspace(max(model.x_min, xc - r), min(model.x_max, xc + r), 20001)
    below = (yc - np.sqrt(np.maximum(r * r - (xs - xc) ** 2, 0))) < model.ground(xs)
    idx = np.where(below)[0]
    if not len(idx):
        raise ValueError("circle does not cut the slope")
    return float(xs[idx[0]]), float(xs[idx[-1]])


def _row(path: str):
    res = find_critical_circle(SlopeModel.from_file(path))
    c = res.circle
    return [Path(path).name, round(res.fos, 4), res.found_by, round(c.xc, 3), round(c.yc, 3),
            round(c.radius, 3), round(c.x_entry, 3), round(c.x_exit, 3)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="slopelem", description="Simplified Bishop LEM for Slide2 models")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fos", help="critical circle of one model")
    a.add_argument("model")
    b = sub.add_parser("circle", help="FoS of a given circle")
    b.add_argument("model"); b.add_argument("xc", type=float); b.add_argument("yc", type=float); b.add_argument("r", type=float)
    c = sub.add_parser("batch", help="critical circles of all .slim files in a folder")
    c.add_argument("folder"); c.add_argument("--out", required=True); c.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)

    if args.cmd == "fos":
        print(dict(zip(FIELDS, _row(args.model))))
    elif args.cmd == "circle":
        m = SlopeModel.from_file(args.model)
        x1, x2 = _ground_x(m, args.xc, args.yc, args.r)
        r = bishop_fos(m, Circle(args.xc, args.yc, args.r, x1, x2))
        print(f"FoS = {r.fos:.4f}  (entry x = {x1:.2f}, exit x = {x2:.2f}, min m_alpha = {r.min_m_alpha:.3f})")
    else:
        files = sorted(str(p) for p in Path(args.folder).glob("*.slim"))
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(FIELDS)
            if args.workers > 1:
                with ProcessPoolExecutor(args.workers) as ex:
                    for row in ex.map(_row, files):
                        w.writerow(row); fh.flush()
            else:
                for f in files:
                    w.writerow(_row(f)); fh.flush()
        print(f"wrote {len(files)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
