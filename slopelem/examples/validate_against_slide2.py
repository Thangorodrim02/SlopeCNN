"""Compare the labels with Slide2 for models that were computed and saved in Slide2.

    python examples/validate_against_slide2.py COMPUTED_FOLDER [data/labels_v3_346.csv]

COMPUTED_FOLDER holds FeedN.slim files saved after running them in Slide2 (with
results). For each model the script reports Slide2's global-minimum Bishop FoS, the
label, and the solver's FoS on Slide2's own critical circle.
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np

from slopelem import SlopeModel, bishop_fos, read_slide2_result


def main(folder: str, labels: str) -> None:
    with open(labels) as fh:
        lab = {int(r["geometry_id"]): float(r["FoS"]) for r in csv.DictReader(fh)}
    diffs = []
    print(f"{'model':>9} {'Slide2':>8} {'label':>7} {'diff %':>7} {'solver on Slide2 circle':>24}")
    for path in sorted(Path(folder).glob("*.slim"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        res = read_slide2_result(path)
        gid = int(re.sub(r"\D", "", path.stem))
        if res is None or gid not in lab:
            continue
        s2 = res.global_fos["bishop simplified"]
        own = bishop_fos(SlopeModel.from_file(path), res.global_minimum["bishop simplified"]).fos
        d = (lab[gid] / s2 - 1) * 100
        diffs.append(d)
        print(f"{path.stem:>9} {s2:8.4f} {lab[gid]:7.3f} {d:7.3f} {own:24.4f}")
    if diffs:
        d = np.array(diffs)
        print(f"\n{len(d)} models: mean diff {d.mean():.3f} %, mean |diff| {np.abs(d).mean():.3f} %, "
              f"max |diff| {np.abs(d).max():.3f} %")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data/labels_v3_346.csv")
