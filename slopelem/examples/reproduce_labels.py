"""Recompute every published FoS label from its published critical circle.

    python examples/reproduce_labels.py MODELS_FOLDER [data/labels_v3_346.csv]

MODELS_FOLDER holds the Slide2 models Feed1.slim ... Feed346.slim.
"""

import csv
import sys
from pathlib import Path

from slopelem import SlopeModel, bishop_fos, circle_through_ground_points, is_admissible


def main(folder: str, labels: str) -> int:
    n = bad = 0
    with open(labels) as fh:
        for r in csv.DictReader(fh):
            path = Path(folder) / f"Feed{r['geometry_id']}.slim"
            if not path.exists():
                continue
            model = SlopeModel.from_file(path)
            c = circle_through_ground_points(model, float(r["x_entry"]), float(r["x_exit"]), float(r["beta_rad"]))
            fos = round(bishop_fos(model, c).fos, 3)
            ok = is_admissible(model, c) and fos == float(r["FoS"])
            n += 1
            bad += not ok
            if not ok:
                print(f"Feed{r['geometry_id']}: label {r['FoS']}, recomputed {fos}")
    print(f"{n - bad} of {n} labels reproduced exactly")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data/labels_v3_346.csv"))
