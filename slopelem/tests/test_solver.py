"""Verification tests for slopelem.

Run with:  pytest            (fast tests, ~1 s)
           pytest -m slow    (full critical-circle search, ~1 min)
"""

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from slopelem import (
    Circle,
    SlopeModel,
    bishop_fos,
    circle_through_ground_points,
    find_critical_circle,
    is_admissible,
    read_slide2_result,
)

DATA = Path(__file__).parent / "data"
LABELS = Path(__file__).parents[1] / "data" / "labels_v3_346.csv"


def _labels():
    with open(LABELS) as fh:
        return {int(r["geometry_id"]): r for r in csv.DictReader(fh)}


# ---------------------------------------------------------------- Slide2 agreement
def test_slice_data_match_slide2():
    """Same slices as Slide2's critical circle -> same base material, c, phi and FoS."""
    path = DATA / "old_Feed142_with_slide2_results.slim"
    model = SlopeModel.from_file(path)
    res = read_slide2_result(path)
    c = res.global_minimum["bishop simplified"]
    out = bishop_fos(model, c, x_bounds=res.bishop_slice_x, sub_columns=8)
    d = res.slice_data
    mat = np.array([s.material for s in out.slices])
    coh = np.array([s.cohesion for s in out.slices])
    phi = np.array([s.phi for s in out.slices])
    w = np.array([s.weight for s in out.slices])
    assert np.array_equal(mat, d["Base Material"][:, 0].astype(int))
    assert np.allclose(coh, d["Base Cohesion"][:, 0], atol=1e-3)
    assert np.allclose(phi, np.degrees(d["Base Friction Angle"][:, 0]), atol=1e-3)
    assert np.median(np.abs(w / d["Slice Weight"][:, 0] - 1)) < 0.002
    assert abs(out.fos / res.global_fos["bishop simplified"] - 1) < 0.001


@pytest.mark.parametrize("gid", [2, 232, 245])
def test_fos_on_slide2_critical_circle(gid):
    """FoS of Slide2's own critical circle (anisotropic models) within 0.3 %."""
    path = DATA / f"slide2_computed_Feed{gid}.slim"
    res = read_slide2_result(path)
    c = res.global_minimum["bishop simplified"]
    fos = bishop_fos(SlopeModel.from_file(path), c).fos
    assert abs(fos / res.global_fos["bishop simplified"] - 1) < 0.003


# ---------------------------------------------------------------- admissibility rules
def test_reverse_curvature_rejected():
    """Slide2 error -114: centre below the higher ground intersection."""
    model = SlopeModel.from_file(DATA / "Feed245.slim")
    c = circle_through_ground_points(model, 462.0, 492.0, math.radians(70))
    bad = Circle(c.xc, max(model.ground(462.0), model.ground(492.0)) - 0.5, c.radius, 462.0, 492.0)
    assert not is_admissible(model, bad)


# ---------------------------------------------------------------- dataset labels
@pytest.mark.parametrize("gid", [7, 163, 245, 288, 294])
def test_labels_reproducible_from_published_circle(gid):
    """Each published label = Bishop FoS of its published circle (3 decimals)."""
    r = _labels()[gid]
    model = SlopeModel.from_file(DATA / f"Feed{gid}.slim")
    c = circle_through_ground_points(model, float(r["x_entry"]), float(r["x_exit"]), float(r["beta_rad"]))
    assert is_admissible(model, c)
    assert round(bishop_fos(model, c).fos, 3) == pytest.approx(float(r["FoS"]), abs=1e-9)


@pytest.mark.slow
def test_search_finds_label():
    """The full search protocol recovers the published minimum (Feed7)."""
    res = find_critical_circle(SlopeModel.from_file(DATA / "Feed7.slim"))
    assert abs(res.fos - float(_labels()[7]["FoS"])) < 0.0015
