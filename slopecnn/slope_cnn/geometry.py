"""
slope_cnn/geometry.py
=====================
Constructs the slope perimeter polygon and layer boundary lines
for a given slope angle and boundary angle perturbations.

Public API
----------
    build_geometry(alpha_deg, bnd_deltas) -> GeometryResult
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np

from .config import (
    TOE_X, TOE_Y, BASE_Y, CREST_Y, RIGHT_X,
    H, L_ORIG, TOP_WIDTH, ANCHOR_X,
    SLOPE_FACE_REL, BOUNDARIES,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeometryResult:
    """All geometric objects for one slope realisation."""
    alpha_deg   : float                   # overall slope angle (°)
    new_L       : float                   # horizontal slope run (m)
    left_x      : float                   # x of left wall
    perimeter   : np.ndarray              # (N+1, 2) closed polygon
    bnd_lines   : Dict[str, np.ndarray]   # key → [[lx,ly],[rx,ry]]
    bnd_deltas  : Dict[str, float]        # key → perturbation (°)

    @property
    def toe(self) -> np.ndarray:
        return np.array([TOE_X, TOE_Y])

    @property
    def crest(self) -> np.ndarray:
        return np.array([TOE_X - self.new_L, CREST_Y])

    def bnd_angle(self, key: str) -> float:
        """Effective dip angle of a boundary (original + delta)."""
        orig = next(b["orig_angle"] for b in BOUNDARIES if b["key"] == key)
        return orig + self.bnd_deltas.get(key, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _seg_intersect(p1, p2, p3, p4):
    """Return intersection point of segments p1-p2 and p3-p4, or None."""
    d1, d2 = p2 - p1, p4 - p3
    cr = d1[0]*d2[1] - d1[1]*d2[0]
    if abs(cr) < 1e-10:
        return None
    t = ((p3[0]-p1[0])*d2[1] - (p3[1]-p1[1])*d2[0]) / cr
    u = ((p3[0]-p1[0])*d1[1] - (p3[1]-p1[1])*d1[0]) / cr
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return p1 + t*d1
    return None


def _build_perimeter(alpha_deg: float) -> tuple:
    """Build the outer slope polygon. Returns (perimeter, left_x, new_L, sf)."""
    new_L   = H / np.tan(np.radians(alpha_deg))
    scale_x = new_L / L_ORIG
    left_x  = TOE_X - new_L - TOP_WIDTH

    sf = np.column_stack([
        TOE_X + SLOPE_FACE_REL[:, 0] * scale_x,
        TOE_Y + SLOPE_FACE_REL[:, 1],
    ])

    pts = (
        [(left_x, BASE_Y), (RIGHT_X, BASE_Y),
         (RIGHT_X, TOE_Y), (TOE_X,   TOE_Y)]
        + [(p[0], p[1]) for p in sf]
        + [(left_x, CREST_Y)]
    )
    perimeter = np.array(pts + [pts[0]], dtype=np.float64)
    return perimeter, left_x, new_L, sf


def _build_boundary_lines(
    perimeter: np.ndarray,
    left_x: float,
    bnd_deltas: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """
    Extend each layer boundary to the left wall and the slope perimeter.

    Strategy
    --------
    - Left endpoint  : extrapolate to x = left_x
    - Right endpoint : shoot a ray rightward from left_pt, find first
                       perimeter intersection (skip the left-wall segment)
    """
    segs    = [(perimeter[i], perimeter[i+1])
               for i in range(len(perimeter)-1)]
    lw_idx  = len(segs) - 1   # left-wall segment index

    bnd_lines = {}
    for bnd in BOUNDARIES:
        ang = bnd["orig_angle"] + bnd_deltas.get(bnd["key"], 0.0)
        m   = np.tan(np.radians(ang))

        # Left endpoint on left wall
        y_l = bnd["anchor_y"] + (left_x - ANCHOR_X) * m
        lpt = np.array([left_x, y_l])

        # Right endpoint — first perimeter hit going rightward
        x_far   = RIGHT_X + 300
        ray_end = np.array([x_far, y_l + (x_far - left_x)*m])

        rpt = None;  best_d = np.inf
        for i, (p1, p2) in enumerate(segs):
            if i == lw_idx:
                continue
            pt = _seg_intersect(lpt, ray_end,
                                p1.astype(float), p2.astype(float))
            if pt is not None:
                d = np.linalg.norm(pt - lpt)
                if d > 0.1 and d < best_d:
                    best_d = d;  rpt = pt

        if rpt is None:   # fallback (should not happen within valid domain)
            x_fb = (left_x + TOE_X) / 2
            rpt  = np.array([x_fb, y_l + (x_fb - left_x)*m])

        bnd_lines[bnd["key"]] = np.array([lpt, rpt])

    return bnd_lines


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def build_geometry(alpha_deg: float, bnd_deltas: Dict[str, float]) -> GeometryResult:
    """
    Build the complete geometric description of one slope realisation.

    Parameters
    ----------
    alpha_deg  : overall slope angle in degrees
    bnd_deltas : {bnd_key: perturbation_degrees}

    Returns
    -------
    GeometryResult  dataclass with perimeter, boundary lines, and metadata
    """
    perimeter, left_x, new_L, _ = _build_perimeter(alpha_deg)
    bnd_lines                    = _build_boundary_lines(
                                       perimeter, left_x, bnd_deltas)

    return GeometryResult(
        alpha_deg  = alpha_deg,
        new_L      = new_L,
        left_x     = left_x,
        perimeter  = perimeter,
        bnd_lines  = bnd_lines,
        bnd_deltas = bnd_deltas,
    )


def crossing_points(geo: GeometryResult):
    """
    Return in-domain crossing points between adjacent boundaries.
    Used for visualisation markers.
    """
    crosses = []
    params  = [
        (b["anchor_y"],
         np.tan(np.radians(geo.bnd_angle(b["key"]))))
        for b in BOUNDARIES
    ]
    for i in range(len(params) - 1):
        y1, m1 = params[i];  y2, m2 = params[i+1]
        if abs(m1 - m2) < 1e-10:
            continue
        xc = ANCHOR_X + (y1 - y2) / (m2 - m1)
        yc = y1 + m1 * (xc - ANCHOR_X)
        if geo.left_x <= xc <= RIGHT_X + 400 and BASE_Y <= yc <= CREST_Y:
            crosses.append((xc, yc))
    return crosses
