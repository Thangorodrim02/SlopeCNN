"""Simplified Bishop factor of safety for circular slip surfaces.

For a circle of centre (xc, yc) and radius R cutting the ground at x_entry < x_exit,
the sliding mass is divided into vertical slices. For slice i::

    FoS = sum[ (c_i b_i + (W_i - u_i b_i) tan(phi_i)) / m_alpha_i ] / sum[ W_i sin(alpha_i) ]
    m_alpha_i = cos(alpha_i) * (1 + tan(alpha_i) tan(phi_i) / FoS)

solved by fixed-point iteration. alpha_i is the base inclination, positive where the
base rises towards the crest (driving), c_i and phi_i come from the anisotropic
strength of the base material, and the pore pressure is u_i = Hu * gamma_w * h_i,
with h_i the vertical depth of the base midpoint below the water table.

Slide2 admissibility rules reproduced here (Slide2 error codes):
  -112  m_alpha < 0.2 in the final iteration  -> surface rejected
  -114  reverse curvature (centre below the higher ground intersection) -> rejected
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .model import GAMMA_W, SlopeModel

M_ALPHA_MIN = 0.2


@dataclass(frozen=True)
class Circle:
    """Circular slip surface between two ground points (x_entry < x_exit)."""

    xc: float
    yc: float
    radius: float
    x_entry: float
    x_exit: float

    def base_y(self, x):
        """Elevation of the lower arc at ``x``."""
        return self.yc - np.sqrt(np.maximum(self.radius**2 - (np.asarray(x) - self.xc) ** 2, 0))


@dataclass(frozen=True)
class SliceData:
    """Quantities of one slice (base midpoint values)."""

    x_left: float
    x_right: float
    weight: float          # kN/m
    alpha: float           # rad, positive = driving
    width: float           # m
    cohesion: float        # kPa
    phi: float             # deg
    pore_pressure: float   # kPa
    material: int


@dataclass(frozen=True)
class BishopResult:
    fos: float
    slices: List[SliceData]
    min_m_alpha: float
    converged: bool


def is_admissible(model: SlopeModel, circle: Circle, n_check: int = 40) -> bool:
    """Geometric admissibility of a circle in the model.

    The arc must lie inside the model (above its base, within its x-range), stay
    below the ground between entry and exit, and not have reverse curvature
    (Slide2 error -114).
    """
    xs = np.linspace(circle.x_entry, circle.x_exit, n_check + 2)[1:-1]
    r2 = circle.radius**2 - (xs - circle.xc) ** 2
    yb = circle.yc - np.sqrt(np.maximum(r2, 0))
    if np.any(r2 < 0):
        return False
    if np.any(yb < model.y_min + 0.5) or circle.x_entry < model.x_min or circle.x_exit > model.x_max:
        return False
    if circle.yc < max(model.ground(circle.x_entry), model.ground(circle.x_exit)):
        return False
    return bool(np.all(yb <= model.ground(xs) + 0.01))


def _slices(model: SlopeModel, circle: Circle, x_bounds: np.ndarray, sub: int) -> List[SliceData]:
    xc, yc, R = circle.xc, circle.yc, circle.radius
    yb = yc - np.sqrt(np.maximum(R * R - (x_bounds - xc) ** 2, 0))
    out: List[SliceData] = []
    for i in range(len(x_bounds) - 1):
        x0, x1 = x_bounds[i], x_bounds[i + 1]
        b = x1 - x0
        if b <= 0:
            continue
        weight = 0.0                         # mid-point rule over `sub` sub-columns
        for s in range(sub):
            xs = x0 + (s + 0.5) * b / sub
            ys = yc - math.sqrt(max(R * R - (xs - xc) ** 2, 0))
            weight += model.column(xs, ys)[0] * b / sub
        xm = 0.5 * (x0 + x1)
        ym = yc - math.sqrt(max(R * R - (xm - xc) ** 2, 0))
        _, m = model.column(xm, ym + 1e-6)
        mat = model.materials[m]
        theta = math.atan2(yb[i + 1] - yb[i], x1 - x0)   # base angle, x increasing
        c, phi = mat.strength(theta)
        u = mat.hu * GAMMA_W * max(0.0, model.water_table(xm) - ym)
        out.append(SliceData(x0, x1, weight, -theta, b, c, phi, u, m))
    return out


def bishop_fos(
    model: SlopeModel,
    circle: Circle,
    n_slices: int = 200,
    sub_columns: int = 3,
    x_bounds: Optional[Sequence[float]] = None,
    apply_m_alpha_check: bool = True,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> BishopResult:
    """Simplified Bishop FoS of ``circle``.

    Parameters
    ----------
    n_slices      : number of equal-width slices (ignored if ``x_bounds`` is given).
    sub_columns   : sub-columns per slice for the weight integration.
    x_bounds      : explicit slice boundaries (e.g. Slide2's, for verification).
    apply_m_alpha_check : reject the surface (FoS = nan) if min m_alpha < 0.2.

    Returns a :class:`BishopResult`; ``fos`` is ``nan`` for an invalid surface.
    """
    xb = np.linspace(circle.x_entry, circle.x_exit, n_slices + 1) if x_bounds is None else np.asarray(x_bounds, float)
    sl = _slices(model, circle, xb, sub_columns)
    if not sl:
        return BishopResult(float("nan"), sl, float("nan"), False)
    W = np.array([s.weight for s in sl])
    al = np.array([s.alpha for s in sl])
    b = np.array([s.width for s in sl])
    c = np.array([s.cohesion for s in sl])
    tp = np.array([math.tan(math.radians(s.phi)) for s in sl])
    u = np.array([s.pore_pressure for s in sl])
    drive = np.sum(W * np.sin(al))
    if drive <= 0:
        return BishopResult(float("nan"), sl, float("nan"), False)
    F, converged = 1.0, False
    for _ in range(max_iter):
        ma = np.cos(al) * (1 + np.tan(al) * tp / F)
        Fn = np.sum((c * b + (W - u * b) * tp) / ma) / drive
        if abs(Fn - F) < tol:
            F, converged = Fn, True
            break
        F = Fn
    min_ma = float(np.min(np.cos(al) * (1 + np.tan(al) * tp / F)))
    if apply_m_alpha_check and min_ma < M_ALPHA_MIN:
        return BishopResult(float("nan"), sl, min_ma, converged)
    return BishopResult(float(F), sl, min_ma, converged)
