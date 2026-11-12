"""Search for the critical (minimum-FoS) circular slip surface.

A circle is parameterised by its two ground intersections and the half-angle it
subtends::

    p = (x_entry, x_exit, beta),  R = chord / (2 sin beta)

so every trial circle passes exactly through the ground at both ends. Two
complementary searches are used:

* ``global_search``: random entry/exit pairs anywhere within the Slide2 slope
  limits, followed by Nelder-Mead refinement of the best candidates;
* ``small_circle_search``: random chords of 15-200 m, to catch shallow circles near
  the crest that a global search samples rarely.

``find_critical_circle`` runs both with fixed seeds (the protocol used for the
SlopeCNN labels) and returns the lowest FoS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .bishop import Circle, bishop_fos, is_admissible
from .model import SlopeModel

PENALTY = 1e6
_MIN_CHORD = 10.0          # m
_BETA_RANGE = (0.05, 1.55)  # rad, admissible half-angle


def circle_through_ground_points(model: SlopeModel, x_entry: float, x_exit: float, beta: float) -> Circle:
    """Circle through the ground at ``x_entry`` and ``x_exit`` with half-angle ``beta`` (rad).

    The centre lies on the perpendicular bisector of the chord, on the upper side.
    """
    yl, yr = model.ground(x_entry), model.ground(x_exit)
    dx, dy = x_exit - x_entry, yr - yl
    chord = math.hypot(dx, dy)
    R = chord / (2 * math.sin(beta))
    h = R * math.cos(beta)
    nx, ny = -dy / chord, dx / chord
    if ny < 0:
        nx, ny = -nx, -ny
    return Circle((x_entry + x_exit) / 2 + nx * h, (yl + yr) / 2 + ny * h, R, x_entry, x_exit)


def objective(model: SlopeModel, p: Sequence[float], n_slices: int) -> float:
    """FoS of the circle ``p = (x_entry, x_exit, beta)``; ``PENALTY`` if inadmissible."""
    xl, xr, beta = p
    lo, hi = model.slope_limits
    if not (lo <= xl < xr <= hi) or xr - xl < _MIN_CHORD or not (_BETA_RANGE[0] < beta < _BETA_RANGE[1]):
        return PENALTY
    circle = circle_through_ground_points(model, xl, xr, beta)
    if not is_admissible(model, circle):
        return PENALTY
    F = bishop_fos(model, circle, n_slices=n_slices, sub_columns=2 if n_slices <= 60 else 3).fos
    return F if np.isfinite(F) and F > 0 else PENALTY


@dataclass(frozen=True)
class SearchResult:
    fos: float
    circle: Circle
    method: str
    seed: int


def _refine(model, candidates, n_refine, step_x, step_beta):
    best_f, best_p = PENALTY, None
    for _, p0 in candidates[:n_refine]:
        simplex = [p0, (p0[0] + step_x, p0[1], p0[2]), (p0[0], p0[1] + step_x, p0[2]), (p0[0], p0[1], p0[2] + step_beta)]
        r = minimize(lambda p: objective(model, p, 100), p0, method="Nelder-Mead",
                     options=dict(xatol=0.05, fatol=1e-5, maxiter=400, initial_simplex=simplex))
        if r.fun < best_f:
            best_f, best_p = r.fun, r.x
    return best_p


def _final(model, p, method, seed) -> SearchResult:
    circle = circle_through_ground_points(model, *p)
    return SearchResult(bishop_fos(model, circle, n_slices=200, sub_columns=3).fos, circle, method, seed)


def global_search(model: SlopeModel, n_random: int = 4000, n_refine: int = 12, seed: int = 1) -> SearchResult:
    """Random entry/exit points within the slope limits, then Nelder-Mead refinement."""
    rng = np.random.default_rng(seed)
    lo, hi = model.slope_limits
    cand: List[Tuple[float, tuple]] = []
    for _ in range(n_random):
        a, b = np.sort(rng.uniform(lo, hi, 2))
        beta = rng.uniform(0.1, 1.2)
        f = objective(model, (a, b, beta), 40)
        if f < PENALTY / 10:
            cand.append((f, (a, b, beta)))
    cand.sort(key=lambda t: t[0])
    return _final(model, _refine(model, cand, n_refine, 15.0, 0.08), "global", seed)


def small_circle_search(model: SlopeModel, n_random: int = 2000, n_refine: int = 6, seed: int = 11,
                        chord_range: Tuple[float, float] = (15.0, 200.0)) -> SearchResult:
    """Random short chords (default 15-200 m) anywhere within the slope limits."""
    rng = np.random.default_rng(seed)
    lo, hi = model.slope_limits
    cand: List[Tuple[float, tuple]] = []
    for _ in range(n_random):
        ch = rng.uniform(*chord_range)
        a = rng.uniform(lo, hi - ch)
        b = a + ch
        beta = rng.uniform(0.15, 1.3)
        f = objective(model, (a, b, beta), 40)
        if f < PENALTY / 10:
            cand.append((f, (a, b, beta)))
    cand.sort(key=lambda t: t[0])
    return _final(model, _refine(model, cand, n_refine, 5.0, 0.05), "small-circle", seed)


@dataclass(frozen=True)
class SearchSettings:
    """Search protocol (defaults = the protocol used for the SlopeCNN labels)."""

    global_seeds: Tuple[int, ...] = (1, 2)
    global_n_random: int = 4000
    global_n_refine: int = 12
    extra_seed: int = 3                 # run if the global seeds disagree by > spread_tol
    extra_n_random: int = 6000
    extra_n_refine: int = 20
    spread_tol: float = 0.01
    small_seeds: Tuple[int, ...] = (11, 12)
    small_n_random: int = 2000
    small_n_refine: int = 6


@dataclass(frozen=True)
class CriticalCircle:
    fos: float
    circle: Circle
    found_by: str
    all_runs: List[SearchResult] = field(default_factory=list)


def find_critical_circle(model: SlopeModel, settings: Optional[SearchSettings] = None) -> CriticalCircle:
    """Run the full search protocol and return the minimum-FoS circle."""
    s = settings or SearchSettings()
    runs = [global_search(model, s.global_n_random, s.global_n_refine, seed) for seed in s.global_seeds]
    f = [r.fos for r in runs]
    if len(f) > 1 and abs(min(f) / max(f) - 1) > s.spread_tol:
        runs.append(global_search(model, s.extra_n_random, s.extra_n_refine, s.extra_seed))
    runs += [small_circle_search(model, s.small_n_random, s.small_n_refine, seed) for seed in s.small_seeds]
    valid = [r for r in runs if np.isfinite(r.fos)]
    best = min(valid, key=lambda r: r.fos)
    return CriticalCircle(best.fos, best.circle, best.method, runs)
