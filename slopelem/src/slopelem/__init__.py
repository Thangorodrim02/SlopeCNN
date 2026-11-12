"""slopelem - simplified Bishop limit-equilibrium analysis of Slide2 slope models.

The package reads Rocscience Slide2 model files (``.slim``/``.sli``), computes the
simplified Bishop factor of safety (FoS) of circular slip surfaces with
anisotropic (bedding-dependent) Mohr-Coulomb strength and Hu pore pressures, and
searches for the critical (minimum-FoS) circle.

Typical use::

    from slopelem import SlopeModel, find_critical_circle

    model = SlopeModel.from_file("Feed7.slim")
    result = find_critical_circle(model)
    print(result.fos, result.circle)
"""

from .model import GAMMA_W, Material, SlopeModel
from .bishop import Circle, SliceData, BishopResult, bishop_fos, is_admissible
from .search import (
    SearchSettings,
    SearchResult,
    CriticalCircle,
    circle_through_ground_points,
    global_search,
    small_circle_search,
    find_critical_circle,
)
from .slide2_results import Slide2Result, read_slide2_result

__version__ = "1.0.0"

__all__ = [
    "GAMMA_W",
    "Material",
    "SlopeModel",
    "Circle",
    "SliceData",
    "BishopResult",
    "bishop_fos",
    "is_admissible",
    "SearchSettings",
    "SearchResult",
    "CriticalCircle",
    "circle_through_ground_points",
    "global_search",
    "small_circle_search",
    "find_critical_circle",
    "Slide2Result",
    "read_slide2_result",
    "__version__",
]
