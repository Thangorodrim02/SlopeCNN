"""slope_cnn - CNN surrogate for slope-stability FoS (Elbistan-Coellolar, Section 4)."""

from .config import Config

__version__ = "2.0.0"

__all__ = ["Config"]

try:
    from .model import SlopeCNN, SoranzoCNN, build_model, count_params
    from .augmentation import augment, AUGMENTATIONS
except ModuleNotFoundError:
    pass  # torch not installed - tensor generation only
