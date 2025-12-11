"""
Narrative structure detectors.

Modules for detecting specific narrative patterns and structures in trajectory data.

Available Detectors:
    - HarmonCircleDetector: Dan Harmon's 8-stage Story Circle
    - KishotenketsuDetector: 4-act East Asian structure (起承転結)
    - AristotleDetector: Classical three-act structure (Poetics)
    - FreytagDetector: Five-act pyramid structure (1863)
    - CampbellDetector: Hero's Journey / Monomyth (12 stages)
"""

from .harmon_circle import HarmonCircleDetector
from .kishotenketsu import KishotenketsuDetector
from .aristotle import AristotleDetector
from .freytag import FreytagDetector
from .campbell import CampbellDetector

__all__ = [
    "HarmonCircleDetector",
    "KishotenketsuDetector",
    "AristotleDetector",
    "FreytagDetector",
    "CampbellDetector",
]
