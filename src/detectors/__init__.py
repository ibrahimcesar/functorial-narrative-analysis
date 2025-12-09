"""
Narrative structure detectors.

Modules for detecting specific narrative patterns and structures in trajectory data.

Available detectors:
- HarmonCircleDetector: 8-stage Dan Harmon Story Circle (Western TV/Film)
- KishotenketsuDetector: 4-act East Asian narrative structure
- ThreeActDetector: Aristotelian 3-act dramatic structure
- FreytagPyramidDetector: Freytag's 5-act dramatic pyramid
- HerosJourneyDetector: Campbell's 12-stage Monomyth/Hero's Journey
"""

from .harmon_circle import HarmonCircleDetector
from .kishotenketsu import KishotenketsuDetector
from .three_act import ThreeActDetector
from .freytag_pyramid import FreytagPyramidDetector
from .heros_journey import HerosJourneyDetector

__all__ = [
    "HarmonCircleDetector",
    "KishotenketsuDetector",
    "ThreeActDetector",
    "FreytagPyramidDetector",
    "HerosJourneyDetector",
]
