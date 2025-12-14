"""
Category-theoretic structures for narrative analysis.

This module implements Category Narr - the category of narrative states
and transitions - making the abstract categorical framework practical.
"""

from .narr import (
    NarrativeState,
    NarrativeMorphism,
    CategoryNarr,
    NarrativeObject,
)
from .natural_transformations import NaturalTransformation

__all__ = [
    "NarrativeState",
    "NarrativeMorphism",
    "CategoryNarr",
    "NarrativeObject",
    "NaturalTransformation",
]
