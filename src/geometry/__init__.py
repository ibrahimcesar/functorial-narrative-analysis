"""
Information Geometry for Narrative Analysis.

This module implements information-geometric measures for analyzing
narrative structure, treating stories as trajectories through
statistical manifolds.

Key concepts:
    - Surprisal: -log P(token | context), measures local unexpectedness
    - KL Divergence: Measures belief update cost between narrative states
    - Fisher Information: Sensitivity of belief distribution to parameters
    - Geodesic Curvature: How much a narrative deviates from "straight" paths

Reference:
    Amari, S. (2016). Information Geometry and Its Applications.
    Schmidhuber, J. (2010). Formal Theory of Creativity, Fun, and Intrinsic Motivation.
"""

from .surprisal import SurprisalExtractor, SurprisalTrajectory
from .curvature import NarrativeCurvature, GeometricFeatures
from .divergence import KLDivergenceAnalyzer, BeliefTrajectory

__all__ = [
    "SurprisalExtractor",
    "SurprisalTrajectory",
    "NarrativeCurvature",
    "GeometricFeatures",
    "KLDivergenceAnalyzer",
    "BeliefTrajectory",
]
