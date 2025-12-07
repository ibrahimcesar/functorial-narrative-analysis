"""
Observation Functors for Functorial Narrative Analysis.

Functors map narrative states to measurable trajectories:
- F_sentiment: Emotional valence (happiness-sadness axis)
- F_arousal: Activation/tension (calm-excited axis)
- F_epistemic: Information/surprise (certainty-uncertainty axis)
- F_thematic: Conceptual distance in semantic space
- F_entropy: Shannon information measures
"""

from .sentiment import SentimentFunctor
from .arousal import ArousalFunctor, JapaneseArousalFunctor
from .entropy import EntropyFunctor, JapaneseEntropyFunctor
from .base import BaseFunctor

__all__ = [
    "BaseFunctor",
    "SentimentFunctor",
    "ArousalFunctor",
    "JapaneseArousalFunctor",
    "EntropyFunctor",
    "JapaneseEntropyFunctor",
]
