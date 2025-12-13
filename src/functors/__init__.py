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
from .thematic import ThematicFunctor, ThematicCoherenceFunctor, JapaneseThematicFunctor
from .epistemic import EpistemicFunctor, JapaneseEpistemicFunctor, EpistemicPatternDetector
from .japanese_sentiment import JapaneseSentimentFunctor, JapaneseSentimentAnalyzer
from .base import BaseFunctor

__all__ = [
    "BaseFunctor",
    "SentimentFunctor",
    "JapaneseSentimentFunctor",
    "JapaneseSentimentAnalyzer",
    "ArousalFunctor",
    "JapaneseArousalFunctor",
    "EntropyFunctor",
    "JapaneseEntropyFunctor",
    "ThematicFunctor",
    "ThematicCoherenceFunctor",
    "JapaneseThematicFunctor",
    "EpistemicFunctor",
    "JapaneseEpistemicFunctor",
    "EpistemicPatternDetector",
]
