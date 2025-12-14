"""
Observation Functors for Functorial Narrative Analysis.

Functors map narrative states to measurable trajectories:
- F_sentiment: Emotional valence (happiness-sadness axis)
- F_arousal: Activation/tension (calm-excited axis)
- F_epistemic: Information/surprise (certainty-uncertainty axis)
- F_thematic: Conceptual distance in semantic space
- F_entropy: Shannon information measures
- F_pacing: Narrative rhythm (scene length, dialogue density)
- F_character: Character presence and focus
- F_voice: Narrative POV and distance
"""

from .sentiment import SentimentFunctor
from .arousal import ArousalFunctor, JapaneseArousalFunctor
from .entropy import EntropyFunctor, JapaneseEntropyFunctor
from .thematic import ThematicFunctor, ThematicCoherenceFunctor, JapaneseThematicFunctor
from .epistemic import EpistemicFunctor, JapaneseEpistemicFunctor, EpistemicPatternDetector
from .pacing import PacingFunctor, JapanesePacingFunctor
from .character_presence import CharacterPresenceFunctor, JapaneseCharacterPresenceFunctor, CharacterArcAnalyzer
from .narrative_voice import NarrativeVoiceFunctor, JapaneseNarrativeVoiceFunctor, POVShiftDetector
from .japanese_sentiment import JapaneseSentimentFunctor, JapaneseSentimentAnalyzer
from .chinese_sentiment import ChineseSentimentFunctor, ClassicalChineseSentimentFunctor, ChineseSentimentAnalyzer
from .arabic_sentiment import ArabicSentimentFunctor, ClassicalArabicSentimentFunctor, ArabicSentimentAnalyzer
from .russian_sentiment import RussianSentimentFunctor, ClassicalRussianSentimentFunctor, RussianSentimentAnalyzer
from .base import BaseFunctor

__all__ = [
    "BaseFunctor",
    # Sentiment
    "SentimentFunctor",
    "JapaneseSentimentFunctor",
    "JapaneseSentimentAnalyzer",
    # Arousal
    "ArousalFunctor",
    "JapaneseArousalFunctor",
    # Entropy
    "EntropyFunctor",
    "JapaneseEntropyFunctor",
    # Thematic
    "ThematicFunctor",
    "ThematicCoherenceFunctor",
    "JapaneseThematicFunctor",
    # Epistemic
    "EpistemicFunctor",
    "JapaneseEpistemicFunctor",
    "EpistemicPatternDetector",
    # Pacing
    "PacingFunctor",
    "JapanesePacingFunctor",
    # Character Presence
    "CharacterPresenceFunctor",
    "JapaneseCharacterPresenceFunctor",
    "CharacterArcAnalyzer",
    # Narrative Voice
    "NarrativeVoiceFunctor",
    "JapaneseNarrativeVoiceFunctor",
    "POVShiftDetector",
    # Chinese Sentiment
    "ChineseSentimentFunctor",
    "ClassicalChineseSentimentFunctor",
    "ChineseSentimentAnalyzer",
    # Arabic Sentiment
    "ArabicSentimentFunctor",
    "ClassicalArabicSentimentFunctor",
    "ArabicSentimentAnalyzer",
    # Russian Sentiment
    "RussianSentimentFunctor",
    "ClassicalRussianSentimentFunctor",
    "RussianSentimentAnalyzer",
]
