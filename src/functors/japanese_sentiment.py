"""
Japanese Sentiment Functor using SentiStrength dictionary.

Implements a dictionary-based sentiment analysis for Japanese text using
the SentiStrength lexicon from Hsu et al. (2018).

Reference:
    Hsu, T. W., Chen, X., & Cheng, X. (2018).
    Japanese SentiStrength: A tool for sentiment analysis in Japanese.
    https://github.com/tiffanywhsu/japanese-sentistrength
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from .base import BaseFunctor


@dataclass
class JapaneseSentimentResult:
    """Result of Japanese sentiment analysis."""
    positive_score: float  # Max positive intensity (1-5)
    negative_score: float  # Max negative intensity (-1 to -5)
    compound: float  # Normalized compound score (-1 to 1)
    word_matches: int  # Number of sentiment words found


class JapaneseSentimentAnalyzer:
    """
    Dictionary-based Japanese sentiment analyzer.

    Uses the SentiStrength lexicon with simple tokenization.
    Scores range from -5 (very negative) to +5 (very positive).
    """

    def __init__(self, dict_path: Optional[Path] = None):
        """
        Initialize analyzer with sentiment dictionary.

        Args:
            dict_path: Path to SentimentLookupTable.txt
        """
        self.sentiment_dict: Dict[str, int] = {}
        self.negation_words: List[str] = []
        self.booster_words: Dict[str, int] = {}

        # Default path relative to project root
        if dict_path is None:
            dict_path = Path(__file__).parent.parent.parent / "vendor/japanese-sentistrength/SentiStrength_DataJapanese"

        self._load_dictionaries(dict_path)

    def _load_dictionaries(self, base_path: Path):
        """Load sentiment dictionaries."""
        base_path = Path(base_path)

        # Load main sentiment lookup table
        sentiment_file = base_path / "SentimentLookupTable.txt"
        if sentiment_file.exists():
            with open(sentiment_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            term = parts[0]
                            try:
                                score = int(parts[1])
                                self.sentiment_dict[term] = score
                            except ValueError:
                                continue

        # Load negation words
        negation_file = base_path / "NegatingWordList.txt"
        if negation_file.exists():
            with open(negation_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().split('\t')[0]
                    if word:
                        self.negation_words.append(word)

        # Load booster words
        booster_file = base_path / "BoosterWordList.txt"
        if booster_file.exists():
            with open(booster_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            try:
                                self.booster_words[parts[0]] = int(parts[1])
                            except ValueError:
                                continue

    def _simple_tokenize(self, text: str) -> List[str]:
        """
        Simple character n-gram based tokenization for Japanese.

        Since we don't have MeCab, we use a sliding window approach
        to find dictionary matches.
        """
        # Remove spaces and normalize
        text = re.sub(r'\s+', '', text)
        return list(text)

    def _find_sentiment_words(self, text: str) -> List[Tuple[str, int]]:
        """
        Find sentiment words in text using longest match.

        Returns list of (word, score) tuples.
        """
        text = re.sub(r'\s+', '', text)
        matches = []

        # Sort dictionary by length (longest first) for greedy matching
        sorted_terms = sorted(self.sentiment_dict.keys(), key=len, reverse=True)

        # Track matched positions to avoid double-counting
        matched_positions = set()

        for term in sorted_terms:
            start = 0
            while True:
                pos = text.find(term, start)
                if pos == -1:
                    break

                # Check if position already matched by longer term
                term_positions = set(range(pos, pos + len(term)))
                if not term_positions & matched_positions:
                    matches.append((term, self.sentiment_dict[term]))
                    matched_positions.update(term_positions)

                start = pos + 1

        return matches

    def _check_negation(self, text: str, word_pos: int, window: int = 3) -> bool:
        """Check if negation word appears near the sentiment word."""
        # Look for negation patterns before the word
        for neg_word in self.negation_words:
            # Check in window before word position
            search_start = max(0, word_pos - len(neg_word) - window * 2)
            search_text = text[search_start:word_pos]
            if neg_word in search_text:
                return True
        return False

    def analyze(self, text: str) -> JapaneseSentimentResult:
        """
        Analyze sentiment of Japanese text.

        Args:
            text: Japanese text to analyze

        Returns:
            JapaneseSentimentResult with scores
        """
        if not text or not self.sentiment_dict:
            return JapaneseSentimentResult(
                positive_score=0.0,
                negative_score=0.0,
                compound=0.0,
                word_matches=0
            )

        # Find sentiment words
        matches = self._find_sentiment_words(text)

        if not matches:
            return JapaneseSentimentResult(
                positive_score=0.0,
                negative_score=0.0,
                compound=0.0,
                word_matches=0
            )

        # Calculate scores
        positive_scores = []
        negative_scores = []

        for word, score in matches:
            # Check for negation
            word_pos = text.find(word)
            if self._check_negation(text, word_pos):
                score = -score  # Flip polarity

            if score > 0:
                positive_scores.append(score)
            elif score < 0:
                negative_scores.append(score)

        # Max scores (SentiStrength style)
        max_pos = max(positive_scores) if positive_scores else 1
        max_neg = min(negative_scores) if negative_scores else -1

        # Compound score: normalized combination
        # Scale from [-5, 5] to [-1, 1]
        if positive_scores or negative_scores:
            sum_scores = sum(positive_scores) + sum(negative_scores)
            n_scores = len(positive_scores) + len(negative_scores)
            compound = sum_scores / (n_scores * 5)  # Normalize to [-1, 1]
            compound = max(-1.0, min(1.0, compound))
        else:
            compound = 0.0

        return JapaneseSentimentResult(
            positive_score=float(max_pos),
            negative_score=float(max_neg),
            compound=float(compound),
            word_matches=len(matches)
        )

    def polarity_scores(self, text: str) -> Dict[str, float]:
        """
        VADER-compatible interface.

        Returns dict with 'compound', 'pos', 'neg', 'neu' keys.
        """
        result = self.analyze(text)

        # Normalize positive/negative to [0, 1]
        pos = result.positive_score / 5.0 if result.positive_score > 0 else 0.0
        neg = abs(result.negative_score) / 5.0 if result.negative_score < 0 else 0.0
        neu = 1.0 - (pos + neg)
        neu = max(0.0, neu)

        return {
            'compound': result.compound,
            'pos': pos,
            'neg': neg,
            'neu': neu
        }


class JapaneseSentimentFunctor(BaseFunctor):
    """
    Observation functor F_sentiment for Japanese text.

    Maps Japanese narrative text to sentiment trajectory using
    SentiStrength dictionary-based analysis.
    """

    name = "JapaneseSentiment"

    def __init__(
        self,
        window_chars: int = 3000,
        overlap: int = 1500,
        n_points: int = 100,
    ):
        """
        Initialize Japanese sentiment functor.

        Args:
            window_chars: Characters per window (Japanese has no spaces)
            overlap: Character overlap between windows
            n_points: Number of points in output trajectory
        """
        self.window_chars = window_chars
        self.overlap = overlap
        self.n_points = n_points
        self.analyzer = JapaneseSentimentAnalyzer()

    def _create_char_windows(self, text: str) -> List[str]:
        """Create overlapping character windows for Japanese text."""
        windows = []
        step = self.window_chars - self.overlap

        for i in range(0, len(text), step):
            window = text[i:i + self.window_chars]
            if len(window) >= self.window_chars // 2:
                windows.append(window)

        return windows

    def __call__(self, windows: List[str]) -> 'Trajectory':
        """
        Apply functor to pre-windowed text.

        Args:
            windows: List of text windows

        Returns:
            Trajectory with sentiment scores
        """
        from .base import Trajectory

        if not windows:
            return Trajectory(
                values=np.zeros(self.n_points),
                time_points=np.linspace(0, 1, self.n_points),
                functor_name=self.name
            )

        scores = []
        for window in windows:
            result = self.analyzer.polarity_scores(window)
            scores.append(result['compound'])

        values = np.array(scores)
        time_points = np.linspace(0, 1, len(scores))

        return Trajectory(
            values=values,
            time_points=time_points,
            functor_name=self.name
        )

    def apply(self, text: str) -> np.ndarray:
        """
        Apply functor to Japanese text directly.

        Args:
            text: Japanese narrative text

        Returns:
            Sentiment trajectory array of shape (n_points,)
        """
        windows = self._create_char_windows(text)

        if len(windows) < 2:
            return np.zeros(self.n_points)

        # Calculate sentiment for each window
        scores = []
        for window in windows:
            result = self.analyzer.polarity_scores(window)
            scores.append(result['compound'])

        # Resample to fixed length
        x_orig = np.linspace(0, 1, len(scores))
        x_new = np.linspace(0, 1, self.n_points)
        trajectory = np.interp(x_new, x_orig, scores)

        return trajectory

    @property
    def description(self) -> str:
        return "Japanese sentiment trajectory using SentiStrength dictionary"
