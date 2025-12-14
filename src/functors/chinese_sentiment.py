"""
F_sentiment_zh: Chinese Sentiment Functor

Maps classical Chinese narrative states to emotional valence scores.
Uses a dictionary-based approach with classical Chinese sentiment lexicons.

For classical Chinese texts, sentiment analysis is particularly challenging due to:
- Highly contextual meaning (单字多义)
- Literary vs literal meanings
- Historical evolution of character semantics
- Dense philosophical/poetic language

This functor uses:
1. A curated lexicon of positive/negative characters from classical texts
2. Negation handling for classical Chinese patterns
3. Intensity modifiers from literary Chinese

Reference lexicons:
- NTUSD (National Taiwan University Sentiment Dictionary)
- DUTIR (Dalian University of Technology Information Retrieval lab)
- Custom classical Chinese extensions
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass

import numpy as np
from .base import BaseFunctor, Trajectory


@dataclass
class ChineseSentimentResult:
    """Result of Chinese sentiment analysis."""
    positive_score: float  # Positive word count weighted
    negative_score: float  # Negative word count weighted
    compound: float  # Normalized compound score (-1 to 1)
    char_matches: int  # Number of sentiment characters found


# Classical Chinese positive sentiment characters
# Organized by semantic category
POSITIVE_CHARS = {
    # Joy/Happiness (喜)
    '喜': 2, '樂': 2, '悅': 2, '歡': 2, '怡': 2, '愉': 2, '嘉': 2,
    '慶': 2, '欣': 2, '逸': 1, '暢': 1, '適': 1, '娛': 1,

    # Virtue/Goodness (德)
    '善': 2, '德': 2, '仁': 3, '義': 2, '禮': 2, '智': 2, '信': 2,
    '忠': 2, '孝': 2, '和': 2, '正': 1, '直': 1, '誠': 2,

    # Beauty/Elegance (美)
    '美': 2, '麗': 2, '華': 1, '雅': 2, '秀': 1, '妙': 2, '絕': 1,
    '佳': 2, '好': 1, '優': 2, '勝': 1, '異': 1,

    # Power/Achievement (功)
    '功': 2, '成': 1, '達': 1, '興': 2, '盛': 2, '昌': 2, '隆': 1,
    '榮': 2, '貴': 1, '富': 1, '強': 1, '旺': 1,

    # Peace/Harmony (安)
    '安': 2, '平': 1, '泰': 2, '寧': 2, '靜': 1, '康': 2, '順': 1,
    '治': 1, '定': 1, '穩': 1,

    # Love/Affection (愛)
    '愛': 2, '親': 2, '慈': 2, '惠': 2, '恩': 2, '澤': 1,

    # Wisdom/Understanding (明)
    '明': 2, '賢': 2, '聖': 3, '哲': 2, '睿': 2, '達': 1,

    # Fortune/Blessing (福)
    '福': 3, '祥': 2, '瑞': 2, '吉': 2, '慶': 2, '壽': 2,
}

# Classical Chinese negative sentiment characters
NEGATIVE_CHARS = {
    # Sorrow/Grief (悲)
    '悲': -2, '哀': -2, '慟': -3, '傷': -2, '痛': -2, '苦': -2,
    '愁': -2, '憂': -2, '嘆': -1, '泣': -2, '哭': -2, '涕': -1,

    # Anger/Hatred (怒)
    '怒': -2, '憤': -2, '恨': -2, '惡': -2, '嫉': -2, '妒': -2,
    '忿': -2, '怨': -2,

    # Fear/Anxiety (懼)
    '懼': -2, '怕': -1, '畏': -2, '驚': -1, '恐': -2, '慄': -2,
    '惶': -2, '慌': -1,

    # Evil/Wickedness (惡)
    '惡': -2, '邪': -2, '奸': -2, '詐': -2, '偽': -2, '暴': -2,
    '兇': -2, '殘': -2, '虐': -3, '賊': -2,

    # Misfortune/Disaster (禍)
    '禍': -2, '災': -2, '難': -2, '殃': -2, '厄': -2, '凶': -2,
    '亡': -2, '敗': -2, '衰': -2, '弱': -1,

    # Death/Destruction (死)
    '死': -2, '亡': -2, '殺': -2, '滅': -2, '喪': -2, '崩': -2,
    '墜': -2, '毀': -2, '破': -1,

    # Poverty/Lack (貧)
    '貧': -1, '窮': -2, '乏': -1, '缺': -1, '困': -2, '飢': -2,
    '餓': -2,

    # Chaos/Disorder (亂)
    '亂': -2, '危': -2, '險': -1, '患': -2, '害': -2, '疾': -1,
    '病': -1,

    # Shame/Disgrace (辱)
    '辱': -2, '恥': -2, '羞': -1, '污': -1, '賤': -2,
}

# Classical Chinese negation patterns
NEGATION_CHARS = {
    '不', '非', '無', '未', '勿', '莫', '弗', '毋', '否', '靡',
    '罔', '亡', '匪', '蔑',
}

# Intensity modifiers (boosters)
INTENSIFIERS = {
    # Extreme
    '甚': 1.5, '極': 1.5, '至': 1.5, '最': 1.5, '殊': 1.3,
    '尤': 1.3, '益': 1.2, '愈': 1.2, '彌': 1.2, '大': 1.3,

    # Diminishers
    '少': 0.7, '微': 0.7, '稍': 0.8, '略': 0.8, '小': 0.8,
    '些': 0.7, '薄': 0.7,
}


class ChineseSentimentAnalyzer:
    """
    Dictionary-based classical Chinese sentiment analyzer.

    Uses character-level analysis appropriate for classical Chinese,
    which is predominantly monosyllabic and context-dependent.
    """

    def __init__(self):
        """Initialize with default lexicons."""
        self.positive_dict = POSITIVE_CHARS.copy()
        self.negative_dict = NEGATIVE_CHARS.copy()
        self.negation_chars = NEGATION_CHARS.copy()
        self.intensifiers = INTENSIFIERS.copy()

    def _analyze_window(self, text: str) -> ChineseSentimentResult:
        """
        Analyze sentiment of a text window.

        Args:
            text: Chinese text (classical or modern)

        Returns:
            ChineseSentimentResult with scores
        """
        # Remove whitespace and punctuation
        text = re.sub(r'[\s\u3000\uff0c\u3002\uff1f\uff01\uff1a\uff1b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015]', '', text)

        positive_sum = 0.0
        negative_sum = 0.0
        char_matches = 0

        i = 0
        while i < len(text):
            char = text[i]
            modifier = 1.0

            # Check for negation in preceding 3 characters
            negated = False
            for j in range(max(0, i-3), i):
                if text[j] in self.negation_chars:
                    negated = True
                    break

            # Check for intensifier in preceding 2 characters
            for j in range(max(0, i-2), i):
                if text[j] in self.intensifiers:
                    modifier = self.intensifiers[text[j]]
                    break

            # Check sentiment
            if char in self.positive_dict:
                score = self.positive_dict[char] * modifier
                if negated:
                    negative_sum += score * 0.7  # Negated positive becomes weaker negative
                else:
                    positive_sum += score
                char_matches += 1

            elif char in self.negative_dict:
                score = abs(self.negative_dict[char]) * modifier
                if negated:
                    positive_sum += score * 0.5  # Negated negative becomes weaker positive
                else:
                    negative_sum += score
                char_matches += 1

            i += 1

        # Normalize to text length (per 100 characters)
        text_len = max(len(text), 1)
        norm_factor = 100.0 / text_len

        pos_normalized = positive_sum * norm_factor
        neg_normalized = negative_sum * norm_factor

        # Compute compound score in [-1, 1]
        total = pos_normalized + neg_normalized
        if total == 0:
            compound = 0.0
        else:
            compound = (pos_normalized - neg_normalized) / (pos_normalized + neg_normalized + 10)

        return ChineseSentimentResult(
            positive_score=pos_normalized,
            negative_score=neg_normalized,
            compound=float(np.clip(compound, -1, 1)),
            char_matches=char_matches
        )

    def analyze(self, text: str) -> ChineseSentimentResult:
        """Analyze sentiment of text."""
        return self._analyze_window(text)


class ChineseSentimentFunctor(BaseFunctor):
    """
    Chinese sentiment functor for classical and modern Chinese text.

    Maps text windows to sentiment scores in [-1, 1] range where:
    - -1 = strongly negative
    - 0 = neutral
    - +1 = strongly positive
    """

    name = "sentiment_zh"

    def __init__(self, use_bert: bool = False):
        """
        Initialize Chinese sentiment functor.

        Args:
            use_bert: Whether to use BERT for modern Chinese (experimental)
        """
        self.analyzer = ChineseSentimentAnalyzer()
        self.use_bert = use_bert
        self.bert_pipeline = None

        if use_bert:
            self._init_bert()

    def _init_bert(self):
        """Initialize Chinese BERT sentiment pipeline."""
        try:
            from transformers import pipeline
            # Use Chinese sentiment model
            self.bert_pipeline = pipeline(
                "sentiment-analysis",
                model="uer/roberta-base-finetuned-chinanews-chinese",
                device=-1
            )
        except Exception as e:
            print(f"Failed to load Chinese BERT: {e}")
            self.use_bert = False

    def _score_window(self, text: str) -> float:
        """
        Compute sentiment score for a text window.

        Args:
            text: Chinese text

        Returns:
            Sentiment score in [-1, 1]
        """
        result = self.analyzer.analyze(text)
        return result.compound

    def __call__(self, windows: List[str]) -> Trajectory:
        """
        Apply sentiment functor to text windows.

        Args:
            windows: List of Chinese text windows

        Returns:
            Trajectory with sentiment scores
        """
        scores = []
        total_matches = 0

        for window in windows:
            result = self.analyzer.analyze(window)
            scores.append(result.compound)
            total_matches += result.char_matches

        values = np.array(scores)
        time_points = np.linspace(0, 1, len(values))

        return Trajectory(
            values=values,
            time_points=time_points,
            functor_name=self.name,
            metadata={
                "language": "zh",
                "method": "dictionary",
                "n_windows": len(windows),
                "total_sentiment_chars": total_matches,
                "mean_sentiment": float(np.mean(values)),
                "sentiment_variance": float(np.var(values)),
            }
        )


class ClassicalChineseSentimentFunctor(ChineseSentimentFunctor):
    """
    Sentiment functor optimized for classical Chinese (文言文).

    Uses character-level analysis which is more appropriate for
    classical Chinese texts that are predominantly monosyllabic.
    """

    name = "sentiment_classical_zh"

    def __init__(self):
        """Initialize with classical Chinese lexicon."""
        super().__init__(use_bert=False)  # BERT not suited for classical Chinese


def create_windows_chinese(text: str, window_size: int = 500, overlap: int = 250) -> List[str]:
    """
    Create overlapping windows from Chinese text.

    Args:
        text: Chinese text
        window_size: Window size in characters
        overlap: Overlap in characters

    Returns:
        List of text windows
    """
    # Remove whitespace for classical Chinese
    text_clean = re.sub(r'\s+', '', text)
    step = window_size - overlap
    windows = []

    for i in range(0, len(text_clean), step):
        window = text_clean[i:i + window_size]
        if len(window) >= window_size // 2:
            windows.append(window)

    return windows if windows else [text]


if __name__ == "__main__":
    # Test with sample classical Chinese
    sample = """
    子曰：「學而時習之，不亦說乎？有朋自遠方來，不亦樂乎？人不知而不慍，不亦君子乎？」
    子曰：「巧言令色，鮮矣仁！」
    曾子曰：「吾日三省吾身：為人謀而不忠乎？與朋友交而不信乎？傳不習乎？」
    """

    analyzer = ChineseSentimentAnalyzer()
    result = analyzer.analyze(sample)

    print(f"Positive score: {result.positive_score:.2f}")
    print(f"Negative score: {result.negative_score:.2f}")
    print(f"Compound: {result.compound:.3f}")
    print(f"Character matches: {result.char_matches}")
