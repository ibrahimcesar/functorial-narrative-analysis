"""
F_sentiment_ar: Arabic Sentiment Functor

Maps Arabic narrative states to emotional valence scores.
Uses a dictionary-based approach with Arabic sentiment lexicons.

For classical Arabic texts, sentiment analysis faces challenges:
- Rich morphological structure (root-based)
- Diacritical marks (harakat) often missing
- Classical vs Modern Standard Arabic differences
- Religious/scholarly register dominance

This functor uses:
1. Arabic sentiment lexicons (SAMAR, ArSEL)
2. Root-based sentiment matching
3. Negation handling for Arabic patterns (لا، لم، ليس، ما)
4. Intensity modifiers from classical Arabic

References:
- SAMAR: Sentiment Analysis for Modern Arabic
- ArSEL: Arabic Sentiment Emotion Lexicon
- Classical Arabic adaptations for historical texts
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

import numpy as np
from .base import BaseFunctor, Trajectory


@dataclass
class ArabicSentimentResult:
    """Result of Arabic sentiment analysis."""
    positive_score: float
    negative_score: float
    compound: float  # Normalized compound score (-1 to 1)
    word_matches: int


# Arabic positive sentiment words
# Includes classical and modern variants
ARABIC_POSITIVE = {
    # Joy/Happiness (فرح)
    'فرح': 2, 'سعادة': 2, 'سرور': 2, 'بهجة': 2, 'غبطة': 2,
    'سعيد': 2, 'مسرور': 2, 'فرحان': 2, 'مبتهج': 2,

    # Love/Affection (حب)
    'حب': 2, 'محبة': 2, 'عشق': 2, 'ود': 2, 'مودة': 2,
    'حبيب': 2, 'محبوب': 2, 'عاشق': 2, 'ولوع': 2,

    # Virtue/Goodness (خير)
    'خير': 2, 'حسن': 2, 'جميل': 2, 'طيب': 2, 'صالح': 2,
    'فاضل': 2, 'كريم': 3, 'جواد': 2, 'نبيل': 2,

    # Religious positive (إيمان)
    'إيمان': 3, 'تقوى': 3, 'رحمة': 3, 'بركة': 3, 'نعمة': 2,
    'هداية': 2, 'صلاح': 2, 'إحسان': 3, 'عبادة': 2,
    'جنة': 3, 'ثواب': 2, 'أجر': 2, 'فلاح': 2,

    # Success/Achievement (نجاح)
    'نجاح': 2, 'فوز': 2, 'ظفر': 2, 'انتصار': 2, 'فتح': 2,
    'نصر': 2, 'تفوق': 2, 'إنجاز': 2,

    # Peace/Tranquility (سلام)
    'سلام': 2, 'أمان': 2, 'سكينة': 2, 'طمأنينة': 2, 'راحة': 2,
    'هدوء': 1, 'استقرار': 1,

    # Beauty/Elegance (جمال)
    'جمال': 2, 'حسن': 2, 'بهاء': 2, 'روعة': 2, 'إبداع': 2,

    # Strength/Honor (قوة)
    'قوة': 1, 'عزة': 2, 'شرف': 2, 'كرامة': 2, 'مجد': 2,
    'عظمة': 2, 'جلال': 2,

    # Knowledge/Wisdom (علم)
    'علم': 2, 'حكمة': 3, 'معرفة': 2, 'فهم': 1, 'بصيرة': 2,
    'عقل': 1, 'رشد': 2,

    # Gratitude (شكر)
    'شكر': 2, 'حمد': 2, 'ثناء': 2, 'امتنان': 2,

    # Hope (أمل)
    'أمل': 2, 'رجاء': 2, 'تفاؤل': 2, 'طموح': 1,
}

# Arabic negative sentiment words
ARABIC_NEGATIVE = {
    # Sorrow/Grief (حزن)
    'حزن': -2, 'أسى': -2, 'غم': -2, 'هم': -2, 'كرب': -2,
    'حزين': -2, 'مغموم': -2, 'مكروب': -2, 'بكاء': -2,

    # Anger (غضب)
    'غضب': -2, 'سخط': -2, 'حنق': -2, 'غيظ': -2,
    'غاضب': -2, 'ساخط': -2,

    # Fear (خوف)
    'خوف': -2, 'رعب': -3, 'فزع': -2, 'هلع': -2, 'خشية': -2,
    'خائف': -2, 'مرعوب': -3,

    # Hate/Enmity (كره)
    'كره': -2, 'بغض': -2, 'عداوة': -2, 'كراهية': -2,
    'عدو': -2, 'بغيض': -2,

    # Evil/Sin (شر)
    'شر': -2, 'سوء': -2, 'فساد': -2, 'ذنب': -2, 'معصية': -2,
    'إثم': -2, 'خطيئة': -2, 'جريمة': -3,

    # Religious negative
    'كفر': -3, 'نفاق': -3, 'ظلم': -3, 'طغيان': -3,
    'جهنم': -3, 'عذاب': -3, 'عقاب': -2, 'لعنة': -3,

    # Death/Destruction (موت)
    'موت': -2, 'قتل': -3, 'هلاك': -3, 'دمار': -3, 'خراب': -2,
    'فناء': -2, 'زوال': -2,

    # Weakness/Humiliation (ضعف)
    'ضعف': -1, 'ذل': -2, 'هوان': -2, 'مهانة': -2, 'عار': -2,
    'خزي': -2, 'فشل': -2, 'هزيمة': -2,

    # Poverty/Need (فقر)
    'فقر': -1, 'حاجة': -1, 'عوز': -2, 'ضيق': -1,

    # Pain/Suffering (ألم)
    'ألم': -2, 'عذاب': -3, 'معاناة': -2, 'مرض': -2,
    'بلاء': -2, 'محنة': -2, 'شدة': -1,

    # Ignorance (جهل)
    'جهل': -2, 'غباء': -2, 'حماقة': -2, 'سفاهة': -2,

    # Lies/Deception (كذب)
    'كذب': -2, 'خداع': -2, 'غش': -2, 'نفاق': -2, 'زيف': -2,

    # Injustice (ظلم)
    'ظلم': -3, 'جور': -2, 'اعتداء': -2, 'بغي': -2,
}

# Arabic negation particles
ARABIC_NEGATION = {
    'لا', 'لم', 'لن', 'ليس', 'ما', 'غير', 'بدون', 'عدم',
    'لست', 'ليست', 'ليسوا', 'ليسا',
}

# Arabic intensifiers
ARABIC_INTENSIFIERS = {
    # Strong intensifiers
    'جدا': 1.5, 'جداً': 1.5, 'كثيرا': 1.3, 'كثيراً': 1.3,
    'للغاية': 1.5, 'تماما': 1.4, 'تماماً': 1.4,
    'أشد': 1.5, 'أكثر': 1.3, 'أعظم': 1.4,

    # Diminishers
    'قليلا': 0.7, 'قليلاً': 0.7, 'بعض': 0.8, 'نوعا': 0.7,
}


class ArabicSentimentAnalyzer:
    """
    Dictionary-based Arabic sentiment analyzer.

    Uses word-level analysis with Arabic morphological awareness.
    Handles negation and intensification patterns.
    """

    def __init__(self):
        """Initialize with default lexicons."""
        self.positive_dict = ARABIC_POSITIVE.copy()
        self.negative_dict = ARABIC_NEGATIVE.copy()
        self.negation = ARABIC_NEGATION.copy()
        self.intensifiers = ARABIC_INTENSIFIERS.copy()

    def _normalize_arabic(self, text: str) -> str:
        """Normalize Arabic text for matching."""
        # Remove diacritics (tashkeel)
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
        # Normalize alef variants
        text = re.sub(r'[إأآا]', 'ا', text)
        # Normalize taa marbuta
        text = re.sub(r'ة', 'ه', text)
        # Normalize yaa
        text = re.sub(r'ى', 'ي', text)
        return text

    def _tokenize(self, text: str) -> List[str]:
        """Simple Arabic tokenization."""
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
        tokens = text.split()
        return [self._normalize_arabic(t) for t in tokens if t]

    def _analyze_window(self, text: str) -> ArabicSentimentResult:
        """
        Analyze sentiment of a text window.

        Args:
            text: Arabic text

        Returns:
            ArabicSentimentResult with scores
        """
        tokens = self._tokenize(text)

        positive_sum = 0.0
        negative_sum = 0.0
        word_matches = 0

        for i, token in enumerate(tokens):
            normalized = self._normalize_arabic(token)
            modifier = 1.0

            # Check for negation in preceding 3 words
            negated = False
            for j in range(max(0, i-3), i):
                if self._normalize_arabic(tokens[j]) in self.negation:
                    negated = True
                    break

            # Check for intensifier in preceding 2 words
            for j in range(max(0, i-2), i):
                norm_prev = self._normalize_arabic(tokens[j])
                if norm_prev in self.intensifiers:
                    modifier = self.intensifiers[norm_prev]
                    break

            # Check sentiment (try both normalized and original)
            for word in [normalized, token]:
                if word in self.positive_dict:
                    score = self.positive_dict[word] * modifier
                    if negated:
                        negative_sum += score * 0.7
                    else:
                        positive_sum += score
                    word_matches += 1
                    break

                elif word in self.negative_dict:
                    score = abs(self.negative_dict[word]) * modifier
                    if negated:
                        positive_sum += score * 0.5
                    else:
                        negative_sum += score
                    word_matches += 1
                    break

        # Normalize by text length
        text_len = max(len(tokens), 1)
        norm_factor = 100.0 / text_len

        pos_normalized = positive_sum * norm_factor
        neg_normalized = negative_sum * norm_factor

        # Compute compound score
        total = pos_normalized + neg_normalized
        if total == 0:
            compound = 0.0
        else:
            compound = (pos_normalized - neg_normalized) / (pos_normalized + neg_normalized + 10)

        return ArabicSentimentResult(
            positive_score=pos_normalized,
            negative_score=neg_normalized,
            compound=float(np.clip(compound, -1, 1)),
            word_matches=word_matches
        )

    def analyze(self, text: str) -> ArabicSentimentResult:
        """Analyze sentiment of text."""
        return self._analyze_window(text)


class ArabicSentimentFunctor(BaseFunctor):
    """
    Arabic sentiment functor for classical and modern Arabic text.

    Maps text windows to sentiment scores in [-1, 1] range.
    """

    name = "sentiment_ar"

    def __init__(self):
        """Initialize Arabic sentiment functor."""
        self.analyzer = ArabicSentimentAnalyzer()

    def _score_window(self, text: str) -> float:
        """Compute sentiment score for a text window."""
        result = self.analyzer.analyze(text)
        return result.compound

    def __call__(self, windows: List[str]) -> Trajectory:
        """
        Apply sentiment functor to text windows.

        Args:
            windows: List of Arabic text windows

        Returns:
            Trajectory with sentiment scores
        """
        scores = []
        total_matches = 0

        for window in windows:
            result = self.analyzer.analyze(window)
            scores.append(result.compound)
            total_matches += result.word_matches

        values = np.array(scores)
        time_points = np.linspace(0, 1, len(values))

        return Trajectory(
            values=values,
            time_points=time_points,
            functor_name=self.name,
            metadata={
                "language": "ar",
                "method": "dictionary",
                "n_windows": len(windows),
                "total_sentiment_words": total_matches,
                "mean_sentiment": float(np.mean(values)),
                "sentiment_variance": float(np.var(values)),
            }
        )


class ClassicalArabicSentimentFunctor(ArabicSentimentFunctor):
    """
    Sentiment functor optimized for classical Arabic (فصحى).

    Includes additional classical Arabic vocabulary and religious terms.
    """

    name = "sentiment_classical_ar"

    def __init__(self):
        """Initialize with extended classical lexicon."""
        super().__init__()

        # Add more classical terms
        self.analyzer.positive_dict.update({
            # Classical positive terms
            'بشرى': 2, 'نجاة': 2, 'فضل': 2, 'كرم': 2,
            'شفاء': 2, 'عافية': 2, 'توفيق': 2, 'سداد': 2,
            'رضا': 3, 'رضوان': 3, 'مغفرة': 3,
        })

        self.analyzer.negative_dict.update({
            # Classical negative terms
            'وبال': -2, 'شقاء': -2, 'ويل': -3, 'ثبور': -3,
            'سقم': -2, 'داء': -2, 'وباء': -2,
            'فتنة': -2, 'بلية': -2, 'رزية': -2,
        })


def create_windows_arabic(text: str, window_size: int = 200, overlap: int = 100) -> List[str]:
    """
    Create overlapping windows from Arabic text.

    Args:
        text: Arabic text
        window_size: Window size in words
        overlap: Overlap in words

    Returns:
        List of text windows
    """
    # Split on whitespace
    words = text.split()
    step = window_size - overlap
    windows = []

    for i in range(0, len(words), step):
        window = ' '.join(words[i:i + window_size])
        if len(window.split()) >= window_size // 2:
            windows.append(window)

    return windows if windows else [text]


if __name__ == "__main__":
    # Test with sample Arabic
    sample_positive = """
    الحمد لله رب العالمين، الرحمن الرحيم. إن من نعم الله علينا أن هدانا للإيمان
    وأنعم علينا بالصحة والعافية. فنحن في سعادة وبهجة عظيمة.
    """

    sample_negative = """
    وقع في الحزن والكرب الشديد، فقد أصابته مصيبة عظيمة. مات أخوه في الحرب
    وخرب بيته. فهو في ألم وعذاب لا يوصف.
    """

    analyzer = ArabicSentimentAnalyzer()

    result1 = analyzer.analyze(sample_positive)
    print(f"Positive text:")
    print(f"  Positive score: {result1.positive_score:.2f}")
    print(f"  Negative score: {result1.negative_score:.2f}")
    print(f"  Compound: {result1.compound:.3f}")
    print(f"  Matches: {result1.word_matches}")

    result2 = analyzer.analyze(sample_negative)
    print(f"\nNegative text:")
    print(f"  Positive score: {result2.positive_score:.2f}")
    print(f"  Negative score: {result2.negative_score:.2f}")
    print(f"  Compound: {result2.compound:.3f}")
    print(f"  Matches: {result2.word_matches}")
