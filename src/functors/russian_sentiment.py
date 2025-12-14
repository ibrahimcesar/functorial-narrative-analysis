"""
F_sentiment_ru: Russian Sentiment Functor

Maps Russian narrative states to emotional valence scores.
Supports multiple methods:
1. Transformer-based (multilingual XLM-RoBERTa) - most accurate
2. Dictionary-based with extended lexicon - faster, works offline

For classical Russian literature, sentiment analysis accounts for:
- Rich morphological inflection (word forms vary significantly)
- Diminutives and augmentatives with emotional connotation
- Negation patterns (не, ни, нет)
- Literary vs colloquial registers

Reference lexicons:
- RuSentiLex (Loukachevitch & Levchik)
- LINIS Crowd sentiment lexicon
- Custom classical Russian extensions
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass

import numpy as np
from .base import BaseFunctor, Trajectory


@dataclass
class RussianSentimentResult:
    """Result of Russian sentiment analysis."""
    positive_score: float  # Positive word count weighted
    negative_score: float  # Negative word count weighted
    compound: float  # Normalized compound score (-1 to 1)
    word_matches: int  # Number of sentiment words found


# Russian positive sentiment words (with typical weight)
# Using short stems (3-4 chars) for broader matching in morphologically rich Russian
# Organized by semantic category
RUSSIAN_POSITIVE = {
    # Joy/Happiness (радость) - very common in Tolstoy
    'счаст': 3, 'радо': 2, 'радост': 2, 'весел': 2, 'весёл': 2,
    'удовол': 2, 'наслажд': 2, 'блажен': 2, 'восторг': 2, 'востор': 2,
    'ликов': 2, 'торжест': 1, 'праздн': 1, 'рад': 1, 'счастл': 3,

    # Love/Affection (любовь) - central theme in Anna Karenina
    'любов': 3, 'любл': 3, 'люби': 3, 'любя': 2, 'любим': 3,
    'нежн': 2, 'ласк': 2, 'привяз': 2, 'обожа': 2, 'дорог': 2,
    'мил': 2, 'родн': 2, 'близ': 2, 'любит': 3, 'люб': 2,

    # Beauty/Goodness (красота)
    'красив': 2, 'красот': 2, 'красав': 2, 'прекрас': 2, 'велик': 2,
    'великол': 2, 'чудес': 2, 'замечат': 2, 'превосх': 2, 'изящ': 2,
    'хорош': 1, 'добр': 2, 'благ': 2, 'свет': 1, 'светл': 2,

    # Hope/Faith (надежда)
    'надежд': 2, 'вер': 1, 'упова': 2, 'мечт': 1, 'ожида': 1,
    'уверен': 2, 'оптими': 2, 'верн': 2, 'верит': 2, 'надея': 2,

    # Peace/Calm (покой)
    'покой': 2, 'покоен': 2, 'покоин': 2, 'мир': 2, 'мирн': 2,
    'тиш': 2, 'тишин': 2, 'спокой': 2, 'спокоен': 2, 'безмятеж': 2,
    'уютн': 2, 'комфорт': 1, 'отдых': 1,

    # Success/Achievement (успех)
    'успех': 2, 'победа': 2, 'победи': 2, 'триумф': 2, 'достиж': 2,
    'удач': 2, 'везен': 1, 'слав': 2, 'честь': 2, 'горд': 1,

    # Gratitude/Kindness (благодарность)
    'благодар': 2, 'признат': 2, 'добро': 2, 'щедр': 2,
    'великодуш': 2, 'милосерд': 2, 'сострад': 2, 'благ': 1,

    # Life/Energy (жизнь) - important in Tolstoy
    'жизн': 1, 'жив': 1, 'живо': 1, 'энерг': 1, 'бодр': 1,
    'здоров': 2, 'сил': 1, 'молод': 1, 'силь': 1, 'живой': 1,

    # Intelligence/Wisdom (ум)
    'умн': 2, 'умён': 2, 'мудр': 2, 'разум': 2, 'талант': 2, 'гени': 2,

    # Classical Russian literary positive
    'благосл': 3, 'священ': 2, 'божеств': 2, 'святой': 2, 'свят': 2,
    'ангел': 2, 'рай': 2, 'небес': 1, 'душ': 1, 'душев': 2,

    # Social positive (important for Tolstoy's aristocratic settings)
    'общест': 1, 'бал': 1, 'танц': 1, 'музык': 1, 'песн': 1,
    'смех': 2, 'смея': 2, 'улыб': 2, 'улыба': 2, 'смеш': 1,

    # Nature positive
    'солнц': 1, 'солнеч': 1, 'весн': 1, 'цвет': 1, 'зелен': 1,
}

# Russian negative sentiment words
# Using short stems for broader matching
RUSSIAN_NEGATIVE = {
    # Sorrow/Grief (печаль) - key for Anna Karenina's emotional arc
    'печал': -2, 'печаль': -2, 'горе': -2, 'горь': -2, 'грусть': -2,
    'грустн': -2, 'груст': -2, 'тоск': -2, 'тоска': -2,
    'скорб': -3, 'уныни': -2, 'уныл': -2, 'меланхол': -2, 'страдан': -2,
    'страда': -2, 'мучен': -2, 'муч': -2, 'боль': -2, 'слез': -2,
    'слёз': -2, 'плач': -2, 'плака': -2, 'рыдан': -2, 'рыда': -2,

    # Fear/Anxiety (страх)
    'страх': -2, 'страшн': -2, 'страш': -2, 'ужас': -3, 'кошмар': -3,
    'испуг': -2, 'боязн': -2, 'боя': -1, 'трепет': -1, 'тревог': -2,
    'беспокой': -2, 'паник': -2, 'волнен': -1, 'волнов': -1,

    # Anger/Hatred (гнев)
    'гнев': -2, 'злоб': -2, 'злост': -2, 'злой': -2, 'зол': -2,
    'ярост': -3, 'ненавист': -3, 'ненави': -3, 'ненавид': -3,
    'вражд': -2, 'жесток': -3, 'жестк': -2, 'свиреп': -2,
    'раздраж': -2, 'бешен': -2, 'сердит': -2,

    # Shame/Guilt (стыд) - important for Anna's arc
    'стыд': -2, 'стыдн': -2, 'позор': -3, 'вин': -2, 'виновн': -2,
    'раскаян': -2, 'сожален': -1, 'смущен': -1, 'унижен': -3,
    'униж': -2, 'совест': -1,

    # Death/Destruction (смерть) - crucial for ending
    'смерт': -2, 'смерть': -2, 'умир': -2, 'умер': -2, 'мертв': -2,
    'мёртв': -2, 'гибел': -3, 'погиб': -3, 'убий': -3, 'убив': -3,
    'убит': -3, 'казн': -3, 'могил': -2, 'похорон': -2, 'траур': -2,
    'уничтож': -2, 'самоубий': -3, 'конец': -1, 'конч': -2,

    # Evil/Sin (зло)
    'зло': -2, 'грех': -2, 'греш': -2, 'порок': -2, 'пороч': -2,
    'преступ': -3, 'подл': -2, 'обман': -2, 'обманыв': -2,
    'предат': -3, 'измен': -2, 'ложь': -2, 'лжив': -2, 'лга': -2,

    # Suffering/Misfortune (несчастье)
    'несчаст': -3, 'несчастл': -3, 'беда': -2, 'бед': -1, 'горьк': -2,
    'бедств': -2, 'катастроф': -3, 'трагед': -3, 'крах': -2, 'разорен': -2,

    # Loneliness/Despair (одиночество)
    'одинок': -2, 'одиноч': -2, 'отчаян': -3, 'отчая': -3, 'безнадеж': -3,
    'безысход': -3, 'разлук': -2, 'разрыв': -2, 'пуст': -1, 'пустот': -2,

    # Illness/Weakness (болезнь)
    'болезн': -2, 'болен': -2, 'больн': -2, 'боле': -2, 'немощ': -2,
    'слаб': -1, 'слабост': -2, 'бессил': -2, 'измучен': -2, 'устал': -1,

    # War/Violence (война) - central to War and Peace
    'войн': -2, 'война': -2, 'битв': -1, 'сражен': -1, 'насил': -3,
    'кров': -2, 'кровь': -2, 'ран': -2, 'раненн': -2, 'ранен': -2,
    'пыт': -3, 'пытк': -3, 'мучит': -2, 'враг': -2,

    # Classical Russian literary negative
    'прокля': -3, 'проклят': -3, 'демон': -2, 'дьявол': -3, 'ад': -3,
    'черт': -2, 'чёрт': -2, 'бес': -2, 'погибел': -3, 'окаян': -2,

    # Additional emotional negatives for Tolstoy
    'ревнов': -2, 'ревност': -2, 'ревнив': -2, 'мрачн': -2, 'мрак': -2,
    'темн': -1, 'тёмн': -1, 'хол': -1, 'холод': -1, 'ненавижу': -3,
}

# Russian negation patterns
RUSSIAN_NEGATION = {
    'не', 'ни', 'нет', 'никак', 'никогда', 'нигде', 'никто', 'ничто',
    'никакой', 'ничей', 'нисколько', 'ниоткуда', 'некуда', 'незачем',
    'нельзя', 'без', 'бес', 'ни-',
}

# Intensity modifiers (boosters)
RUSSIAN_INTENSIFIERS = {
    # Extreme
    'очень': 1.5, 'весьма': 1.4, 'крайне': 1.5, 'чрезвычайно': 1.6,
    'необычайно': 1.5, 'совершенно': 1.4, 'абсолютно': 1.5,
    'совсем': 1.3, 'вполне': 1.2, 'слишком': 1.3, 'столь': 1.3,
    'так': 1.2, 'такой': 1.2, 'сколь': 1.3, 'более': 1.2,

    # Diminishers
    'немного': 0.7, 'слегка': 0.7, 'чуть': 0.6, 'едва': 0.6,
    'почти': 0.8, 'несколько': 0.8, 'отчасти': 0.7, 'мало': 0.6,
}


def normalize_russian(text: str) -> str:
    """
    Normalize Russian text for analysis.

    - Converts to lowercase
    - Removes ё -> е (common normalization)
    - Removes punctuation except hyphens in words
    """
    text = text.lower()
    text = text.replace('ё', 'е')
    # Keep only Cyrillic letters, spaces, and hyphens
    text = re.sub(r'[^\u0400-\u04ff\s\-]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def simple_russian_stem(word: str) -> str:
    """
    Simple suffix removal for Russian words.
    Not a full stemmer, but handles common inflections.
    """
    # Common noun/adjective endings to remove
    suffixes = [
        'ость', 'ение', 'ание', 'ства', 'ство',
        'ного', 'ному', 'ными', 'ного', 'ная', 'ное', 'ный', 'ные',
        'ого', 'ому', 'ыми', 'его', 'ему', 'ими',
        'ая', 'ое', 'ый', 'ые', 'ая', 'яя', 'ее', 'ий', 'ие',
        'ов', 'ев', 'ам', 'ям', 'ах', 'ях', 'ом', 'ем', 'ей',
        'а', 'я', 'о', 'е', 'ы', 'и', 'у', 'ю',
    ]

    for suffix in suffixes:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


class RussianSentimentAnalyzer:
    """
    Dictionary-based Russian sentiment analyzer.

    Uses word-level analysis with simple stemming to handle
    Russian morphological variation.
    """

    def __init__(self, use_stemming: bool = True):
        """Initialize with default lexicons."""
        self.positive_dict = RUSSIAN_POSITIVE.copy()
        self.negative_dict = RUSSIAN_NEGATIVE.copy()
        self.negation_words = RUSSIAN_NEGATION.copy()
        self.intensifiers = RUSSIAN_INTENSIFIERS.copy()
        self.use_stemming = use_stemming

        # Create stemmed versions of lexicons
        if use_stemming:
            self._create_stemmed_lexicons()

    def _create_stemmed_lexicons(self):
        """Create stemmed versions of lexicon entries."""
        # Positive lexicon already uses stems
        # Negative lexicon already uses stems
        pass

    def _find_sentiment(self, word: str) -> Tuple[float, str]:
        """
        Find sentiment score for a word.

        Returns:
            (score, type) where type is 'positive', 'negative', or 'neutral'
        """
        # Check direct match first
        if word in self.positive_dict:
            return (self.positive_dict[word], 'positive')
        if word in self.negative_dict:
            return (self.negative_dict[word], 'negative')

        # Try stem matching
        if self.use_stemming:
            stem = simple_russian_stem(word)
            # Check if word starts with any lexicon entry (prefix match)
            for lexicon_stem, score in self.positive_dict.items():
                if word.startswith(lexicon_stem) or stem.startswith(lexicon_stem):
                    return (score, 'positive')
            for lexicon_stem, score in self.negative_dict.items():
                if word.startswith(lexicon_stem) or stem.startswith(lexicon_stem):
                    return (score, 'negative')

        return (0, 'neutral')

    def _analyze_window(self, text: str) -> RussianSentimentResult:
        """
        Analyze sentiment of a text window.

        Args:
            text: Russian text

        Returns:
            RussianSentimentResult with scores
        """
        # Normalize text
        text = normalize_russian(text)
        words = text.split()

        positive_sum = 0.0
        negative_sum = 0.0
        word_matches = 0

        i = 0
        while i < len(words):
            word = words[i]
            modifier = 1.0

            # Check for negation in preceding 3 words
            negated = False
            for j in range(max(0, i-3), i):
                prev_word = words[j]
                if prev_word in self.negation_words:
                    negated = True
                    break
                # Check for не- prefix in current word
                if word.startswith('не') and len(word) > 3:
                    negated = True
                    word = word[2:]  # Remove не- prefix for matching

            # Check for intensifier in preceding 2 words
            for j in range(max(0, i-2), i):
                prev_word = words[j]
                if prev_word in self.intensifiers:
                    modifier = self.intensifiers[prev_word]
                    break

            # Find sentiment
            score, sent_type = self._find_sentiment(word)

            if sent_type == 'positive':
                if negated:
                    negative_sum += abs(score) * modifier * 0.7
                else:
                    positive_sum += score * modifier
                word_matches += 1
            elif sent_type == 'negative':
                if negated:
                    positive_sum += abs(score) * modifier * 0.5
                else:
                    negative_sum += abs(score) * modifier
                word_matches += 1

            i += 1

        # Normalize to text length (per 100 words)
        word_count = max(len(words), 1)
        norm_factor = 100.0 / word_count

        pos_normalized = positive_sum * norm_factor
        neg_normalized = negative_sum * norm_factor

        # Compute compound score in [-1, 1]
        total = pos_normalized + neg_normalized
        if total == 0:
            compound = 0.0
        else:
            compound = (pos_normalized - neg_normalized) / (pos_normalized + neg_normalized + 15)

        return RussianSentimentResult(
            positive_score=pos_normalized,
            negative_score=neg_normalized,
            compound=float(np.clip(compound, -1, 1)),
            word_matches=word_matches
        )

    def analyze(self, text: str) -> RussianSentimentResult:
        """Analyze sentiment of text."""
        return self._analyze_window(text)


class TransformerRussianSentimentAnalyzer:
    """
    Transformer-based Russian sentiment analyzer using multilingual models.

    Uses XLM-RoBERTa fine-tuned for sentiment analysis, which handles
    Russian morphology much better than dictionary approaches.
    """

    def __init__(self, model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"):
        """
        Initialize transformer-based analyzer.

        Args:
            model_name: HuggingFace model name. Options:
                - "cardiffnlp/twitter-xlm-roberta-base-sentiment" (default, multilingual)
                - "blanchefort/rubert-base-cased-sentiment" (Russian-specific)
                - "cointegrated/rubert-tiny-sentiment-balanced" (fast Russian)
        """
        self.model_name = model_name
        self.pipeline = None
        self._init_pipeline()

    def _init_pipeline(self):
        """Initialize the sentiment pipeline."""
        try:
            from transformers import pipeline
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)

            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=-1,  # CPU
                truncation=True,
                max_length=512
            )
            print(f"Loaded transformer model: {self.model_name}")
        except ImportError as e:
            print(f"Warning: Missing dependency for transformer model: {e}")
            print("Install with: pip install transformers torch sentencepiece protobuf")
            print("Falling back to dictionary-based analysis")
            self.pipeline = None
        except Exception as e:
            print(f"Warning: Could not load transformer model: {e}")
            print("Falling back to dictionary-based analysis")
            self.pipeline = None

    def analyze(self, text: str) -> RussianSentimentResult:
        """
        Analyze sentiment using transformer model.

        Args:
            text: Russian text

        Returns:
            RussianSentimentResult with scores
        """
        if self.pipeline is None:
            # Fallback to dictionary
            fallback = RussianSentimentAnalyzer()
            return fallback.analyze(text)

        # Truncate text to reasonable size for transformer
        text_truncated = text[:1500]  # ~500 words

        try:
            result = self.pipeline(text_truncated)[0]
            label = result['label'].lower()
            score = result['score']

            # Map model output to compound score
            # XLM-RoBERTa sentiment uses: negative, neutral, positive
            if 'positive' in label or label == 'pos':
                compound = score
                positive_score = score * 100
                negative_score = 0
            elif 'negative' in label or label == 'neg':
                compound = -score
                positive_score = 0
                negative_score = score * 100
            else:  # neutral
                compound = 0.0
                positive_score = (1 - score) * 50
                negative_score = (1 - score) * 50

            return RussianSentimentResult(
                positive_score=positive_score,
                negative_score=negative_score,
                compound=float(np.clip(compound, -1, 1)),
                word_matches=-1  # Not applicable for transformer
            )
        except Exception as e:
            print(f"Transformer inference error: {e}")
            # Fallback
            fallback = RussianSentimentAnalyzer()
            return fallback.analyze(text)


class RussianSentimentFunctor(BaseFunctor):
    """
    Russian sentiment functor for classical and modern Russian text.

    Maps text windows to sentiment scores in [-1, 1] range where:
    - -1 = strongly negative
    - 0 = neutral
    - +1 = strongly positive

    Supports two methods:
    - 'transformer': Uses multilingual XLM-RoBERTa (more accurate, slower)
    - 'dictionary': Uses lexicon-based analysis (faster, works offline)
    """

    name = "sentiment_ru"

    def __init__(self, method: str = "transformer", use_stemming: bool = True):
        """
        Initialize Russian sentiment functor.

        Args:
            method: 'transformer' (default) or 'dictionary'
            use_stemming: Whether to use simple stemming for dictionary matching
        """
        self.method = method
        if method == "transformer":
            self.analyzer = TransformerRussianSentimentAnalyzer()
        else:
            self.analyzer = RussianSentimentAnalyzer(use_stemming=use_stemming)

    def _score_window(self, text: str) -> float:
        """
        Compute sentiment score for a text window.

        Args:
            text: Russian text

        Returns:
            Sentiment score in [-1, 1]
        """
        result = self.analyzer.analyze(text)
        return result.compound

    def __call__(self, windows: List[str]) -> Trajectory:
        """
        Apply sentiment functor to text windows.

        Args:
            windows: List of Russian text windows

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
                "language": "ru",
                "method": self.method,
                "n_windows": len(windows),
                "total_sentiment_words": total_matches if self.method == "dictionary" else -1,
                "mean_sentiment": float(np.mean(values)),
                "sentiment_variance": float(np.var(values)),
            }
        )


class ClassicalRussianSentimentFunctor(RussianSentimentFunctor):
    """
    Sentiment functor optimized for 19th century Russian literature.

    By default uses transformer-based analysis for best results.
    Falls back to dictionary with extended classical vocabulary.
    """

    name = "sentiment_classical_ru"

    def __init__(self, method: str = "transformer"):
        """
        Initialize with classical Russian support.

        Args:
            method: 'transformer' (default, recommended) or 'dictionary'
        """
        super().__init__(method=method, use_stemming=True)

        # If using dictionary, add classical literary vocabulary
        if method == "dictionary" and hasattr(self.analyzer, 'positive_dict'):
            classical_positive = {
                # Nobility/Honor
                'честн': 2, 'благородн': 2, 'достоин': 2, 'высок': 1,
                'возвышенн': 2, 'чист': 2, 'невинн': 2, 'целомудр': 2,

                # Social positive
                'обществ': 1, 'свет': 1, 'бал': 1, 'праздник': 1,

                # Religious
                'бог': 1, 'господ': 1, 'молитв': 1, 'храм': 1, 'церков': 1,

                # Nature (often positive in Russian lit)
                'весн': 1, 'солнц': 1, 'свет': 1, 'заря': 1, 'луг': 1,
            }

            classical_negative = {
                # Social negative
                'раб': -2, 'крепостн': -2, 'помещик': -1, 'барин': 0,

                # Moral decay
                'падени': -2, 'грехопадени': -3, 'разврат': -3, 'распутств': -2,

                # Russian soul concepts
                'тоска': -2, 'хандра': -2, 'сплин': -2,

                # Death/Suffering
                'чахотк': -2, 'лихорадк': -2, 'горячк': -2,
            }

            self.analyzer.positive_dict.update(classical_positive)
            self.analyzer.negative_dict.update(classical_negative)


def create_windows_russian(text: str, window_size: int = 1000, overlap: int = 500) -> List[str]:
    """
    Create overlapping windows from Russian text.

    Args:
        text: Russian text
        window_size: Window size in words
        overlap: Overlap in words

    Returns:
        List of text windows
    """
    # Normalize and split into words
    text_norm = normalize_russian(text)
    words = text_norm.split()

    step = window_size - overlap
    windows = []

    for i in range(0, len(words), step):
        window_words = words[i:i + window_size]
        if len(window_words) >= window_size // 2:
            windows.append(' '.join(window_words))

    return windows if windows else [text]


if __name__ == "__main__":
    # Test with sample Russian text (from Anna Karenina opening)
    sample = """
    Все счастливые семьи похожи друг на друга, каждая несчастливая семья
    несчастлива по-своему. Всё смешалось в доме Облонских. Жена узнала,
    что муж был в связи с бывшею в их доме француженкою-гувернанткой,
    и объявила мужу, что не может жить с ним в одном доме. Положение это
    продолжалось уже третий день и мучительно чувствовалось и самими
    супругами, и всеми членами семьи, и домочадцами.
    """

    analyzer = RussianSentimentAnalyzer()
    result = analyzer.analyze(sample)

    print(f"Positive score: {result.positive_score:.2f}")
    print(f"Negative score: {result.negative_score:.2f}")
    print(f"Compound: {result.compound:.3f}")
    print(f"Word matches: {result.word_matches}")
