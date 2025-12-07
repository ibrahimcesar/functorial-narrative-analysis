"""
F_sentiment for Japanese: Emotional Valence Functor (Japanese)

Maps Japanese narrative states to positions on a happiness-sadness axis.
Uses dictionary-based sentiment analysis optimized for Japanese text.
"""

import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm
import click
from rich.console import Console

from .base import BaseFunctor, Trajectory

console = Console()


# Japanese sentiment lexicon (basic positive/negative words)
# This is a simplified lexicon - for production, use a full dictionary like:
# - Japanese Sentiment Polarity Dictionary
# - ML-Ask emotion dictionary
POSITIVE_WORDS = [
    '嬉しい', '楽しい', '幸せ', '喜び', '愛', '好き', '美しい', '素晴らしい',
    '良い', 'よい', 'いい', '笑', '明るい', '希望', '感謝', '優しい',
    '幸福', '満足', '安心', '穏やか', '快い', '心地よい', '爽やか',
    '輝く', '微笑', 'ほほえみ', '歓喜', '感動', '素敵', '最高',
    'うれしい', 'たのしい', 'しあわせ', 'よろこび', 'すき', 'うつくしい',
]

NEGATIVE_WORDS = [
    '悲しい', '辛い', '苦しい', '怒り', '憎い', '嫌い', '醜い', '恐ろしい',
    '悪い', 'わるい', '泣', '暗い', '絶望', '不安', '寂しい', '孤独',
    '不幸', '不満', '心配', '恐怖', '痛い', '苦痛', '悲惨', '残酷',
    '涙', '嘆き', '憂鬱', '憎悪', '失望', '後悔', '罪', '死',
    'かなしい', 'つらい', 'くるしい', 'いかり', 'にくい', 'きらい',
]

# Intensifiers and negations
INTENSIFIERS = ['とても', '非常に', '大変', 'すごく', '極めて', '実に']
NEGATIONS = ['ない', 'なかった', 'ません', 'ず', 'ぬ', '不']


class JapaneseSentimentFunctor(BaseFunctor):
    """
    Sentiment analysis functor for Japanese text.

    Uses dictionary-based approach with optional transformer model.
    Maps text windows to sentiment scores in [-1, 1] range.
    """

    name = "sentiment_ja"

    def __init__(
        self,
        method: str = "dictionary",
        use_transformer: bool = False,
        model_name: str = "cl-tohoku/bert-base-japanese-v3"
    ):
        """
        Initialize Japanese sentiment functor.

        Args:
            method: "dictionary" or "transformer"
            use_transformer: Whether to use transformer model
            model_name: HuggingFace model for transformer method
        """
        self.method = method
        self.use_transformer = use_transformer
        self.model_name = model_name

        # Build word sets for fast lookup
        self.positive_words = set(POSITIVE_WORDS)
        self.negative_words = set(NEGATIVE_WORDS)
        self.intensifiers = set(INTENSIFIERS)
        self.negations = set(NEGATIONS)

        # Initialize transformer if needed
        self.pipeline = None
        if use_transformer or method == "transformer":
            self._init_transformer()

    def _init_transformer(self):
        """Initialize transformer sentiment pipeline."""
        try:
            from transformers import pipeline, AutoTokenizer
            console.print("[yellow]Loading Japanese sentiment model...[/yellow]")

            # Use a Japanese sentiment model if available
            self.pipeline = pipeline(
                "sentiment-analysis",
                model="daigo/bert-base-japanese-sentiment",
                tokenizer="daigo/bert-base-japanese-sentiment",
                device=-1
            )
            console.print("[green]✓ Japanese sentiment model loaded[/green]")
        except Exception as e:
            console.print(f"[yellow]Transformer not available: {e}. Using dictionary method.[/yellow]")
            self.method = "dictionary"
            self.use_transformer = False

    def _tokenize_japanese(self, text: str) -> List[str]:
        """
        Simple tokenization for Japanese.

        For production, use MeCab or SudachiPy for proper morphological analysis.
        """
        # Remove punctuation and split by character groups
        # This is a simple approach - real tokenization needs morphological analysis
        tokens = []

        # Extract potential words using regex patterns
        # Hiragana sequences
        hiragana = re.findall(r'[ぁ-ん]+', text)
        # Katakana sequences
        katakana = re.findall(r'[ァ-ン]+', text)
        # Kanji sequences (potential words)
        kanji = re.findall(r'[一-龯]+', text)
        # Mixed kanji-hiragana (common word pattern)
        mixed = re.findall(r'[一-龯]+[ぁ-ん]+', text)

        tokens.extend(hiragana)
        tokens.extend(katakana)
        tokens.extend(kanji)
        tokens.extend(mixed)

        return tokens

    def _dictionary_score(self, text: str) -> float:
        """
        Compute sentiment using dictionary lookup.

        Args:
            text: Japanese text

        Returns:
            Sentiment score in [-1, 1]
        """
        # Check for word matches
        positive_count = 0
        negative_count = 0

        # Check each sentiment word in the text
        for word in self.positive_words:
            positive_count += text.count(word)

        for word in self.negative_words:
            negative_count += text.count(word)

        # Check for negations that might flip sentiment
        negation_count = sum(text.count(neg) for neg in self.negations)

        # Simple scoring
        total = positive_count + negative_count
        if total == 0:
            return 0.0

        # Base score
        score = (positive_count - negative_count) / (total + 1)

        # Negation can flip sentiment (simplified)
        if negation_count > 0 and abs(score) > 0.1:
            score *= 0.5  # Dampen rather than flip completely

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, score))

    def _transformer_score(self, text: str) -> float:
        """
        Compute sentiment using transformer model.

        Args:
            text: Japanese text

        Returns:
            Sentiment score in [-1, 1]
        """
        if not self.pipeline:
            return self._dictionary_score(text)

        try:
            # Truncate for transformer
            truncated = text[:512]
            result = self.pipeline(truncated)[0]

            label = result["label"].lower()
            score = result["score"]

            # Map to [-1, 1]
            if "positive" in label or "ポジティブ" in label:
                return score
            elif "negative" in label or "ネガティブ" in label:
                return -score
            else:
                return 0.0

        except Exception:
            return self._dictionary_score(text)

    def _score_window(self, text: str) -> float:
        """
        Compute sentiment for a text window.

        Args:
            text: Window text

        Returns:
            Sentiment score in [-1, 1]
        """
        if self.method == "transformer" and self.pipeline:
            return self._transformer_score(text)
        else:
            return self._dictionary_score(text)

    def __call__(self, windows: List[str]) -> Trajectory:
        """
        Apply Japanese sentiment functor to text windows.

        Args:
            windows: List of text windows

        Returns:
            Trajectory with sentiment scores
        """
        scores = []

        for window in windows:
            score = self._score_window(window)
            scores.append(score)

        values = np.array(scores)
        time_points = np.linspace(0, 1, len(values))

        return Trajectory(
            values=values,
            time_points=time_points,
            functor_name=self.name,
            metadata={
                "method": self.method,
                "language": "ja",
                "n_windows": len(windows),
            }
        )


def create_windows_japanese(
    text: str,
    window_size: int = 500,
    overlap: int = 250
) -> List[str]:
    """
    Create overlapping windows from Japanese text.

    Args:
        text: Japanese text
        window_size: Characters per window
        overlap: Overlap between windows

    Returns:
        List of text windows
    """
    # Clean whitespace but preserve structure
    text = re.sub(r'\s+', '', text)

    step = window_size - overlap
    windows = []

    for i in range(0, len(text), step):
        window = text[i:i + window_size]
        if len(window) >= window_size // 2:
            windows.append(window)

    return windows if windows else [text]


def process_japanese_corpus(
    input_dir: Path,
    output_dir: Path,
    method: str = "dictionary",
    window_size: int = 500,
    overlap: int = 250
):
    """
    Process a corpus of Japanese texts through the sentiment functor.

    Args:
        input_dir: Directory containing JSON files with Japanese text
        output_dir: Output directory for trajectories
        method: Sentiment method
        window_size: Window size in characters
        overlap: Window overlap in characters
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize functor
    functor = JapaneseSentimentFunctor(method=method)

    # Find all text files
    text_files = list(Path(input_dir).glob("*.json"))
    text_files = [f for f in text_files if f.name != "manifest.json"]
    console.print(f"[blue]Processing {len(text_files)} Japanese texts...[/blue]")

    results = []

    for text_file in tqdm(text_files, desc="Extracting sentiment (JA)"):
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            text = data.get("text", "")
            if not text:
                continue

            # Create windows
            windows = create_windows_japanese(text, window_size, overlap)

            # Apply functor
            trajectory = functor(windows)

            # Add metadata
            trajectory.metadata.update({
                "source_id": data.get("id", text_file.stem),
                "title": data.get("title", "Unknown"),
                "author": data.get("author", "Unknown"),
                "char_count": data.get("char_count", len(text)),
                "language": "ja",
            })

            # Save trajectory
            out_file = output_dir / f"{text_file.stem}_sentiment.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory.to_dict(), f, ensure_ascii=False, indent=2)

            results.append({
                "id": data.get("id", text_file.stem),
                "title": data.get("title"),
                "mean_sentiment": float(np.mean(trajectory.values)),
                "std_sentiment": float(np.std(trajectory.values)),
                "trajectory_file": str(out_file.name),
            })

        except Exception as e:
            console.print(f"[red]Error processing {text_file}: {e}[/red]")
            continue

    # Save manifest
    manifest = {
        "functor": "sentiment_ja",
        "method": method,
        "language": "ja",
        "window_size": window_size,
        "overlap": overlap,
        "count": len(results),
        "trajectories": results,
    }

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    console.print(f"[bold green]✓ Processed {len(results)} Japanese texts[/bold green]")
    console.print(f"[green]Saved to {output_dir}[/green]")


@click.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True),
              help='Input directory with Japanese text JSON files')
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(),
              help='Output directory for trajectories')
@click.option('--method', '-m', default='dictionary',
              type=click.Choice(['dictionary', 'transformer']),
              help='Sentiment analysis method')
@click.option('--window-size', '-w', default=500, help='Window size in characters')
@click.option('--overlap', default=250, help='Window overlap in characters')
def main(input_dir: str, output_dir: str, method: str, window_size: int, overlap: int):
    """Extract sentiment trajectories from Japanese text corpus."""
    process_japanese_corpus(
        Path(input_dir),
        Path(output_dir),
        method=method,
        window_size=window_size,
        overlap=overlap
    )


if __name__ == "__main__":
    main()
