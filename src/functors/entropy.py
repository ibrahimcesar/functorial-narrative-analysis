"""
F_entropy: Narrative Complexity/Predictability Functor

Maps narrative states to information-theoretic measures of complexity.
Captures how "surprising" or "predictable" narrative segments are.

High entropy = unpredictable, complex, information-dense
Low entropy = predictable, repetitive, formulaic

This functor is particularly relevant for detecting:
- Plot twists and surprises (entropy spikes)
- Formulaic sections (low entropy)
- Narrative complexity differences across cultures
- The "ten" (twist) in kishōtenketsu structure
"""

import json
import re
import math
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

import numpy as np
from tqdm import tqdm
import click
from rich.console import Console

from .base import BaseFunctor, Trajectory

console = Console()


class EntropyFunctor(BaseFunctor):
    """
    Entropy functor measuring narrative complexity and predictability.

    Uses multiple entropy measures:
    1. Lexical entropy: vocabulary diversity within window
    2. N-gram entropy: predictability of word sequences
    3. Structural entropy: sentence length variation

    Combined into a single complexity score in [0, 1].
    """

    name = "entropy"

    def __init__(
        self,
        method: str = "combined",
        ngram_size: int = 2,
        vocabulary_weight: float = 0.4,
        ngram_weight: float = 0.3,
        structural_weight: float = 0.3,
    ):
        """
        Initialize entropy functor.

        Args:
            method: "lexical", "ngram", "structural", or "combined"
            ngram_size: Size of n-grams for sequence entropy
            vocabulary_weight: Weight for lexical diversity
            ngram_weight: Weight for n-gram predictability
            structural_weight: Weight for structural variation
        """
        self.method = method
        self.ngram_size = ngram_size
        self.vocabulary_weight = vocabulary_weight
        self.ngram_weight = ngram_weight
        self.structural_weight = structural_weight

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        # Lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words

    def _compute_lexical_entropy(self, words: List[str]) -> float:
        """
        Compute lexical entropy (vocabulary diversity).

        Uses Shannon entropy over word frequency distribution.
        Higher = more diverse vocabulary.
        """
        if len(words) == 0:
            return 0.0

        # Word frequency distribution
        freq = Counter(words)
        total = len(words)

        # Shannon entropy
        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by maximum possible entropy (uniform distribution)
        max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1.0

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _compute_ngram_entropy(self, words: List[str]) -> float:
        """
        Compute n-gram entropy (sequence predictability).

        Measures how predictable word sequences are.
        Higher = less predictable, more surprising.
        """
        if len(words) < self.ngram_size:
            return 0.5  # Neutral for short texts

        # Build n-grams
        ngrams = []
        for i in range(len(words) - self.ngram_size + 1):
            ngram = tuple(words[i:i + self.ngram_size])
            ngrams.append(ngram)

        if len(ngrams) == 0:
            return 0.5

        # Frequency distribution
        freq = Counter(ngrams)
        total = len(ngrams)

        # Shannon entropy
        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize
        max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1.0

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _compute_structural_entropy(self, text: str) -> float:
        """
        Compute structural entropy (sentence variation).

        Measures variation in sentence lengths and structures.
        Higher = more varied sentence structure.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.5  # Neutral for single sentence

        # Sentence lengths
        lengths = [len(s.split()) for s in sentences]

        # Coefficient of variation (normalized std dev)
        mean_len = np.mean(lengths)
        if mean_len == 0:
            return 0.5

        std_len = np.std(lengths)
        cv = std_len / mean_len

        # Also consider punctuation diversity
        punct_counts = {
            'question': text.count('?'),
            'exclaim': text.count('!'),
            'period': text.count('.'),
            'comma': text.count(','),
            'semicolon': text.count(';'),
            'colon': text.count(':'),
        }

        total_punct = sum(punct_counts.values())
        if total_punct > 0:
            punct_entropy = 0.0
            for count in punct_counts.values():
                if count > 0:
                    p = count / total_punct
                    punct_entropy -= p * math.log2(p)
            max_punct_entropy = math.log2(len(punct_counts))
            punct_diversity = punct_entropy / max_punct_entropy if max_punct_entropy > 0 else 0
        else:
            punct_diversity = 0.0

        # Combine CV and punctuation diversity
        # CV typically ranges 0-2, normalize to 0-1
        cv_normalized = min(1.0, cv / 1.5)

        return 0.6 * cv_normalized + 0.4 * punct_diversity

    def _score_window(self, text: str) -> float:
        """
        Compute entropy score for a text window.

        Args:
            text: Window text

        Returns:
            Entropy score in [0, 1]
        """
        words = self._tokenize(text)

        if self.method == "lexical":
            return self._compute_lexical_entropy(words)
        elif self.method == "ngram":
            return self._compute_ngram_entropy(words)
        elif self.method == "structural":
            return self._compute_structural_entropy(text)
        else:  # combined
            lexical = self._compute_lexical_entropy(words)
            ngram = self._compute_ngram_entropy(words)
            structural = self._compute_structural_entropy(text)

            combined = (
                self.vocabulary_weight * lexical +
                self.ngram_weight * ngram +
                self.structural_weight * structural
            )
            return max(0.0, min(1.0, combined))

    def __call__(self, windows: List[str]) -> Trajectory:
        """
        Apply entropy functor to text windows.

        Args:
            windows: List of text windows

        Returns:
            Trajectory with entropy scores
        """
        scores = [self._score_window(window) for window in windows]

        values = np.array(scores)
        time_points = np.linspace(0, 1, len(values))

        return Trajectory(
            values=values,
            time_points=time_points,
            functor_name=self.name,
            metadata={
                "method": self.method,
                "ngram_size": self.ngram_size,
                "n_windows": len(windows),
            }
        )


class JapaneseEntropyFunctor(BaseFunctor):
    """
    Entropy functor for Japanese text.

    Japanese-specific considerations:
    - Character-based entropy (kanji, hiragana, katakana mix)
    - No word boundaries, so use character n-grams
    - Particle diversity as structural measure
    """

    name = "entropy_ja"

    def __init__(self, ngram_size: int = 3):
        self.ngram_size = ngram_size

        # Common particles for structural analysis
        self.particles = ['は', 'が', 'を', 'に', 'で', 'と', 'も', 'の', 'へ', 'から', 'まで', 'より']

    def _char_type_entropy(self, text: str) -> float:
        """
        Compute entropy of character type distribution.

        Japanese texts mix kanji, hiragana, katakana.
        More mixing = higher complexity.
        """
        hiragana = len(re.findall(r'[ぁ-ん]', text))
        katakana = len(re.findall(r'[ァ-ン]', text))
        kanji = len(re.findall(r'[一-龯]', text))
        other = len(text) - hiragana - katakana - kanji

        total = hiragana + katakana + kanji + other
        if total == 0:
            return 0.5

        counts = [hiragana, katakana, kanji, other]
        counts = [c for c in counts if c > 0]

        if len(counts) <= 1:
            return 0.0

        entropy = 0.0
        for c in counts:
            p = c / total
            entropy -= p * math.log2(p)

        max_entropy = math.log2(4)  # Four character types
        return entropy / max_entropy

    def _ngram_entropy(self, text: str) -> float:
        """Compute character n-gram entropy."""
        # Remove whitespace
        text = re.sub(r'\s+', '', text)

        if len(text) < self.ngram_size:
            return 0.5

        ngrams = [text[i:i+self.ngram_size] for i in range(len(text) - self.ngram_size + 1)]

        freq = Counter(ngrams)
        total = len(ngrams)

        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _particle_diversity(self, text: str) -> float:
        """Measure diversity of grammatical particles."""
        counts = {p: text.count(p) for p in self.particles}
        total = sum(counts.values())

        if total == 0:
            return 0.5

        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        max_entropy = math.log2(len(self.particles))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _score_window(self, text: str) -> float:
        """Compute entropy for Japanese text window."""
        char_type = self._char_type_entropy(text)
        ngram = self._ngram_entropy(text)
        particle = self._particle_diversity(text)

        # Combined score
        return 0.3 * char_type + 0.4 * ngram + 0.3 * particle

    def __call__(self, windows: List[str]) -> Trajectory:
        scores = [self._score_window(window) for window in windows]

        return Trajectory(
            values=np.array(scores),
            time_points=np.linspace(0, 1, len(scores)),
            functor_name=self.name,
            metadata={"language": "ja", "n_windows": len(windows)}
        )


def process_corpus(
    input_dir: Path,
    output_dir: Path,
    language: str = "en",
    window_size: int = 1000,
    overlap: int = 500
):
    """
    Process a corpus through the entropy functor.

    Args:
        input_dir: Directory containing text JSON files
        output_dir: Output directory for trajectories
        language: "en" or "ja"
        window_size: Window size (words for EN, chars for JA)
        overlap: Window overlap
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select functor
    if language == "ja":
        functor = JapaneseEntropyFunctor()
    else:
        functor = EntropyFunctor()

    # Find text files
    text_files = list(Path(input_dir).glob("*.json"))
    text_files = [f for f in text_files if f.name != "manifest.json"]
    console.print(f"[blue]Processing {len(text_files)} texts for entropy...[/blue]")

    results = []

    for text_file in tqdm(text_files, desc="Extracting entropy"):
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            text = data.get("text", "")
            if not text:
                continue

            # Create windows
            if language == "ja":
                text_clean = re.sub(r'\s+', '', text)
                step = window_size - overlap
                windows = []
                for i in range(0, len(text_clean), step):
                    window = text_clean[i:i + window_size]
                    if len(window) >= window_size // 2:
                        windows.append(window)
                windows = windows if windows else [text]
            else:
                words = text.split()
                step = window_size - overlap
                windows = []
                for i in range(0, len(words), step):
                    window = ' '.join(words[i:i + window_size])
                    if len(window.split()) >= window_size // 2:
                        windows.append(window)
                windows = windows if windows else [text]

            # Apply functor
            trajectory = functor(windows)

            # Add metadata
            trajectory.metadata.update({
                "source_id": data.get("id", text_file.stem),
                "title": data.get("title", "Unknown"),
                "language": language,
            })

            # Save trajectory
            out_file = output_dir / f"{text_file.stem}_entropy.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory.to_dict(), f, ensure_ascii=False, indent=2)

            results.append({
                "id": data.get("id", text_file.stem),
                "title": data.get("title"),
                "mean_entropy": float(np.mean(trajectory.values)),
                "std_entropy": float(np.std(trajectory.values)),
                "max_entropy": float(np.max(trajectory.values)),
                "min_entropy": float(np.min(trajectory.values)),
                "trajectory_file": str(out_file.name),
            })

        except Exception as e:
            console.print(f"[red]Error processing {text_file}: {e}[/red]")
            continue

    # Save manifest
    manifest = {
        "functor": functor.name,
        "language": language,
        "window_size": window_size,
        "overlap": overlap,
        "count": len(results),
        "trajectories": results,
    }

    with open(output_dir / "manifest.json", 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    console.print(f"[bold green]✓ Processed {len(results)} texts for entropy[/bold green]")

    return results


def detect_entropy_spikes(
    trajectory: np.ndarray,
    threshold: float = 1.5
) -> List[Dict]:
    """
    Detect entropy spikes (potential plot twists/surprises).

    Args:
        trajectory: Entropy values
        threshold: Standard deviations above mean for spike

    Returns:
        List of spike locations with metadata
    """
    mean_val = np.mean(trajectory)
    std_val = np.std(trajectory)

    spike_threshold = mean_val + threshold * std_val
    spikes = []

    for i, val in enumerate(trajectory):
        if val > spike_threshold:
            spikes.append({
                "position": i / len(trajectory),
                "value": float(val),
                "z_score": float((val - mean_val) / std_val) if std_val > 0 else 0,
            })

    return spikes


@click.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True))
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path())
@click.option('--language', '-l', default='en', type=click.Choice(['en', 'ja']))
@click.option('--window-size', '-w', default=1000)
@click.option('--overlap', default=500)
def main(input_dir: str, output_dir: str, language: str, window_size: int, overlap: int):
    """Extract entropy trajectories from text corpus."""
    process_corpus(Path(input_dir), Path(output_dir), language, window_size, overlap)


if __name__ == "__main__":
    main()
