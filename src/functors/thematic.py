"""
F_thematic: Thematic Distance Functor

Maps narrative states to positions in semantic space, measuring
thematic drift and conceptual distance over the narrative arc.

Uses sentence embeddings to track:
- Thematic coherence (similarity to overall theme)
- Thematic drift (distance from previous window)
- Conceptual novelty (distance from running centroid)

This functor captures the "semantic shape" of a narrative -
how topics and themes evolve through the text.
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm
import click
from rich.console import Console

from .base import BaseFunctor, Trajectory

console = Console()


class ThematicFunctor(BaseFunctor):
    """
    Thematic functor measuring semantic drift in narrative space.

    Uses TF-IDF vectors as a lightweight embedding approach.
    For production, consider using sentence-transformers.
    """

    name = "thematic"

    def __init__(self, method: str = "tfidf", vocab_size: int = 5000):
        """
        Initialize thematic functor.

        Args:
            method: "tfidf" for TF-IDF vectors, "word2vec" for word embeddings
            vocab_size: Maximum vocabulary size for TF-IDF
        """
        self.method = method
        self.vocab_size = vocab_size
        self._vocabulary = None
        self._idf = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        # Remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", " ", text)
        words = text.split()
        # Filter short words and numbers
        words = [w for w in words if len(w) > 2 and not w.isdigit()]
        return words

    def _build_vocabulary(self, windows: List[str]) -> None:
        """Build vocabulary from all windows."""
        word_freq = {}
        for window in windows:
            for word in self._tokenize(window):
                word_freq[word] = word_freq.get(word, 0) + 1

        # Take top vocab_size words by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        self._vocabulary = {word: i for i, (word, _) in enumerate(sorted_words[:self.vocab_size])}

        # Compute IDF
        n_docs = len(windows)
        doc_freq = {word: 0 for word in self._vocabulary}
        for window in windows:
            seen = set()
            for word in self._tokenize(window):
                if word in self._vocabulary and word not in seen:
                    doc_freq[word] += 1
                    seen.add(word)

        self._idf = np.zeros(len(self._vocabulary))
        for word, idx in self._vocabulary.items():
            df = doc_freq[word]
            self._idf[idx] = np.log((n_docs + 1) / (df + 1)) + 1  # Smoothed IDF

    def _window_to_vector(self, window: str) -> np.ndarray:
        """Convert window to TF-IDF vector."""
        words = self._tokenize(window)

        # Term frequency
        tf = np.zeros(len(self._vocabulary))
        for word in words:
            if word in self._vocabulary:
                tf[self._vocabulary[word]] += 1

        # Normalize TF
        if tf.sum() > 0:
            tf = tf / tf.sum()

        # TF-IDF
        tfidf = tf * self._idf

        # L2 normalize
        norm = np.linalg.norm(tfidf)
        if norm > 0:
            tfidf = tfidf / norm

        return tfidf

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between vectors."""
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 0 and norm2 > 0:
            return dot / (norm1 * norm2)
        return 0.0

    def __call__(self, windows: List[str]) -> Trajectory:
        """
        Apply thematic functor to text windows.

        Computes thematic drift as 1 - similarity to previous window.

        Args:
            windows: List of text windows

        Returns:
            Trajectory with thematic drift scores
        """
        if len(windows) < 2:
            return Trajectory(
                values=np.array([0.5]),
                time_points=np.array([0.5]),
                functor_name=self.name,
                metadata={"method": self.method, "n_windows": len(windows)}
            )

        # Build vocabulary
        self._build_vocabulary(windows)

        # Convert all windows to vectors
        vectors = [self._window_to_vector(w) for w in windows]

        # Compute thematic drift (1 - similarity to previous)
        drift_scores = [0.5]  # First window has no previous
        for i in range(1, len(vectors)):
            sim = self._cosine_similarity(vectors[i], vectors[i-1])
            drift = 1 - sim  # Higher drift = more thematic change
            drift_scores.append(drift)

        values = np.array(drift_scores)
        time_points = np.linspace(0, 1, len(values))

        return Trajectory(
            values=values,
            time_points=time_points,
            functor_name=self.name,
            metadata={
                "method": self.method,
                "vocab_size": len(self._vocabulary),
                "n_windows": len(windows),
            }
        )


class ThematicCoherenceFunctor(BaseFunctor):
    """
    Measures thematic coherence - similarity to overall narrative theme.

    High coherence = window closely matches overall themes
    Low coherence = window diverges from main themes (subplot, digression)
    """

    name = "thematic_coherence"

    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self._vocabulary = None
        self._idf = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r"[^\w\s']", " ", text)
        words = text.split()
        words = [w for w in words if len(w) > 2 and not w.isdigit()]
        return words

    def _build_vocabulary(self, windows: List[str]) -> None:
        """Build vocabulary from all windows."""
        word_freq = {}
        for window in windows:
            for word in self._tokenize(window):
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        self._vocabulary = {word: i for i, (word, _) in enumerate(sorted_words[:self.vocab_size])}

        n_docs = len(windows)
        doc_freq = {word: 0 for word in self._vocabulary}
        for window in windows:
            seen = set()
            for word in self._tokenize(window):
                if word in self._vocabulary and word not in seen:
                    doc_freq[word] += 1
                    seen.add(word)

        self._idf = np.zeros(len(self._vocabulary))
        for word, idx in self._vocabulary.items():
            df = doc_freq[word]
            self._idf[idx] = np.log((n_docs + 1) / (df + 1)) + 1

    def _window_to_vector(self, window: str) -> np.ndarray:
        """Convert window to TF-IDF vector."""
        words = self._tokenize(window)
        tf = np.zeros(len(self._vocabulary))
        for word in words:
            if word in self._vocabulary:
                tf[self._vocabulary[word]] += 1
        if tf.sum() > 0:
            tf = tf / tf.sum()
        tfidf = tf * self._idf
        norm = np.linalg.norm(tfidf)
        if norm > 0:
            tfidf = tfidf / norm
        return tfidf

    def __call__(self, windows: List[str]) -> Trajectory:
        """
        Compute thematic coherence for each window.

        Coherence = similarity to centroid of all windows.
        """
        if len(windows) < 2:
            return Trajectory(
                values=np.array([1.0]),
                time_points=np.array([0.5]),
                functor_name=self.name,
                metadata={"n_windows": len(windows)}
            )

        self._build_vocabulary(windows)
        vectors = [self._window_to_vector(w) for w in windows]

        # Compute centroid (overall theme)
        centroid = np.mean(vectors, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        # Coherence = similarity to centroid
        coherence_scores = []
        for v in vectors:
            sim = np.dot(v, centroid)
            coherence_scores.append(sim)

        values = np.array(coherence_scores)
        time_points = np.linspace(0, 1, len(values))

        return Trajectory(
            values=values,
            time_points=time_points,
            functor_name=self.name,
            metadata={"n_windows": len(windows)}
        )


class JapaneseThematicFunctor(BaseFunctor):
    """
    Thematic functor for Japanese text.

    Uses character n-grams since Japanese lacks word boundaries.
    """

    name = "thematic_ja"

    def __init__(self, ngram_size: int = 3, vocab_size: int = 5000):
        self.ngram_size = ngram_size
        self.vocab_size = vocab_size
        self._vocabulary = None
        self._idf = None

    def _get_ngrams(self, text: str) -> List[str]:
        """Extract character n-grams from Japanese text."""
        # Remove whitespace and special characters
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[。、！？「」『』（）\[\]【】]', '', text)

        ngrams = []
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.append(text[i:i + self.ngram_size])
        return ngrams

    def _build_vocabulary(self, windows: List[str]) -> None:
        """Build vocabulary from n-grams."""
        ngram_freq = {}
        for window in windows:
            for ngram in self._get_ngrams(window):
                ngram_freq[ngram] = ngram_freq.get(ngram, 0) + 1

        sorted_ngrams = sorted(ngram_freq.items(), key=lambda x: -x[1])
        self._vocabulary = {ng: i for i, (ng, _) in enumerate(sorted_ngrams[:self.vocab_size])}

        n_docs = len(windows)
        doc_freq = {ng: 0 for ng in self._vocabulary}
        for window in windows:
            seen = set()
            for ngram in self._get_ngrams(window):
                if ngram in self._vocabulary and ngram not in seen:
                    doc_freq[ngram] += 1
                    seen.add(ngram)

        self._idf = np.zeros(len(self._vocabulary))
        for ngram, idx in self._vocabulary.items():
            df = doc_freq[ngram]
            self._idf[idx] = np.log((n_docs + 1) / (df + 1)) + 1

    def _window_to_vector(self, window: str) -> np.ndarray:
        """Convert window to TF-IDF vector using n-grams."""
        ngrams = self._get_ngrams(window)
        tf = np.zeros(len(self._vocabulary))
        for ngram in ngrams:
            if ngram in self._vocabulary:
                tf[self._vocabulary[ngram]] += 1
        if tf.sum() > 0:
            tf = tf / tf.sum()
        tfidf = tf * self._idf
        norm = np.linalg.norm(tfidf)
        if norm > 0:
            tfidf = tfidf / norm
        return tfidf

    def __call__(self, windows: List[str]) -> Trajectory:
        """Apply thematic functor to Japanese text windows."""
        if len(windows) < 2:
            return Trajectory(
                values=np.array([0.5]),
                time_points=np.array([0.5]),
                functor_name=self.name,
                metadata={"n_windows": len(windows)}
            )

        self._build_vocabulary(windows)
        vectors = [self._window_to_vector(w) for w in windows]

        # Compute drift
        drift_scores = [0.5]
        for i in range(1, len(vectors)):
            sim = np.dot(vectors[i], vectors[i-1])
            drift = 1 - sim
            drift_scores.append(drift)

        values = np.array(drift_scores)
        time_points = np.linspace(0, 1, len(values))

        return Trajectory(
            values=values,
            time_points=time_points,
            functor_name=self.name,
            metadata={
                "ngram_size": self.ngram_size,
                "vocab_size": len(self._vocabulary),
                "n_windows": len(windows),
            }
        )


def process_corpus(
    input_dir: Path,
    output_dir: Path,
    language: str = "en",
    window_size: int = 1000,
    overlap: int = 500
):
    """
    Process a corpus through the thematic functor.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if language == "ja":
        functor = JapaneseThematicFunctor()
    else:
        functor = ThematicFunctor()

    text_files = list(Path(input_dir).glob("*.json"))
    text_files = [f for f in text_files if f.name != "manifest.json"]
    console.print(f"[blue]Processing {len(text_files)} texts for thematic drift...[/blue]")

    results = []

    for text_file in tqdm(text_files, desc="Extracting thematic trajectories"):
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

            trajectory = functor(windows)
            trajectory.metadata.update({
                "source_id": data.get("id", text_file.stem),
                "title": data.get("title", "Unknown"),
                "language": language,
            })

            out_file = output_dir / f"{text_file.stem}_thematic.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory.to_dict(), f, ensure_ascii=False, indent=2)

            results.append({
                "id": data.get("id", text_file.stem),
                "title": data.get("title"),
                "mean_drift": float(np.mean(trajectory.values)),
                "std_drift": float(np.std(trajectory.values)),
                "trajectory_file": str(out_file.name),
            })

        except Exception as e:
            console.print(f"[red]Error processing {text_file}: {e}[/red]")
            continue

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

    console.print(f"[bold green]✓ Processed {len(results)} texts for thematic drift[/bold green]")


@click.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True))
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path())
@click.option('--language', '-l', default='en', type=click.Choice(['en', 'ja']))
@click.option('--window-size', '-w', default=1000)
@click.option('--overlap', default=500)
def main(input_dir: str, output_dir: str, language: str, window_size: int, overlap: int):
    """Extract thematic drift trajectories from text corpus."""
    process_corpus(Path(input_dir), Path(output_dir), language, window_size, overlap)


if __name__ == "__main__":
    main()
