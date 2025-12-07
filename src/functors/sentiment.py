"""
F_sentiment: Emotional Valence Functor

Maps narrative states to positions on a happiness-sadness axis.
This is the Reagan et al. functor that identified the six story shapes.

Implements an ensemble of:
- VADER: Rule-based sentiment analyzer (fast, interpretable)
- BERT: Transformer-based multilingual sentiment (accurate, contextual)
"""

import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
import click
from rich.console import Console

from .base import BaseFunctor, Trajectory

console = Console()


class SentimentFunctor(BaseFunctor):
    """
    Sentiment analysis functor using VADER + optional BERT ensemble.

    Maps text windows to sentiment scores in [-1, 1] range where:
    - -1 = strongly negative
    - 0 = neutral
    - +1 = strongly positive
    """

    name = "sentiment"

    def __init__(
        self,
        method: str = "vader",
        use_bert: bool = False,
        bert_model: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    ):
        """
        Initialize sentiment functor.

        Args:
            method: "vader", "bert", or "ensemble"
            use_bert: Whether to use BERT (slower but more accurate)
            bert_model: HuggingFace model ID for BERT sentiment
        """
        self.method = method
        self.use_bert = use_bert or method in ["bert", "ensemble"]
        self.bert_model_name = bert_model

        # Initialize VADER
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

        # Initialize BERT if needed
        self.bert_pipeline = None
        if self.use_bert:
            self._init_bert()

    def _init_bert(self):
        """Initialize BERT sentiment pipeline."""
        try:
            from transformers import pipeline
            console.print("[yellow]Loading BERT sentiment model...[/yellow]")
            self.bert_pipeline = pipeline(
                "sentiment-analysis",
                model=self.bert_model_name,
                device=-1  # CPU; use 0 for GPU
            )
            console.print("[green]✓ BERT model loaded[/green]")
        except Exception as e:
            console.print(f"[red]Failed to load BERT: {e}. Falling back to VADER only.[/red]")
            self.use_bert = False
            self.method = "vader"

    def _vader_score(self, text: str) -> float:
        """
        Compute VADER sentiment score.

        Args:
            text: Input text

        Returns:
            Compound sentiment score in [-1, 1]
        """
        scores = self.vader.polarity_scores(text)
        return scores["compound"]

    def _bert_score(self, text: str) -> float:
        """
        Compute BERT sentiment score.

        Args:
            text: Input text (truncated to 512 tokens)

        Returns:
            Sentiment score mapped to [-1, 1]
        """
        if not self.bert_pipeline:
            return 0.0

        # Truncate for BERT
        truncated = text[:512]

        try:
            result = self.bert_pipeline(truncated)[0]
            label = result["label"]
            score = result["score"]

            # Map star ratings or labels to [-1, 1]
            if "star" in label.lower() or label.isdigit():
                # 1-5 star rating -> [-1, 1]
                stars = int(label.replace(" stars", "").replace(" star", ""))
                return (stars - 3) / 2
            elif label.lower() in ["positive", "pos"]:
                return score
            elif label.lower() in ["negative", "neg"]:
                return -score
            else:
                return 0.0

        except Exception:
            return 0.0

    def _score_window(self, text: str) -> float:
        """
        Compute sentiment score for a single window.

        Args:
            text: Window text

        Returns:
            Sentiment score in [-1, 1]
        """
        if self.method == "vader":
            return self._vader_score(text)
        elif self.method == "bert":
            return self._bert_score(text)
        else:  # ensemble
            vader = self._vader_score(text)
            bert = self._bert_score(text)
            return (vader + bert) / 2

    def __call__(self, windows: List[str]) -> Trajectory:
        """
        Apply sentiment functor to text windows.

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
                "n_windows": len(windows),
            }
        )


def process_corpus(
    input_dir: Path,
    output_dir: Path,
    method: str = "vader",
    window_size: int = 1000,
    overlap: int = 500
):
    """
    Process a corpus of texts through the sentiment functor.

    Args:
        input_dir: Directory containing JSON files with text
        output_dir: Output directory for trajectories
        method: Sentiment method ("vader", "bert", "ensemble")
        window_size: Window size in words
        overlap: Window overlap in words
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize functor
    functor = SentimentFunctor(method=method)

    # Find all text files
    text_files = list(Path(input_dir).glob("*.json"))
    console.print(f"[blue]Processing {len(text_files)} texts...[/blue]")

    results = []

    for text_file in tqdm(text_files, desc="Extracting sentiment"):
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both raw text and preprocessed (windowed) formats
            if "windows" in data:
                # Already preprocessed
                windows = data["windows"]
                trajectory = functor(windows)
                word_count = data.get("total_words", sum(len(w.split()) for w in windows))
            elif "text" in data:
                # Raw text - need to window it
                text = data["text"]
                if not text:
                    continue
                trajectory = functor.process_text(text, window_size, overlap)
                word_count = len(text.split())
            else:
                continue

            # Add metadata
            trajectory.metadata.update({
                "source_id": data.get("id", text_file.stem),
                "title": data.get("title", "Unknown"),
                "author": data.get("author", "Unknown"),
                "word_count": data.get("word_count", word_count),
            })

            # Save trajectory
            out_file = output_dir / f"{text_file.stem}_sentiment.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory.to_dict(), f, indent=2)

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
        "functor": "sentiment",
        "method": method,
        "window_size": window_size,
        "overlap": overlap,
        "count": len(results),
        "trajectories": results,
    }

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    console.print(f"[bold green]✓ Processed {len(results)} texts[/bold green]")
    console.print(f"[green]Saved to {output_dir}[/green]")


@click.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True),
              help='Input directory with text JSON files')
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(),
              help='Output directory for trajectories')
@click.option('--method', '-m', default='vader',
              type=click.Choice(['vader', 'bert', 'ensemble']),
              help='Sentiment analysis method')
@click.option('--window-size', '-w', default=1000, help='Window size in words')
@click.option('--overlap', default=500, help='Window overlap in words')
def main(input_dir: str, output_dir: str, method: str, window_size: int, overlap: int):
    """Extract sentiment trajectories from text corpus."""
    process_corpus(
        Path(input_dir),
        Path(output_dir),
        method=method,
        window_size=window_size,
        overlap=overlap
    )


if __name__ == "__main__":
    main()
