#!/usr/bin/env python3
"""
Analyze Chinese Classical Literature Corpus

Runs multi-functor analysis on the Chinese classical texts corpus
using functors adapted for classical Chinese.

Usage:
    python scripts/analyze_chinese_corpus.py --input data/raw/chinese --output output/chinese
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

import numpy as np
import click
from rich.console import Console
from rich.table import Table
from scipy.stats import pearsonr
from tqdm import tqdm

# Import functors
from src.functors.chinese_sentiment import ClassicalChineseSentimentFunctor, create_windows_chinese
from src.functors.entropy import EntropyFunctor
from src.detectors.icc import ICCDetector

console = Console()


@dataclass
class ChineseAnalysisResult:
    """Results from Chinese text analysis."""
    id: str
    title: str
    title_en: str
    author: str
    dynasty: str
    genre: str
    char_count: int

    # ICC classification
    icc_class: str
    icc_confidence: float

    # Sentiment trajectory
    sentiment_mean: float
    sentiment_std: float
    sentiment_range: float

    # Entropy trajectory
    entropy_mean: float
    entropy_std: float

    # Derived patterns
    emotional_arc: str  # Rise, Fall, Rise-Fall, etc.
    narrative_complexity: float


def analyze_chinese_text(
    text: str,
    metadata: Dict,
    window_size: int = 500,
    overlap: int = 250
) -> Optional[ChineseAnalysisResult]:
    """
    Analyze a single Chinese text.

    Args:
        text: Full text content
        metadata: Text metadata (title, author, etc.)
        window_size: Window size in characters
        overlap: Window overlap

    Returns:
        ChineseAnalysisResult or None on error
    """
    # Create windows
    windows = create_windows_chinese(text, window_size, overlap)

    if len(windows) < 3:
        return None

    # Initialize functors
    sentiment_functor = ClassicalChineseSentimentFunctor()
    entropy_functor = EntropyFunctor(method="combined")

    # Apply functors
    sentiment_traj = sentiment_functor(windows)
    entropy_traj = entropy_functor(windows)

    # ICC classification
    detector = ICCDetector()
    icc_result = detector.detect(sentiment_traj.values)

    # Analyze emotional arc
    n = len(sentiment_traj.values)
    if n >= 4:
        first_quarter = np.mean(sentiment_traj.values[:n//4])
        mid = np.mean(sentiment_traj.values[n//4:3*n//4])
        last_quarter = np.mean(sentiment_traj.values[3*n//4:])

        if first_quarter < mid and mid > last_quarter:
            emotional_arc = "Rise-Fall"
        elif first_quarter > mid and mid < last_quarter:
            emotional_arc = "Fall-Rise"
        elif first_quarter < last_quarter:
            emotional_arc = "Rising"
        elif first_quarter > last_quarter:
            emotional_arc = "Falling"
        else:
            emotional_arc = "Stable"
    else:
        emotional_arc = "Unknown"

    # Narrative complexity score
    complexity = float(
        np.mean(entropy_traj.values) * 0.6 +
        np.std(sentiment_traj.values) * 0.4
    )

    return ChineseAnalysisResult(
        id=metadata.get("id", "unknown"),
        title=metadata.get("title", "Unknown"),
        title_en=metadata.get("title_en", "Unknown"),
        author=metadata.get("author", "Unknown"),
        dynasty=metadata.get("dynasty", "Unknown"),
        genre=metadata.get("genre", "unknown"),
        char_count=len(text),
        icc_class=icc_result.icc_class,
        icc_confidence=icc_result.confidence,
        sentiment_mean=float(np.mean(sentiment_traj.values)),
        sentiment_std=float(np.std(sentiment_traj.values)),
        sentiment_range=float(np.max(sentiment_traj.values) - np.min(sentiment_traj.values)),
        entropy_mean=float(np.mean(entropy_traj.values)),
        entropy_std=float(np.std(entropy_traj.values)),
        emotional_arc=emotional_arc,
        narrative_complexity=complexity,
    )


def process_corpus(
    input_dir: Path,
    output_dir: Path,
    window_size: int = 500,
    overlap: int = 250,
    min_chars: int = 2000,
) -> List[ChineseAnalysisResult]:
    """
    Process entire Chinese corpus.

    Args:
        input_dir: Directory with text JSON files
        output_dir: Output directory
        window_size: Analysis window size
        overlap: Window overlap
        min_chars: Minimum text length

    Returns:
        List of analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts_dir = input_dir / "texts"
    if not texts_dir.exists():
        texts_dir = input_dir

    text_files = list(texts_dir.glob("*.json"))
    text_files = [f for f in text_files if f.name not in ["manifest.json", "metadata.json"]]

    console.print(f"[blue]Processing {len(text_files)} Chinese texts...[/blue]")

    results = []

    for text_file in tqdm(text_files, desc="Analyzing"):
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            text = data.get("text", "")
            if len(text) < min_chars:
                continue

            result = analyze_chinese_text(
                text=text,
                metadata=data,
                window_size=window_size,
                overlap=overlap
            )

            if result:
                results.append(result)

                # Save individual result
                out_file = output_dir / f"{data.get('id', text_file.stem)}_analysis.json"
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(asdict(result), f, ensure_ascii=False, indent=2)

        except Exception as e:
            console.print(f"[red]Error processing {text_file}: {e}[/red]")
            continue

    # Save summary
    summary = {
        "corpus": "chinese_classical",
        "total_texts": len(results),
        "total_characters": sum(r.char_count for r in results),
        "results": [asdict(r) for r in results]
    }

    with open(output_dir / "analysis_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    console.print(f"[green]✓ Analyzed {len(results)} texts[/green]")

    return results


def display_results(results: List[ChineseAnalysisResult]):
    """Display results in a formatted table."""
    table = Table(title="Chinese Classical Literature Analysis")

    table.add_column("Title", style="cyan", max_width=15)
    table.add_column("English", style="dim", max_width=20)
    table.add_column("Dynasty", style="yellow")
    table.add_column("Genre", style="green")
    table.add_column("ICC", style="magenta")
    table.add_column("Sent", justify="right")
    table.add_column("Arc", style="blue")

    for r in sorted(results, key=lambda x: -x.char_count)[:25]:
        table.add_row(
            r.title[:14] + "..." if len(r.title) > 15 else r.title,
            r.title_en[:19] + "..." if len(r.title_en) > 20 else r.title_en,
            r.dynasty[:8],
            r.genre[:10],
            r.icc_class,
            f"{r.sentiment_mean:.2f}",
            r.emotional_arc
        )

    console.print(table)

    # Summary statistics
    console.print("\n[bold]ICC Distribution:[/bold]")
    icc_counts = {}
    for r in results:
        icc_counts[r.icc_class] = icc_counts.get(r.icc_class, 0) + 1
    for icc, count in sorted(icc_counts.items()):
        pct = 100 * count / len(results)
        console.print(f"  {icc}: {count} ({pct:.1f}%)")

    console.print("\n[bold]Emotional Arc Distribution:[/bold]")
    arc_counts = {}
    for r in results:
        arc_counts[r.emotional_arc] = arc_counts.get(r.emotional_arc, 0) + 1
    for arc, count in sorted(arc_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results)
        console.print(f"  {arc}: {count} ({pct:.1f}%)")

    # By dynasty
    console.print("\n[bold]By Dynasty:[/bold]")
    dynasty_stats = {}
    for r in results:
        if r.dynasty not in dynasty_stats:
            dynasty_stats[r.dynasty] = {"count": 0, "sentiment_sum": 0, "entropy_sum": 0}
        dynasty_stats[r.dynasty]["count"] += 1
        dynasty_stats[r.dynasty]["sentiment_sum"] += r.sentiment_mean
        dynasty_stats[r.dynasty]["entropy_sum"] += r.entropy_mean

    for dynasty, stats in sorted(dynasty_stats.items(), key=lambda x: -x[1]["count"]):
        avg_sent = stats["sentiment_sum"] / stats["count"]
        avg_ent = stats["entropy_sum"] / stats["count"]
        console.print(f"  {dynasty}: {stats['count']} texts, avg sentiment={avg_sent:.3f}, avg entropy={avg_ent:.3f}")


@click.command()
@click.option('--input', '-i', 'input_dir', default='data/raw/chinese',
              type=click.Path(exists=True), help='Input directory')
@click.option('--output', '-o', 'output_dir', default='output/chinese',
              type=click.Path(), help='Output directory')
@click.option('--window-size', '-w', default=500, help='Window size (chars)')
@click.option('--overlap', default=250, help='Window overlap')
@click.option('--min-chars', '-m', default=2000, help='Minimum text length')
def main(input_dir: str, output_dir: str, window_size: int, overlap: int, min_chars: int):
    """
    Analyze Chinese classical literature corpus.

    Runs sentiment and entropy analysis on classical Chinese texts
    with ICC classification.
    """
    console.print("=" * 60)
    console.print("[bold]CHINESE CLASSICAL LITERATURE ANALYSIS[/bold]")
    console.print(f"Input: {input_dir}")
    console.print(f"Output: {output_dir}")
    console.print("=" * 60)

    results = process_corpus(
        Path(input_dir),
        Path(output_dir),
        window_size=window_size,
        overlap=overlap,
        min_chars=min_chars
    )

    if results:
        display_results(results)

    console.print(f"\n[bold green]Done! Results saved to: {output_dir}[/bold green]")


if __name__ == "__main__":
    main()
