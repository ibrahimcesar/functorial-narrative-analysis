#!/usr/bin/env python3
"""
Visualize Chinese Classical Literature Analysis

Creates visualizations for Chinese classical texts analysis:
- Sentiment trajectories
- ICC class distribution by dynasty
- Emotional arc patterns
- Comparative analysis across genres

Usage:
    python scripts/visualize_chinese.py --input output/chinese --output visualizations/chinese
"""

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import click
from rich.console import Console

console = Console()

# Configure matplotlib for Chinese characters and JetBrains Mono
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Heiti SC', 'PingFang SC', 'sans-serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'medium'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Use JetBrains Mono for non-Chinese text
try:
    fm.fontManager.addfont('/Users/ibrahimcesar/Library/Fonts/JetBrainsMono-Regular.ttf')
except:
    pass


def load_results(input_dir: Path) -> List[Dict]:
    """Load analysis results."""
    summary_file = input_dir / "analysis_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("results", [])
    return []


def plot_icc_by_dynasty(results: List[Dict], output_path: Path):
    """Plot ICC class distribution by dynasty."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by dynasty
    dynasty_icc = {}
    for r in results:
        dynasty = r["dynasty"]
        icc = r["icc_class"]
        if dynasty not in dynasty_icc:
            dynasty_icc[dynasty] = {}
        dynasty_icc[dynasty][icc] = dynasty_icc[dynasty].get(icc, 0) + 1

    # Sort dynasties chronologically
    dynasty_order = ["Zhou", "Spring and Autumn", "Warring States", "Han", "Wei", "Zhou/Han"]
    dynasties = [d for d in dynasty_order if d in dynasty_icc]
    dynasties.extend([d for d in dynasty_icc if d not in dynasty_order])

    icc_classes = ["ICC-0", "ICC-1", "ICC-2", "ICC-3", "ICC-4", "ICC-5"]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']

    x = np.arange(len(dynasties))
    width = 0.12
    multiplier = 0

    for icc, color in zip(icc_classes, colors):
        counts = [dynasty_icc.get(d, {}).get(icc, 0) for d in dynasties]
        offset = width * multiplier
        ax.bar(x + offset, counts, width, label=icc, color=color, alpha=0.8)
        multiplier += 1

    ax.set_xlabel('Dynasty')
    ax.set_ylabel('Number of Texts')
    ax.set_title('ICC Classification Distribution by Dynasty\nChinese Classical Literature')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(dynasties, rotation=15)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path / "icc_by_dynasty.png", dpi=150)
    plt.close()
    console.print(f"[green]Saved: {output_path / 'icc_by_dynasty.png'}[/green]")


def plot_emotional_arcs(results: List[Dict], output_path: Path):
    """Plot emotional arc distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Overall distribution
    arc_counts = {}
    for r in results:
        arc = r["emotional_arc"]
        arc_counts[arc] = arc_counts.get(arc, 0) + 1

    arcs = list(arc_counts.keys())
    counts = [arc_counts[a] for a in arcs]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    wedges, texts, autotexts = ax1.pie(
        counts, labels=arcs, autopct='%1.1f%%',
        colors=colors[:len(arcs)], startangle=90
    )
    ax1.set_title('Emotional Arc Distribution')

    # By genre
    genre_arcs = {}
    for r in results:
        genre = r["genre"]
        arc = r["emotional_arc"]
        if genre not in genre_arcs:
            genre_arcs[genre] = {}
        genre_arcs[genre][arc] = genre_arcs[genre].get(arc, 0) + 1

    genres = list(genre_arcs.keys())
    arc_types = ["Rise-Fall", "Fall-Rise", "Rising", "Falling", "Stable"]

    x = np.arange(len(genres))
    width = 0.15

    for i, arc in enumerate(arc_types):
        counts = [genre_arcs.get(g, {}).get(arc, 0) for g in genres]
        ax2.bar(x + i * width, counts, width, label=arc, alpha=0.8)

    ax2.set_xlabel('Genre')
    ax2.set_ylabel('Count')
    ax2.set_title('Emotional Arcs by Genre')
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(genres, rotation=45, ha='right')
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path / "emotional_arcs.png", dpi=150)
    plt.close()
    console.print(f"[green]Saved: {output_path / 'emotional_arcs.png'}[/green]")


def plot_sentiment_distribution(results: List[Dict], output_path: Path):
    """Plot sentiment distribution across texts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Sentiment by genre
    ax = axes[0, 0]
    genre_sentiment = {}
    for r in results:
        genre = r["genre"]
        if genre not in genre_sentiment:
            genre_sentiment[genre] = []
        genre_sentiment[genre].append(r["sentiment_mean"])

    genres = list(genre_sentiment.keys())
    sentiment_data = [genre_sentiment[g] for g in genres]

    bp = ax.boxplot(sentiment_data, labels=genres, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(genres)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Genre')
    ax.set_ylabel('Mean Sentiment')
    ax.set_title('Sentiment Distribution by Genre')
    ax.tick_params(axis='x', rotation=45)

    # 2. Sentiment by dynasty
    ax = axes[0, 1]
    dynasty_sentiment = {}
    for r in results:
        dynasty = r["dynasty"]
        if dynasty not in dynasty_sentiment:
            dynasty_sentiment[dynasty] = []
        dynasty_sentiment[dynasty].append(r["sentiment_mean"])

    dynasties = list(dynasty_sentiment.keys())
    sentiment_data = [dynasty_sentiment[d] for d in dynasties]

    bp = ax.boxplot(sentiment_data, labels=dynasties, patch_artist=True)
    colors = plt.cm.Paired(np.linspace(0, 1, len(dynasties)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Dynasty')
    ax.set_ylabel('Mean Sentiment')
    ax.set_title('Sentiment Distribution by Dynasty')
    ax.tick_params(axis='x', rotation=15)

    # 3. Sentiment vs Entropy scatter
    ax = axes[1, 0]
    sentiments = [r["sentiment_mean"] for r in results]
    entropies = [r["entropy_mean"] for r in results]
    genres = [r["genre"] for r in results]

    unique_genres = list(set(genres))
    genre_colors = {g: plt.cm.tab10(i) for i, g in enumerate(unique_genres)}

    for genre in unique_genres:
        mask = [g == genre for g in genres]
        s = [sentiments[i] for i, m in enumerate(mask) if m]
        e = [entropies[i] for i, m in enumerate(mask) if m]
        ax.scatter(s, e, c=[genre_colors[genre]], label=genre, alpha=0.7, s=50)

    ax.set_xlabel('Mean Sentiment')
    ax.set_ylabel('Mean Entropy')
    ax.set_title('Sentiment vs Entropy')
    ax.legend(loc='best', fontsize=8)

    # 4. ICC confidence histogram
    ax = axes[1, 1]
    confidences = [r["icc_confidence"] for r in results]
    ax.hist(confidences, bins=20, color='steelblue', alpha=0.7, edgecolor='white')
    ax.set_xlabel('ICC Classification Confidence')
    ax.set_ylabel('Count')
    ax.set_title('ICC Classification Confidence Distribution')
    ax.axvline(x=np.mean(confidences), color='red', linestyle='--',
               label=f'Mean: {np.mean(confidences):.2f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / "sentiment_analysis.png", dpi=150)
    plt.close()
    console.print(f"[green]Saved: {output_path / 'sentiment_analysis.png'}[/green]")


def plot_top_texts_comparison(results: List[Dict], output_path: Path):
    """Compare top texts by various metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Sort by different metrics
    # 1. Most positive texts
    ax = axes[0, 0]
    top_positive = sorted(results, key=lambda x: -x["sentiment_mean"])[:10]
    titles = [f"{r['title'][:8]}" for r in top_positive]
    sentiments = [r["sentiment_mean"] for r in top_positive]
    colors = ['#2ecc71' if s > 0 else '#e74c3c' for s in sentiments]
    ax.barh(titles, sentiments, color=colors, alpha=0.8)
    ax.set_xlabel('Mean Sentiment')
    ax.set_title('Top 10 Most Positive Texts')
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    # 2. Most negative texts
    ax = axes[0, 1]
    top_negative = sorted(results, key=lambda x: x["sentiment_mean"])[:10]
    titles = [f"{r['title'][:8]}" for r in top_negative]
    sentiments = [r["sentiment_mean"] for r in top_negative]
    colors = ['#2ecc71' if s > 0 else '#e74c3c' for s in sentiments]
    ax.barh(titles, sentiments, color=colors, alpha=0.8)
    ax.set_xlabel('Mean Sentiment')
    ax.set_title('Top 10 Most Negative Texts')
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    # 3. Most variable sentiment (narrative tension)
    ax = axes[1, 0]
    top_variable = sorted(results, key=lambda x: -x["sentiment_std"])[:10]
    titles = [f"{r['title'][:8]}" for r in top_variable]
    stds = [r["sentiment_std"] for r in top_variable]
    ax.barh(titles, stds, color='#9b59b6', alpha=0.8)
    ax.set_xlabel('Sentiment Standard Deviation')
    ax.set_title('Top 10 Most Emotionally Variable Texts')

    # 4. Highest complexity
    ax = axes[1, 1]
    top_complex = sorted(results, key=lambda x: -x["narrative_complexity"])[:10]
    titles = [f"{r['title'][:8]}" for r in top_complex]
    complexity = [r["narrative_complexity"] for r in top_complex]
    ax.barh(titles, complexity, color='#f39c12', alpha=0.8)
    ax.set_xlabel('Narrative Complexity Score')
    ax.set_title('Top 10 Most Complex Narratives')

    plt.tight_layout()
    plt.savefig(output_path / "top_texts_comparison.png", dpi=150)
    plt.close()
    console.print(f"[green]Saved: {output_path / 'top_texts_comparison.png'}[/green]")


def plot_genre_radar(results: List[Dict], output_path: Path):
    """Create radar chart comparing genres."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Aggregate metrics by genre
    genre_metrics = {}
    for r in results:
        genre = r["genre"]
        if genre not in genre_metrics:
            genre_metrics[genre] = {
                "sentiment_mean": [],
                "sentiment_std": [],
                "entropy_mean": [],
                "complexity": [],
            }
        genre_metrics[genre]["sentiment_mean"].append(r["sentiment_mean"])
        genre_metrics[genre]["sentiment_std"].append(r["sentiment_std"])
        genre_metrics[genre]["entropy_mean"].append(r["entropy_mean"])
        genre_metrics[genre]["complexity"].append(r["narrative_complexity"])

    # Calculate averages
    genres = list(genre_metrics.keys())
    metrics = ["sentiment_mean", "sentiment_std", "entropy_mean", "complexity"]
    metric_labels = ["Positivity", "Emotional Range", "Entropy", "Complexity"]

    # Normalize metrics to 0-1 range
    all_values = {m: [] for m in metrics}
    for genre in genres:
        for m in metrics:
            avg = np.mean(genre_metrics[genre][m])
            all_values[m].append(avg)

    # Min-max normalization
    normalized = {}
    for m in metrics:
        min_val = min(all_values[m])
        max_val = max(all_values[m])
        if max_val - min_val > 0:
            normalized[m] = [(v - min_val) / (max_val - min_val) for v in all_values[m]]
        else:
            normalized[m] = [0.5] * len(all_values[m])

    # Plot
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    colors = plt.cm.Set2(np.linspace(0, 1, len(genres)))

    for i, genre in enumerate(genres):
        values = [normalized[m][i] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=genre)
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_title('Genre Comparison Radar Chart\nChinese Classical Literature')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(output_path / "genre_radar.png", dpi=150, bbox_inches='tight')
    plt.close()
    console.print(f"[green]Saved: {output_path / 'genre_radar.png'}[/green]")


def create_summary_dashboard(results: List[Dict], output_path: Path):
    """Create summary statistics dashboard."""
    fig = plt.figure(figsize=(16, 10))

    # Title
    fig.suptitle('Chinese Classical Literature Analysis Summary\n中國古典文獻分析',
                 fontsize=16, fontweight='bold')

    # Summary stats
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.axis('off')
    stats_text = f"""
    Total Texts: {len(results)}
    Total Characters: {sum(r['char_count'] for r in results):,}

    Mean Sentiment: {np.mean([r['sentiment_mean'] for r in results]):.3f}
    Mean Entropy: {np.mean([r['entropy_mean'] for r in results]):.3f}
    Mean Complexity: {np.mean([r['narrative_complexity'] for r in results]):.3f}

    Dynasties: {len(set(r['dynasty'] for r in results))}
    Genres: {len(set(r['genre'] for r in results))}
    """
    ax1.text(0.1, 0.9, stats_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')
    ax1.set_title('Summary Statistics', fontweight='bold')

    # ICC distribution pie
    ax2 = fig.add_subplot(2, 3, 2)
    icc_counts = {}
    for r in results:
        icc_counts[r["icc_class"]] = icc_counts.get(r["icc_class"], 0) + 1
    labels = list(icc_counts.keys())
    sizes = list(icc_counts.values())
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('ICC Classification')

    # Dynasty distribution
    ax3 = fig.add_subplot(2, 3, 3)
    dynasty_counts = {}
    for r in results:
        dynasty_counts[r["dynasty"]] = dynasty_counts.get(r["dynasty"], 0) + 1
    dynasties = list(dynasty_counts.keys())
    counts = [dynasty_counts[d] for d in dynasties]
    ax3.barh(dynasties, counts, color='steelblue', alpha=0.8)
    ax3.set_xlabel('Number of Texts')
    ax3.set_title('Distribution by Dynasty')

    # Genre distribution
    ax4 = fig.add_subplot(2, 3, 4)
    genre_counts = {}
    for r in results:
        genre_counts[r["genre"]] = genre_counts.get(r["genre"], 0) + 1
    genres = list(genre_counts.keys())
    counts = [genre_counts[g] for g in genres]
    ax4.barh(genres, counts, color='forestgreen', alpha=0.8)
    ax4.set_xlabel('Number of Texts')
    ax4.set_title('Distribution by Genre')

    # Sentiment histogram
    ax5 = fig.add_subplot(2, 3, 5)
    sentiments = [r["sentiment_mean"] for r in results]
    ax5.hist(sentiments, bins=20, color='coral', alpha=0.7, edgecolor='white')
    ax5.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Mean Sentiment')
    ax5.set_ylabel('Count')
    ax5.set_title('Sentiment Distribution')

    # Emotional arc
    ax6 = fig.add_subplot(2, 3, 6)
    arc_counts = {}
    for r in results:
        arc_counts[r["emotional_arc"]] = arc_counts.get(r["emotional_arc"], 0) + 1
    arcs = list(arc_counts.keys())
    counts = [arc_counts[a] for a in arcs]
    ax6.pie(counts, labels=arcs, autopct='%1.1f%%', startangle=90)
    ax6.set_title('Emotional Arcs')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path / "summary_dashboard.png", dpi=150)
    plt.close()
    console.print(f"[green]Saved: {output_path / 'summary_dashboard.png'}[/green]")


@click.command()
@click.option('--input', '-i', 'input_dir', default='output/chinese',
              type=click.Path(exists=True), help='Analysis results directory')
@click.option('--output', '-o', 'output_dir', default='visualizations/chinese',
              type=click.Path(), help='Output directory for visualizations')
def main(input_dir: str, output_dir: str):
    """
    Generate visualizations for Chinese classical literature analysis.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print("=" * 60)
    console.print("[bold]CHINESE CLASSICAL LITERATURE VISUALIZATION[/bold]")
    console.print(f"Input: {input_path}")
    console.print(f"Output: {output_path}")
    console.print("=" * 60)

    results = load_results(input_path)

    if not results:
        console.print("[red]No results found![/red]")
        return

    console.print(f"[blue]Loaded {len(results)} analysis results[/blue]")

    # Generate visualizations
    plot_icc_by_dynasty(results, output_path)
    plot_emotional_arcs(results, output_path)
    plot_sentiment_distribution(results, output_path)
    plot_top_texts_comparison(results, output_path)
    plot_genre_radar(results, output_path)
    create_summary_dashboard(results, output_path)

    console.print(f"\n[bold green]Done! Visualizations saved to: {output_path}[/bold green]")


if __name__ == "__main__":
    main()
