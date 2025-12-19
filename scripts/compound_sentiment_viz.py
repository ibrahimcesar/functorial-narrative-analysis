#!/usr/bin/env python3
"""
Compound Sentiment Visualization

Creates a visualization showing cumulative sentiment trajectory -
a line that rises with positive sentiment and falls with negative,
creating a visual "emotional altitude" chart for a narrative.

The compound line integrates sentiment over time:
- When sentiment is positive, the line rises
- When sentiment is negative, the line falls
- The slope represents emotional momentum
- Peaks show moments of sustained joy
- Valleys show sustained suffering

This is particularly powerful for novels like Anna Karenina where
the emotional trajectory has a clear arc.
"""

import json
import re
import math
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Optional
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from functors.russian_sentiment import (
    RussianSentimentAnalyzer,
    normalize_russian
)


def load_russian_text(json_path: Path) -> str:
    """Load Russian text from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('text', '')


def create_windows(text: str, window_size: int = 500, overlap: int = 250) -> List[str]:
    """Create overlapping windows from text."""
    words = text.split()
    windows = []
    step = window_size - overlap

    for i in range(0, len(words), step):
        window_words = words[i:i + window_size]
        if len(window_words) >= window_size // 2:
            windows.append(' '.join(window_words))

    return windows


def analyze_sentiment_windows(text: str, window_size: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze sentiment for sliding windows through the text.

    Returns:
        positions: Normalized positions [0, 1]
        sentiments: Sentiment scores [-1, 1]
    """
    analyzer = RussianSentimentAnalyzer(use_stemming=True)
    windows = create_windows(text, window_size=window_size, overlap=window_size // 2)

    sentiments = []
    for window in windows:
        result = analyzer.analyze(window)
        sentiments.append(result.compound)

    positions = np.linspace(0, 1, len(sentiments))
    return positions, np.array(sentiments)


def compute_compound_sentiment(sentiments: np.ndarray, normalize: bool = True,
                                center: bool = True) -> np.ndarray:
    """
    Compute compound (cumulative) sentiment.

    The compound sentiment integrates sentiment over time:
    - Positive sentiment adds to the cumulative value
    - Negative sentiment subtracts from it

    Args:
        sentiments: Array of sentiment scores
        normalize: Whether to normalize to [-1, 1] range
        center: Whether to center the sentiments (subtract mean) before cumsum

    Returns:
        Cumulative sentiment trajectory
    """
    # Center the sentiments to remove overall bias
    if center:
        centered = sentiments - np.mean(sentiments)
    else:
        centered = sentiments

    # Cumulative sum
    compound = np.cumsum(centered)

    if normalize:
        # Normalize to roughly [-1, 1] range while preserving shape
        max_abs = max(abs(compound.min()), abs(compound.max()), 1)
        compound = compound / max_abs

    return compound


def plot_compound_sentiment(
    text: str,
    title: str = "Compound Sentiment Trajectory",
    output_path: Optional[Path] = None,
    window_size: int = 500,
    smooth_sigma: float = 3.0,
    show_raw: bool = True,
    annotations: Optional[List[Tuple[float, str]]] = None
):
    """
    Create compound sentiment visualization.

    Args:
        text: Input text
        title: Plot title
        output_path: Where to save the figure
        window_size: Size of analysis windows
        smooth_sigma: Gaussian smoothing sigma
        show_raw: Whether to show raw sentiment in addition to compound
        annotations: List of (position, label) tuples for key moments
    """
    # Analyze sentiment
    positions, sentiments = analyze_sentiment_windows(text, window_size)

    # Compute compound sentiment
    compound = compute_compound_sentiment(sentiments)

    # Smooth both
    sentiments_smooth = gaussian_filter1d(sentiments, sigma=smooth_sigma)
    compound_smooth = gaussian_filter1d(compound, sigma=smooth_sigma)

    # Create figure
    if show_raw:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                  gridspec_kw={'height_ratios': [2, 1]})
        ax_compound, ax_raw = axes
    else:
        fig, ax_compound = plt.subplots(1, 1, figsize=(14, 6))
        ax_raw = None

    # Style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Color based on compound value
    # Create gradient fill
    for i in range(len(positions) - 1):
        color = '#2ecc71' if compound_smooth[i] > 0 else '#e74c3c'
        alpha = min(0.6, abs(compound_smooth[i]) * 0.8 + 0.1)
        ax_compound.fill_between(
            [positions[i], positions[i+1]],
            [0, 0],
            [compound_smooth[i], compound_smooth[i+1]],
            color=color,
            alpha=alpha,
            linewidth=0
        )

    # Plot main compound line
    ax_compound.plot(positions, compound_smooth, color='#2c3e50', linewidth=2.5,
                     label='Compound Sentiment')

    # Zero line
    ax_compound.axhline(y=0, color='#7f8c8d', linestyle='-', linewidth=1, alpha=0.5)

    # Mark peaks and valleys
    # Find local maxima and minima
    from scipy.signal import argrelextrema

    # Use smoothed data for extrema detection
    heavily_smoothed = gaussian_filter1d(compound, sigma=smooth_sigma * 3)

    maxima = argrelextrema(heavily_smoothed, np.greater, order=15)[0]
    minima = argrelextrema(heavily_smoothed, np.less, order=15)[0]

    # Plot peaks
    for idx in maxima[:5]:  # Top 5 peaks
        ax_compound.scatter(positions[idx], compound_smooth[idx],
                           color='#27ae60', s=100, zorder=5, edgecolor='white', linewidth=2)

    # Plot valleys
    for idx in minima[:5]:  # Bottom 5 valleys
        ax_compound.scatter(positions[idx], compound_smooth[idx],
                           color='#c0392b', s=100, zorder=5, edgecolor='white', linewidth=2)

    # Add annotations if provided
    if annotations:
        for pos, label in annotations:
            idx = int(pos * len(compound_smooth))
            if 0 <= idx < len(compound_smooth):
                y_val = compound_smooth[idx]
                ax_compound.annotate(
                    label,
                    xy=(pos, y_val),
                    xytext=(pos, y_val + 0.15 if y_val > 0 else y_val - 0.15),
                    fontsize=9,
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='#34495e', lw=1),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                )

    # Labels for compound plot
    ax_compound.set_ylabel('Emotional Altitude', fontsize=12, fontweight='bold')
    ax_compound.set_title(f'{title}\nCompound Sentiment Trajectory', fontsize=14, fontweight='bold')
    ax_compound.set_xlim(0, 1)

    # Add rise/fall indicators on the right side
    ax_compound.text(0.98, 0.95, '↑ Rising: Joy, Hope, Love', transform=ax_compound.transAxes,
                    fontsize=10, color='#27ae60', verticalalignment='top', ha='right')
    ax_compound.text(0.98, 0.05, '↓ Falling: Sorrow, Fear, Despair', transform=ax_compound.transAxes,
                    fontsize=10, color='#c0392b', verticalalignment='bottom', ha='right')

    # Raw sentiment plot
    if ax_raw is not None:
        # Fill positive/negative regions
        ax_raw.fill_between(positions, 0, sentiments_smooth,
                           where=sentiments_smooth > 0,
                           color='#2ecc71', alpha=0.3, label='Positive')
        ax_raw.fill_between(positions, 0, sentiments_smooth,
                           where=sentiments_smooth < 0,
                           color='#e74c3c', alpha=0.3, label='Negative')

        # Plot line
        ax_raw.plot(positions, sentiments_smooth, color='#2c3e50', linewidth=1.5)
        ax_raw.axhline(y=0, color='#7f8c8d', linestyle='-', linewidth=1, alpha=0.5)

        ax_raw.set_ylabel('Instantaneous\nSentiment', fontsize=11, fontweight='bold')
        ax_raw.set_xlabel('Narrative Progress', fontsize=12, fontweight='bold')
        ax_raw.set_ylim(-1, 1)
        ax_raw.legend(loc='upper right', fontsize=9)
    else:
        ax_compound.set_xlabel('Narrative Progress', fontsize=12, fontweight='bold')

    # Statistics annotation
    stats_text = (
        f"Peak: {compound_smooth.max():.2f} at {positions[np.argmax(compound_smooth)]:.0%}\n"
        f"Valley: {compound_smooth.min():.2f} at {positions[np.argmin(compound_smooth)]:.0%}\n"
        f"Final: {compound_smooth[-1]:.2f}\n"
        f"Volatility: {np.std(sentiments):.3f}"
    )
    ax_compound.annotate(
        stats_text,
        xy=(0.98, 0.98), xycoords='axes fraction',
        fontsize=9, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#bdc3c7')
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()

    return {
        'positions': positions,
        'sentiments': sentiments,
        'compound': compound,
        'peak_position': positions[np.argmax(compound_smooth)],
        'valley_position': positions[np.argmin(compound_smooth)],
        'final_compound': compound_smooth[-1]
    }


def plot_multiple_compounds(
    texts: List[Tuple[str, str]],  # (text, title) pairs
    output_path: Optional[Path] = None,
    window_size: int = 500,
    smooth_sigma: float = 5.0
):
    """
    Compare compound sentiment trajectories of multiple texts.

    Args:
        texts: List of (text, title) tuples
        output_path: Where to save
        window_size: Analysis window size
        smooth_sigma: Smoothing sigma
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(texts)))

    for i, (text, title) in enumerate(texts):
        positions, sentiments = analyze_sentiment_windows(text, window_size)
        compound = compute_compound_sentiment(sentiments)
        compound_smooth = gaussian_filter1d(compound, sigma=smooth_sigma)

        # Interpolate to common x-axis
        common_x = np.linspace(0, 1, 200)
        compound_interp = np.interp(common_x, positions, compound_smooth)

        ax.plot(common_x, compound_interp, color=colors[i], linewidth=2.5,
                label=title[:40] + ('...' if len(title) > 40 else ''))

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Narrative Progress', fontsize=12)
    ax.set_ylabel('Compound Sentiment (Emotional Altitude)', fontsize=12)
    ax.set_title('Comparative Emotional Trajectories', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Demo with Anna Karenina
    data_path = Path(__file__).parent.parent / "data/raw/russian/texts/anna_karenina_ru.json"

    if data_path.exists():
        print("Loading Anna Karenina...")
        text = load_russian_text(data_path)
        print(f"Text length: {len(text):,} characters")

        # Output directory
        output_dir = Path(__file__).parent.parent / "output/compound_sentiment"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Key moments in Anna Karenina (approximate positions)
        anna_annotations = [
            (0.05, "Opening"),
            (0.20, "Laska hunting"),
            (0.35, "Anna-Vronsky affair"),
            (0.65, "Descent begins"),
            (0.85, "Crisis"),
            (0.95, "Tragic end")
        ]

        # Create visualization
        result = plot_compound_sentiment(
            text,
            title="Anna Karenina (Анна Каренина)",
            output_path=output_dir / "anna_karenina_compound.png",
            window_size=500,
            smooth_sigma=3.0,
            annotations=anna_annotations
        )

        print("\nAnalysis Summary:")
        print(f"  Peak emotional altitude: {result['peak_position']:.1%} through narrative")
        print(f"  Lowest point: {result['valley_position']:.1%} through narrative")
        print(f"  Final compound: {result['final_compound']:.3f}")

        # Also do comparison with other Russian novels
        war_peace_path = Path(__file__).parent.parent / "data/raw/russian/texts/war_and_peace_ru.json"
        crime_path = Path(__file__).parent.parent / "data/raw/russian/texts/crime_and_punishment_ru.json"

        texts_to_compare = [("Anna Karenina", text)]

        if war_peace_path.exists():
            wp_text = load_russian_text(war_peace_path)
            texts_to_compare.append(("War and Peace", wp_text))

        if crime_path.exists():
            cp_text = load_russian_text(crime_path)
            texts_to_compare.append(("Crime and Punishment", cp_text))

        if len(texts_to_compare) > 1:
            print("\nCreating comparison plot...")
            plot_multiple_compounds(
                [(t, n) for n, t in texts_to_compare],
                output_path=output_dir / "russian_novels_comparison.png",
                smooth_sigma=5.0
            )
    else:
        print(f"Anna Karenina not found at {data_path}")
        print("Please download the corpus first.")
