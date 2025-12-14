#!/usr/bin/env python3
"""
Cross-Linguistic Translation Comparison

Compares narrative trajectories between original texts and their translations
to measure how emotional arcs, pacing, and other features are preserved or
diverge in translation.

This is a novel application of functorial narrative analysis to translation studies.

Usage:
    python scripts/compare_translations.py --original anna_karenina_ru --translation anna_karenina
    python scripts/compare_translations.py --original-file data/raw/russian/texts/anna_karenina_ru.json \
                                           --translation-file data/raw/tolstoy/anna_karenina.json
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
import re

import numpy as np
import click
from rich.console import Console
from rich.table import Table
from scipy.stats import pearsonr, spearmanr
from scipy.signal import correlate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Import functors
from src.functors.sentiment import SentimentFunctor
from src.functors.russian_sentiment import ClassicalRussianSentimentFunctor, create_windows_russian
from src.functors.entropy import EntropyFunctor
from src.functors.pacing import PacingFunctor
from src.detectors.icc import ICCDetector

console = Console()


@dataclass
class TranslationDivergence:
    """Measures of divergence between original and translation."""
    # Correlation measures
    pearson_correlation: float
    spearman_correlation: float

    # Distance measures
    mean_absolute_difference: float
    root_mean_squared_difference: float
    max_divergence: float
    max_divergence_position: float  # Position in [0,1] where max divergence occurs

    # Phase shift (translation may lag/lead)
    optimal_lag: int  # Positive = translation lags, negative = leads
    lag_correlation: float  # Correlation at optimal lag

    # Arc preservation
    arc_match: bool  # Do overall arcs match?
    original_arc: str
    translation_arc: str

    # Section-wise divergence
    divergence_by_quarter: List[float] = field(default_factory=list)


@dataclass
class TranslationComparisonResult:
    """Complete translation comparison results."""
    original_id: str
    translation_id: str
    original_language: str
    translation_language: str
    title: str

    # Functor-wise divergence
    sentiment_divergence: TranslationDivergence
    entropy_divergence: TranslationDivergence

    # Overall score (0 = identical, 1 = completely different)
    overall_divergence_score: float

    # ICC results
    original_icc: str
    translation_icc: str
    icc_match: bool

    # Key divergence points
    divergence_peaks: List[Dict]  # [{position, original_value, translation_value, magnitude}]


def create_windows_english(text: str, window_size: int = 1000, overlap: int = 500) -> List[str]:
    """Create overlapping windows from English text."""
    words = text.split()
    step = window_size - overlap
    windows = []

    for i in range(0, len(words), step):
        window_words = words[i:i + window_size]
        if len(window_words) >= window_size // 2:
            windows.append(' '.join(window_words))

    return windows if windows else [text]


def normalize_trajectory(values: np.ndarray) -> np.ndarray:
    """Normalize trajectory to [0, 1] range."""
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val - min_val == 0:
        return np.zeros_like(values)
    return (values - min_val) / (max_val - min_val)


def resample_trajectory(values: np.ndarray, target_length: int) -> np.ndarray:
    """Resample trajectory to a target length using interpolation."""
    if len(values) == target_length:
        return values

    x_original = np.linspace(0, 1, len(values))
    x_target = np.linspace(0, 1, target_length)

    interpolator = interp1d(x_original, values, kind='linear', fill_value='extrapolate')
    return interpolator(x_target)


def classify_arc(values: np.ndarray) -> str:
    """Classify the emotional arc shape."""
    n = len(values)
    if n < 4:
        return "Unknown"

    first_quarter = np.mean(values[:n//4])
    mid = np.mean(values[n//4:3*n//4])
    last_quarter = np.mean(values[3*n//4:])

    if first_quarter < mid and mid > last_quarter:
        return "Rise-Fall"
    elif first_quarter > mid and mid < last_quarter:
        return "Fall-Rise"
    elif first_quarter < last_quarter and mid < last_quarter:
        return "Rising"
    elif first_quarter > last_quarter and mid > first_quarter:
        return "Falling"
    else:
        return "Stable"


def compute_optimal_lag(signal1: np.ndarray, signal2: np.ndarray, max_lag: int = 10) -> Tuple[int, float]:
    """
    Find optimal lag between two signals using cross-correlation.

    Returns:
        (optimal_lag, correlation_at_lag)
    """
    # Normalize signals
    signal1 = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-8)
    signal2 = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-8)

    best_lag = 0
    best_corr = np.corrcoef(signal1, signal2)[0, 1]

    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            continue
        if lag > 0:
            s1 = signal1[lag:]
            s2 = signal2[:-lag]
        else:
            s1 = signal1[:lag]
            s2 = signal2[-lag:]

        if len(s1) < 3:
            continue

        corr = np.corrcoef(s1, s2)[0, 1]
        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag

    return best_lag, best_corr


def compute_divergence(original: np.ndarray, translation: np.ndarray) -> TranslationDivergence:
    """Compute divergence metrics between two trajectories."""
    # Ensure same length
    target_len = min(len(original), len(translation))
    if target_len < 10:
        target_len = max(len(original), len(translation))

    orig_resampled = resample_trajectory(original, target_len)
    trans_resampled = resample_trajectory(translation, target_len)

    # Correlation measures
    pearson_r, _ = pearsonr(orig_resampled, trans_resampled)
    spearman_r, _ = spearmanr(orig_resampled, trans_resampled)

    # Distance measures
    differences = np.abs(orig_resampled - trans_resampled)
    mad = float(np.mean(differences))
    rmsd = float(np.sqrt(np.mean(differences ** 2)))
    max_div = float(np.max(differences))
    max_div_pos = float(np.argmax(differences) / len(differences))

    # Phase shift
    optimal_lag, lag_corr = compute_optimal_lag(orig_resampled, trans_resampled)

    # Arc classification
    orig_arc = classify_arc(orig_resampled)
    trans_arc = classify_arc(trans_resampled)

    # Divergence by quarter
    quarter_len = target_len // 4
    div_by_quarter = []
    for i in range(4):
        start = i * quarter_len
        end = start + quarter_len if i < 3 else target_len
        quarter_mad = float(np.mean(np.abs(orig_resampled[start:end] - trans_resampled[start:end])))
        div_by_quarter.append(quarter_mad)

    return TranslationDivergence(
        pearson_correlation=float(pearson_r) if not np.isnan(pearson_r) else 0.0,
        spearman_correlation=float(spearman_r) if not np.isnan(spearman_r) else 0.0,
        mean_absolute_difference=mad,
        root_mean_squared_difference=rmsd,
        max_divergence=max_div,
        max_divergence_position=max_div_pos,
        optimal_lag=optimal_lag,
        lag_correlation=float(lag_corr) if not np.isnan(lag_corr) else 0.0,
        arc_match=(orig_arc == trans_arc),
        original_arc=orig_arc,
        translation_arc=trans_arc,
        divergence_by_quarter=div_by_quarter
    )


def find_divergence_peaks(
    original: np.ndarray,
    translation: np.ndarray,
    threshold: float = 0.3,
    min_distance: int = 5
) -> List[Dict]:
    """Find significant divergence points."""
    # Ensure same length
    target_len = min(len(original), len(translation))
    orig = resample_trajectory(original, target_len)
    trans = resample_trajectory(translation, target_len)

    differences = np.abs(orig - trans)

    peaks = []
    last_peak = -min_distance

    for i in range(len(differences)):
        if differences[i] > threshold and i - last_peak >= min_distance:
            # Check if local maximum
            is_peak = True
            for j in range(max(0, i-2), min(len(differences), i+3)):
                if j != i and differences[j] > differences[i]:
                    is_peak = False
                    break

            if is_peak:
                peaks.append({
                    "position": float(i / len(differences)),
                    "original_value": float(orig[i]),
                    "translation_value": float(trans[i]),
                    "magnitude": float(differences[i])
                })
                last_peak = i

    return sorted(peaks, key=lambda x: -x["magnitude"])[:10]  # Top 10 peaks


def compare_translations(
    original_text: str,
    translation_text: str,
    original_language: str,
    translation_language: str,
    original_id: str,
    translation_id: str,
    title: str,
    window_size: int = 1000
) -> TranslationComparisonResult:
    """
    Compare original text with its translation.

    Args:
        original_text: Text in original language
        translation_text: Translated text
        original_language: Language code of original (e.g., 'ru', 'zh')
        translation_language: Language code of translation (e.g., 'en')
        original_id: ID of original text
        translation_id: ID of translation
        title: Title of work
        window_size: Window size for analysis

    Returns:
        TranslationComparisonResult with all divergence metrics
    """
    # Create windows based on language
    if original_language == 'ru':
        original_windows = create_windows_russian(original_text, window_size, window_size // 2)
        original_sentiment_functor = ClassicalRussianSentimentFunctor()
    else:
        original_windows = create_windows_english(original_text, window_size, window_size // 2)
        original_sentiment_functor = SentimentFunctor()

    if translation_language == 'en':
        translation_windows = create_windows_english(translation_text, window_size, window_size // 2)
        translation_sentiment_functor = SentimentFunctor()
    else:
        translation_windows = create_windows_english(translation_text, window_size, window_size // 2)
        translation_sentiment_functor = SentimentFunctor()

    console.print(f"[dim]Original windows: {len(original_windows)}, Translation windows: {len(translation_windows)}[/dim]")

    # Apply sentiment functors
    original_sentiment = original_sentiment_functor(original_windows)
    translation_sentiment = translation_sentiment_functor(translation_windows)

    # Apply entropy functors
    entropy_functor = EntropyFunctor(method="combined")
    original_entropy = entropy_functor(original_windows)
    translation_entropy = entropy_functor(translation_windows)

    # Compute divergence
    sentiment_div = compute_divergence(original_sentiment.values, translation_sentiment.values)
    entropy_div = compute_divergence(original_entropy.values, translation_entropy.values)

    # Find divergence peaks
    div_peaks = find_divergence_peaks(original_sentiment.values, translation_sentiment.values)

    # ICC classification
    icc_detector = ICCDetector()
    original_icc = icc_detector.detect(original_sentiment.values)
    translation_icc = icc_detector.detect(translation_sentiment.values)

    # Overall divergence score (0-1, lower is better)
    overall_score = 1.0 - (
        0.4 * max(sentiment_div.pearson_correlation, 0) +
        0.3 * max(entropy_div.pearson_correlation, 0) +
        0.2 * (1 if sentiment_div.arc_match else 0) +
        0.1 * (1 if original_icc.icc_class == translation_icc.icc_class else 0)
    )

    return TranslationComparisonResult(
        original_id=original_id,
        translation_id=translation_id,
        original_language=original_language,
        translation_language=translation_language,
        title=title,
        sentiment_divergence=sentiment_div,
        entropy_divergence=entropy_div,
        overall_divergence_score=overall_score,
        original_icc=original_icc.icc_class,
        translation_icc=translation_icc.icc_class,
        icc_match=(original_icc.icc_class == translation_icc.icc_class),
        divergence_peaks=div_peaks
    )


def plot_comparison(
    original_text: str,
    translation_text: str,
    original_language: str,
    translation_language: str,
    title: str,
    output_path: Path,
    window_size: int = 1000
):
    """Generate comparison visualization."""
    # Create windows based on language
    if original_language == 'ru':
        original_windows = create_windows_russian(original_text, window_size, window_size // 2)
        original_sentiment_functor = ClassicalRussianSentimentFunctor()
    else:
        original_windows = create_windows_english(original_text, window_size, window_size // 2)
        original_sentiment_functor = SentimentFunctor()

    translation_windows = create_windows_english(translation_text, window_size, window_size // 2)
    translation_sentiment_functor = SentimentFunctor()

    # Apply functors
    original_sentiment = original_sentiment_functor(original_windows)
    translation_sentiment = translation_sentiment_functor(translation_windows)

    entropy_functor = EntropyFunctor(method="combined")
    original_entropy = entropy_functor(original_windows)
    translation_entropy = entropy_functor(translation_windows)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Translation Comparison: {title}\n{original_language.upper()} Original vs {translation_language.upper()} Translation',
                 fontsize=14, fontweight='bold')

    # Sentiment trajectories
    ax1 = axes[0, 0]
    x_orig = np.linspace(0, 100, len(original_sentiment.values))
    x_trans = np.linspace(0, 100, len(translation_sentiment.values))
    ax1.plot(x_orig, original_sentiment.values, 'b-', label=f'Original ({original_language.upper()})', alpha=0.8)
    ax1.plot(x_trans, translation_sentiment.values, 'r--', label=f'Translation ({translation_language.upper()})', alpha=0.8)
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Narrative Progress (%)')
    ax1.set_ylabel('Sentiment Score')
    ax1.set_title('Sentiment Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Entropy trajectories
    ax2 = axes[0, 1]
    x_orig = np.linspace(0, 100, len(original_entropy.values))
    x_trans = np.linspace(0, 100, len(translation_entropy.values))
    ax2.plot(x_orig, original_entropy.values, 'b-', label=f'Original ({original_language.upper()})', alpha=0.8)
    ax2.plot(x_trans, translation_entropy.values, 'r--', label=f'Translation ({translation_language.upper()})', alpha=0.8)
    ax2.set_xlabel('Narrative Progress (%)')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Entropy Trajectories')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Divergence plot
    ax3 = axes[1, 0]
    target_len = min(len(original_sentiment.values), len(translation_sentiment.values))
    orig_resampled = resample_trajectory(original_sentiment.values, target_len)
    trans_resampled = resample_trajectory(translation_sentiment.values, target_len)
    divergence = np.abs(orig_resampled - trans_resampled)
    x_div = np.linspace(0, 100, len(divergence))
    ax3.fill_between(x_div, 0, divergence, alpha=0.6, color='purple')
    ax3.plot(x_div, divergence, 'purple', linewidth=1)
    ax3.set_xlabel('Narrative Progress (%)')
    ax3.set_ylabel('Absolute Divergence')
    ax3.set_title('Sentiment Divergence (|Original - Translation|)')
    ax3.grid(True, alpha=0.3)

    # Scatter plot with regression
    ax4 = axes[1, 1]
    ax4.scatter(orig_resampled, trans_resampled, alpha=0.5, s=30)

    # Add regression line
    z = np.polyfit(orig_resampled, trans_resampled, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(orig_resampled), max(orig_resampled), 100)
    ax4.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Regression (r={np.corrcoef(orig_resampled, trans_resampled)[0,1]:.3f})')

    # Perfect correlation line
    ax4.plot([-1, 1], [-1, 1], 'k--', alpha=0.5, label='Perfect match')
    ax4.set_xlabel(f'Original Sentiment ({original_language.upper()})')
    ax4.set_ylabel(f'Translation Sentiment ({translation_language.upper()})')
    ax4.set_title('Sentiment Correlation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    console.print(f"[green]✓ Saved visualization: {output_path}[/green]")


def display_results(result: TranslationComparisonResult):
    """Display comparison results in formatted table."""
    console.print("\n" + "=" * 70)
    console.print(f"[bold]TRANSLATION COMPARISON: {result.title}[/bold]")
    console.print(f"Original ({result.original_language.upper()}) vs Translation ({result.translation_language.upper()})")
    console.print("=" * 70)

    # Summary metrics
    table = Table(title="Divergence Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Sentiment", justify="right")
    table.add_column("Entropy", justify="right")

    sd = result.sentiment_divergence
    ed = result.entropy_divergence

    table.add_row("Pearson Correlation", f"{sd.pearson_correlation:.3f}", f"{ed.pearson_correlation:.3f}")
    table.add_row("Spearman Correlation", f"{sd.spearman_correlation:.3f}", f"{ed.spearman_correlation:.3f}")
    table.add_row("Mean Abs Difference", f"{sd.mean_absolute_difference:.3f}", f"{ed.mean_absolute_difference:.3f}")
    table.add_row("Max Divergence", f"{sd.max_divergence:.3f}", f"{ed.max_divergence:.3f}")
    table.add_row("Max Div Position", f"{sd.max_divergence_position*100:.1f}%", f"{ed.max_divergence_position*100:.1f}%")
    table.add_row("Optimal Lag", f"{sd.optimal_lag}", f"{ed.optimal_lag}")
    table.add_row("Original Arc", sd.original_arc, ed.original_arc)
    table.add_row("Translation Arc", sd.translation_arc, ed.translation_arc)
    table.add_row("Arc Match", "✓" if sd.arc_match else "✗", "✓" if ed.arc_match else "✗")

    console.print(table)

    # ICC comparison
    console.print(f"\n[bold]ICC Classification:[/bold]")
    console.print(f"  Original: {result.original_icc}")
    console.print(f"  Translation: {result.translation_icc}")
    console.print(f"  Match: {'✓ Yes' if result.icc_match else '✗ No'}")

    # Overall score
    score_color = "green" if result.overall_divergence_score < 0.3 else "yellow" if result.overall_divergence_score < 0.5 else "red"
    console.print(f"\n[bold]Overall Divergence Score:[/bold] [{score_color}]{result.overall_divergence_score:.3f}[/{score_color}]")
    console.print(f"  (0 = identical trajectories, 1 = completely different)")

    # Interpretation
    console.print(f"\n[bold]Interpretation:[/bold]")
    if result.overall_divergence_score < 0.25:
        console.print("  [green]The translation preserves the narrative dynamics well.[/green]")
    elif result.overall_divergence_score < 0.4:
        console.print("  [yellow]The translation shows moderate divergence from original.[/yellow]")
    else:
        console.print("  [red]Significant divergence between original and translation.[/red]")

    # Divergence by section
    console.print(f"\n[bold]Sentiment Divergence by Narrative Section:[/bold]")
    sections = ["Beginning", "Rising Action", "Climax", "Resolution"]
    for i, (section, div) in enumerate(zip(sections, sd.divergence_by_quarter)):
        bar = "█" * int(div * 20)
        console.print(f"  {section:15} {bar:20} {div:.3f}")

    # Key divergence points
    if result.divergence_peaks:
        console.print(f"\n[bold]Key Divergence Points:[/bold]")
        for i, peak in enumerate(result.divergence_peaks[:5]):
            console.print(f"  {i+1}. At {peak['position']*100:.1f}%: Original={peak['original_value']:.2f}, "
                         f"Translation={peak['translation_value']:.2f} (Δ={peak['magnitude']:.2f})")


@click.command()
@click.option('--original-file', '-o', type=click.Path(exists=True),
              help='Path to original text JSON file')
@click.option('--translation-file', '-t', type=click.Path(exists=True),
              help='Path to translation text JSON file')
@click.option('--original-lang', default='ru', help='Original language code')
@click.option('--translation-lang', default='en', help='Translation language code')
@click.option('--window-size', '-w', default=1000, help='Window size for analysis')
@click.option('--output-dir', default='output/translations', help='Output directory')
@click.option('--visualize/--no-visualize', default=True, help='Generate visualization')
def main(
    original_file: str,
    translation_file: str,
    original_lang: str,
    translation_lang: str,
    window_size: int,
    output_dir: str,
    visualize: bool
):
    """
    Compare original text with its translation.

    Analyzes how narrative trajectories (sentiment, entropy, etc.)
    are preserved or diverge in translation.
    """
    console.print("=" * 70)
    console.print("[bold]CROSS-LINGUISTIC TRANSLATION COMPARISON[/bold]")
    console.print("=" * 70)

    # Load texts
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    with open(translation_file, 'r', encoding='utf-8') as f:
        translation_data = json.load(f)

    original_text = original_data.get('text', '')
    translation_text = translation_data.get('text', '')
    title = original_data.get('title_en', original_data.get('title', 'Unknown'))

    console.print(f"Original: {original_data.get('title', 'Unknown')} ({original_lang})")
    console.print(f"Translation: {translation_data.get('title', 'Unknown')} ({translation_lang})")
    console.print(f"Original length: {len(original_text):,} chars")
    console.print(f"Translation length: {len(translation_text):,} chars")

    # Run comparison
    result = compare_translations(
        original_text=original_text,
        translation_text=translation_text,
        original_language=original_lang,
        translation_language=translation_lang,
        original_id=original_data.get('id', 'original'),
        translation_id=translation_data.get('id', 'translation'),
        title=title,
        window_size=window_size
    )

    # Display results
    display_results(result)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = output_path / f"{result.original_id}_vs_{result.translation_id}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(asdict(result), f, indent=2)
    console.print(f"\n[green]✓ Results saved: {result_file}[/green]")

    # Generate visualization
    if visualize:
        viz_file = output_path / f"{result.original_id}_vs_{result.translation_id}.png"
        plot_comparison(
            original_text=original_text,
            translation_text=translation_text,
            original_language=original_lang,
            translation_language=translation_lang,
            title=title,
            output_path=viz_file,
            window_size=window_size
        )


if __name__ == "__main__":
    main()
