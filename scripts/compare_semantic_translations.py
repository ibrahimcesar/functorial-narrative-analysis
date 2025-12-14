#!/usr/bin/env python3
"""
Compare Semantic/Plot Trajectories Between Original and Translation

This script extends translation comparison beyond sentiment to include:
- Character interactions (who interacts with whom)
- Narrative events (what happens)
- Plot state (tension, structure)

Usage:
    python scripts/compare_semantic_translations.py \
        --original data/raw/russian/texts/anna_karenina_ru.json \
        --translation data/raw/gutenberg/texts/anna_karenina.json \
        --output output/translations/
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import click
from rich.console import Console
from rich.table import Table
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

from src.functors.sentiment import SentimentFunctor
from src.functors.russian_sentiment import ClassicalRussianSentimentFunctor, create_windows_russian
from src.functors.semantic_tracking import (
    CharacterInteractionFunctor,
    RussianCharacterInteractionFunctor,
    NarrativeEventFunctor,
    RussianNarrativeEventFunctor,
    PlotStateFunctor,
    RussianPlotStateFunctor,
    compare_semantic_trajectories,
)
from src.functors.base import Trajectory

console = Console()


@dataclass
class SemanticComparisonResult:
    """Result of semantic trajectory comparison."""
    title: str
    original_language: str
    translation_language: str

    # Character interaction metrics
    interaction_correlation: float
    interaction_mad: float
    original_top_relationships: Dict[str, float]
    translation_top_relationships: Dict[str, float]

    # Narrative event metrics
    event_correlation: float
    event_mad: float
    event_peak_alignment: float
    original_event_types: Dict[str, int]
    translation_event_types: Dict[str, int]

    # Plot state metrics
    plot_correlation: float
    plot_mad: float
    original_climax_position: float
    translation_climax_position: float
    original_structure: str
    translation_structure: str

    # Overall divergence
    semantic_divergence: float


def create_windows_english(text: str, window_size: int = 1000, overlap: int = 500) -> List[str]:
    """Create overlapping windows from English text."""
    words = text.split()
    step = window_size - overlap
    windows = []
    for i in range(0, len(words), step):
        window = ' '.join(words[i:i + window_size])
        if len(window.split()) >= window_size // 2:
            windows.append(window)
    return windows if windows else [text]


def compare_semantic_dimensions(
    original_text: str,
    translation_text: str,
    original_language: str,
    translation_language: str,
    title: str,
    window_size: int = 1000,
) -> SemanticComparisonResult:
    """
    Compare original and translation across semantic dimensions.

    Args:
        original_text: Original text
        translation_text: Translation text
        original_language: Language code of original
        translation_language: Language code of translation
        title: Title of the work
        window_size: Window size for analysis

    Returns:
        SemanticComparisonResult with all metrics
    """
    # Create windows
    if original_language == 'ru':
        original_windows = create_windows_russian(original_text, window_size, window_size // 2)
    else:
        original_windows = create_windows_english(original_text, window_size, window_size // 2)

    translation_windows = create_windows_english(translation_text, window_size, window_size // 2)

    console.print(f"[blue]Original: {len(original_windows)} windows, Translation: {len(translation_windows)} windows[/blue]")

    # Initialize functors
    if original_language == 'ru':
        orig_interaction = RussianCharacterInteractionFunctor()
        orig_event = RussianNarrativeEventFunctor()
        orig_plot = RussianPlotStateFunctor()
    else:
        orig_interaction = CharacterInteractionFunctor()
        orig_event = NarrativeEventFunctor()
        orig_plot = PlotStateFunctor()

    trans_interaction = CharacterInteractionFunctor()
    trans_event = NarrativeEventFunctor()
    trans_plot = PlotStateFunctor()

    # Apply functors
    console.print("[cyan]Analyzing character interactions...[/cyan]")
    orig_interaction_traj = orig_interaction(original_windows)
    trans_interaction_traj = trans_interaction(translation_windows)

    console.print("[cyan]Analyzing narrative events...[/cyan]")
    orig_event_traj = orig_event(original_windows)
    trans_event_traj = trans_event(translation_windows)

    console.print("[cyan]Analyzing plot state...[/cyan]")
    orig_plot_traj = orig_plot(original_windows)
    trans_plot_traj = trans_plot(translation_windows)

    # Compare each dimension
    interaction_comparison = compare_semantic_trajectories(orig_interaction_traj, trans_interaction_traj)
    event_comparison = compare_semantic_trajectories(orig_event_traj, trans_event_traj)
    plot_comparison = compare_semantic_trajectories(orig_plot_traj, trans_plot_traj)

    # Calculate overall semantic divergence
    # Weight the three dimensions (events and plot matter more for "what happens")
    semantic_divergence = (
        0.2 * (1 - max(0, interaction_comparison["pearson_correlation"])) +
        0.4 * (1 - max(0, event_comparison["pearson_correlation"])) +
        0.4 * (1 - max(0, plot_comparison["pearson_correlation"]))
    )

    return SemanticComparisonResult(
        title=title,
        original_language=original_language,
        translation_language=translation_language,
        # Character interactions
        interaction_correlation=interaction_comparison["pearson_correlation"],
        interaction_mad=interaction_comparison["mean_absolute_difference"],
        original_top_relationships=orig_interaction_traj.metadata.get("top_relationships", {}),
        translation_top_relationships=trans_interaction_traj.metadata.get("top_relationships", {}),
        # Narrative events
        event_correlation=event_comparison["pearson_correlation"],
        event_mad=event_comparison["mean_absolute_difference"],
        event_peak_alignment=event_comparison["peak_alignment"],
        original_event_types=orig_event_traj.metadata.get("event_type_totals", {}),
        translation_event_types=trans_event_traj.metadata.get("event_type_totals", {}),
        # Plot state
        plot_correlation=plot_comparison["pearson_correlation"],
        plot_mad=plot_comparison["mean_absolute_difference"],
        original_climax_position=orig_plot_traj.metadata.get("climax_position", 0.5),
        translation_climax_position=trans_plot_traj.metadata.get("climax_position", 0.5),
        original_structure=orig_plot_traj.metadata.get("narrative_structure", "unknown"),
        translation_structure=trans_plot_traj.metadata.get("narrative_structure", "unknown"),
        # Overall
        semantic_divergence=semantic_divergence,
    )


def visualize_semantic_comparison(
    original_text: str,
    translation_text: str,
    original_language: str,
    translation_language: str,
    title: str,
    output_path: Path,
    window_size: int = 1000
):
    """Generate visualization comparing semantic trajectories."""
    # Create windows
    if original_language == 'ru':
        original_windows = create_windows_russian(original_text, window_size, window_size // 2)
    else:
        original_windows = create_windows_english(original_text, window_size, window_size // 2)

    translation_windows = create_windows_english(translation_text, window_size, window_size // 2)

    # Initialize functors
    if original_language == 'ru':
        orig_interaction = RussianCharacterInteractionFunctor()
        orig_event = RussianNarrativeEventFunctor()
        orig_plot = RussianPlotStateFunctor()
    else:
        orig_interaction = CharacterInteractionFunctor()
        orig_event = NarrativeEventFunctor()
        orig_plot = PlotStateFunctor()

    trans_interaction = CharacterInteractionFunctor()
    trans_event = NarrativeEventFunctor()
    trans_plot = PlotStateFunctor()

    # Apply functors and resample
    n_points = 100

    orig_int = orig_interaction(original_windows).resample(n_points)
    trans_int = trans_interaction(translation_windows).resample(n_points)

    orig_evt = orig_event(original_windows).resample(n_points)
    trans_evt = trans_event(translation_windows).resample(n_points)

    orig_plt = orig_plot(original_windows).resample(n_points)
    trans_plt = trans_plot(translation_windows).resample(n_points)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Semantic Comparison: {title}\n{original_language.upper()} → {translation_language.upper()}',
                 fontsize=14, fontweight='bold')

    time_points = np.linspace(0, 100, n_points)

    # Character Interactions
    ax1 = axes[0, 0]
    ax1.plot(time_points, orig_int.values, 'b-', linewidth=2, label=f'Original ({original_language.upper()})', alpha=0.8)
    ax1.plot(time_points, trans_int.values, 'r-', linewidth=2, label=f'Translation ({translation_language.upper()})', alpha=0.8)
    ax1.fill_between(time_points, orig_int.values, trans_int.values, alpha=0.2, color='purple')
    ax1.set_xlabel('Narrative Progress (%)')
    ax1.set_ylabel('Interaction Density')
    ax1.set_title('Character Interactions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Add correlation
    corr = np.corrcoef(orig_int.values, trans_int.values)[0, 1]
    ax1.text(0.02, 0.98, f'r = {corr:.3f}', transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Narrative Events
    ax2 = axes[0, 1]
    ax2.plot(time_points, orig_evt.values, 'b-', linewidth=2, label=f'Original ({original_language.upper()})', alpha=0.8)
    ax2.plot(time_points, trans_evt.values, 'r-', linewidth=2, label=f'Translation ({translation_language.upper()})', alpha=0.8)
    ax2.fill_between(time_points, orig_evt.values, trans_evt.values, alpha=0.2, color='purple')
    ax2.set_xlabel('Narrative Progress (%)')
    ax2.set_ylabel('Event Density')
    ax2.set_title('Narrative Events')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    corr = np.corrcoef(orig_evt.values, trans_evt.values)[0, 1]
    ax2.text(0.02, 0.98, f'r = {corr:.3f}', transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot State (Tension)
    ax3 = axes[1, 0]
    ax3.plot(time_points, orig_plt.values, 'b-', linewidth=2, label=f'Original ({original_language.upper()})', alpha=0.8)
    ax3.plot(time_points, trans_plt.values, 'r-', linewidth=2, label=f'Translation ({translation_language.upper()})', alpha=0.8)
    ax3.fill_between(time_points, orig_plt.values, trans_plt.values, alpha=0.2, color='purple')
    ax3.set_xlabel('Narrative Progress (%)')
    ax3.set_ylabel('Tension Level')
    ax3.set_title('Plot State (Tension)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    corr = np.corrcoef(orig_plt.values, trans_plt.values)[0, 1]
    ax3.text(0.02, 0.98, f'r = {corr:.3f}', transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Combined divergence
    ax4 = axes[1, 1]
    int_div = np.abs(orig_int.values - trans_int.values)
    evt_div = np.abs(orig_evt.values - trans_evt.values)
    plt_div = np.abs(orig_plt.values - trans_plt.values)

    ax4.stackplot(time_points, int_div, evt_div, plt_div,
                  labels=['Interaction', 'Events', 'Plot'],
                  colors=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax4.set_xlabel('Narrative Progress (%)')
    ax4.set_ylabel('Divergence')
    ax4.set_title('Semantic Divergence by Dimension')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    console.print(f"[green]✓ Saved visualization: {output_path}[/green]")


def print_comparison_report(result: SemanticComparisonResult):
    """Print a formatted comparison report."""
    console.print("\n" + "=" * 70)
    console.print(f"[bold]SEMANTIC COMPARISON REPORT: {result.title}[/bold]")
    console.print(f"Original: {result.original_language.upper()} → Translation: {result.translation_language.upper()}")
    console.print("=" * 70)

    # Summary table
    table = Table(title="Semantic Trajectory Correlations")
    table.add_column("Dimension", style="cyan")
    table.add_column("Correlation", style="green")
    table.add_column("MAD", style="yellow")
    table.add_column("Match", style="magenta")

    # Character interactions
    int_match = "✓" if result.interaction_correlation > 0.5 else "✗"
    table.add_row(
        "Character Interactions",
        f"{result.interaction_correlation:.3f}",
        f"{result.interaction_mad:.3f}",
        int_match
    )

    # Narrative events
    evt_match = "✓" if result.event_correlation > 0.5 else "✗"
    table.add_row(
        "Narrative Events",
        f"{result.event_correlation:.3f}",
        f"{result.event_mad:.3f}",
        evt_match
    )

    # Plot state
    plt_match = "✓" if result.plot_correlation > 0.5 else "✗"
    table.add_row(
        "Plot State (Tension)",
        f"{result.plot_correlation:.3f}",
        f"{result.plot_mad:.3f}",
        plt_match
    )

    console.print(table)

    # Climax position comparison
    console.print("\n[bold]Climax Position:[/bold]")
    console.print(f"  Original:    {result.original_climax_position*100:.1f}% through narrative")
    console.print(f"  Translation: {result.translation_climax_position*100:.1f}% through narrative")
    climax_diff = abs(result.original_climax_position - result.translation_climax_position)
    console.print(f"  Difference:  {climax_diff*100:.1f}%")

    # Structure comparison
    console.print("\n[bold]Narrative Structure:[/bold]")
    console.print(f"  Original:    {result.original_structure}")
    console.print(f"  Translation: {result.translation_structure}")
    structure_match = result.original_structure == result.translation_structure
    console.print(f"  Match:       {'✓' if structure_match else '✗'}")

    # Top relationships
    if result.original_top_relationships:
        console.print("\n[bold]Top Character Relationships (Original):[/bold]")
        for pair, strength in list(result.original_top_relationships.items())[:5]:
            console.print(f"  {pair}: {strength:.1f}")

    if result.translation_top_relationships:
        console.print("\n[bold]Top Character Relationships (Translation):[/bold]")
        for pair, strength in list(result.translation_top_relationships.items())[:5]:
            console.print(f"  {pair}: {strength:.1f}")

    # Event type comparison
    console.print("\n[bold]Event Types (Original vs Translation):[/bold]")
    all_types = set(result.original_event_types.keys()) | set(result.translation_event_types.keys())
    for etype in sorted(all_types):
        orig_count = result.original_event_types.get(etype, 0)
        trans_count = result.translation_event_types.get(etype, 0)
        console.print(f"  {etype}: {orig_count} vs {trans_count}")

    # Overall divergence
    console.print("\n" + "-" * 70)
    console.print(f"[bold]Overall Semantic Divergence: {result.semantic_divergence:.3f}[/bold]")
    if result.semantic_divergence < 0.3:
        console.print("[green]Interpretation: High semantic preservation - plot structure well maintained[/green]")
    elif result.semantic_divergence < 0.5:
        console.print("[yellow]Interpretation: Moderate divergence - some plot/event differences[/yellow]")
    else:
        console.print("[red]Interpretation: High divergence - significant plot structure changes[/red]")


@click.command()
@click.option('--original', '-o', required=True, type=click.Path(exists=True),
              help='Path to original text JSON')
@click.option('--translation', '-t', required=True, type=click.Path(exists=True),
              help='Path to translation text JSON')
@click.option('--output', '-O', default='output/translations/',
              help='Output directory for visualizations')
@click.option('--window-size', '-w', default=1000, help='Window size in words')
def main(original: str, translation: str, output: str, window_size: int):
    """
    Compare semantic trajectories between original and translation.

    Goes beyond sentiment to compare what actually happens in the narrative:
    character interactions, events, and plot structure.
    """
    # Load texts
    with open(original, 'r', encoding='utf-8') as f:
        orig_data = json.load(f)

    with open(translation, 'r', encoding='utf-8') as f:
        trans_data = json.load(f)

    original_text = orig_data.get('text', '')
    translation_text = trans_data.get('text', '')

    # Detect languages
    original_language = orig_data.get('language', 'ru')
    translation_language = trans_data.get('language', 'en')

    title = orig_data.get('title_en', orig_data.get('title', 'Unknown'))

    console.print("=" * 70)
    console.print(f"[bold]SEMANTIC TRANSLATION COMPARISON[/bold]")
    console.print(f"Title: {title}")
    console.print(f"Original: {len(original_text):,} chars ({original_language})")
    console.print(f"Translation: {len(translation_text):,} chars ({translation_language})")
    console.print("=" * 70)

    # Run comparison
    result = compare_semantic_dimensions(
        original_text=original_text,
        translation_text=translation_text,
        original_language=original_language,
        translation_language=translation_language,
        title=title,
        window_size=window_size,
    )

    # Print report
    print_comparison_report(result)

    # Generate visualization
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    orig_id = Path(original).stem
    trans_id = Path(translation).stem
    viz_path = output_dir / f"{orig_id}_vs_{trans_id}_semantic.png"

    visualize_semantic_comparison(
        original_text=original_text,
        translation_text=translation_text,
        original_language=original_language,
        translation_language=translation_language,
        title=title,
        output_path=viz_path,
        window_size=window_size,
    )

    # Save results JSON
    results_path = output_dir / f"{orig_id}_vs_{trans_id}_semantic.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "title": result.title,
            "original_language": result.original_language,
            "translation_language": result.translation_language,
            "interaction_correlation": result.interaction_correlation,
            "event_correlation": result.event_correlation,
            "plot_correlation": result.plot_correlation,
            "semantic_divergence": result.semantic_divergence,
            "original_climax_position": result.original_climax_position,
            "translation_climax_position": result.translation_climax_position,
            "original_structure": result.original_structure,
            "translation_structure": result.translation_structure,
        }, f, indent=2)
    console.print(f"[green]✓ Saved results: {results_path}[/green]")


if __name__ == "__main__":
    main()
