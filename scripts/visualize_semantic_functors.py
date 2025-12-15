#!/usr/bin/env python3
"""
Visualize Semantic/Plot Tracking Functors

Generates rich visualizations for:
1. Character Interactions - relationship graphs and interaction density
2. Narrative Events - event type distribution and plot point detection
3. Plot State - tension curves and narrative structure analysis

Usage:
    python scripts/visualize_semantic_functors.py \
        --input data/raw/tolstoy/anna_karenina.json \
        --output output/visualizations/ \
        --language en
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import click
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from src.functors.semantic_tracking import (
    CharacterInteractionFunctor,
    RussianCharacterInteractionFunctor,
    NarrativeEventFunctor,
    RussianNarrativeEventFunctor,
    PlotStateFunctor,
    RussianPlotStateFunctor,
    EventType,
    PlotStateType,
)
# Note: create_windows_russian lowercases text, breaking character name extraction
# We use create_windows_preserve_case instead for semantic analysis

console = Console()


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


def create_windows_preserve_case(text: str, window_size: int = 1000, overlap: int = 500) -> List[str]:
    """Create overlapping windows preserving original case (for character extraction)."""
    words = text.split()
    step = window_size - overlap
    windows = []
    for i in range(0, len(words), step):
        window = ' '.join(words[i:i + window_size])
        if len(window.split()) >= window_size // 2:
            windows.append(window)
    return windows if windows else [text]


def visualize_character_interactions(
    text: str,
    language: str,
    title: str,
    output_path: Path,
    window_size: int = 1000
):
    """
    Visualize character interactions over the narrative.

    Creates:
    - Interaction density over time
    - Top character relationships heatmap
    - Relationship strength evolution
    """
    # Create windows - MUST preserve case for character name extraction
    # (the Russian sentiment windows lowercase the text, breaking name detection)
    windows = create_windows_preserve_case(text, window_size, window_size // 2)

    if language == 'ru':
        functor = RussianCharacterInteractionFunctor()
    else:
        functor = CharacterInteractionFunctor()

    console.print(f"[cyan]Processing {len(windows)} windows for character interactions...[/cyan]")

    # Apply functor
    trajectory = functor(windows)

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Character Interaction Analysis: {title}', fontsize=16, fontweight='bold')

    # 1. Interaction density over time
    ax1 = fig.add_subplot(2, 2, 1)
    time_points = np.linspace(0, 100, len(trajectory.values))
    ax1.fill_between(time_points, trajectory.values, alpha=0.3, color='blue')
    ax1.plot(time_points, trajectory.values, 'b-', linewidth=2)
    ax1.set_xlabel('Narrative Progress (%)')
    ax1.set_ylabel('Interaction Density')
    ax1.set_title('Character Interaction Intensity Over Time')
    ax1.grid(True, alpha=0.3)

    # Add smoothed trendline
    if len(trajectory.values) > 10:
        window = min(15, len(trajectory.values) // 3)
        smoothed = np.convolve(trajectory.values, np.ones(window)/window, mode='same')
        ax1.plot(time_points, smoothed, 'r--', linewidth=2, alpha=0.7, label='Trend')
        ax1.legend()

    # 2. Top relationships bar chart
    ax2 = fig.add_subplot(2, 2, 2)
    top_rels = trajectory.metadata.get('top_relationships', {})
    if top_rels:
        pairs = list(top_rels.keys())[:10]
        strengths = [top_rels[p] for p in pairs]

        # Shorten labels if needed
        short_pairs = [p[:20] + '...' if len(p) > 20 else p for p in pairs]

        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(pairs)))
        bars = ax2.barh(range(len(pairs)), strengths, color=colors)
        ax2.set_yticks(range(len(pairs)))
        ax2.set_yticklabels(short_pairs, fontsize=9)
        ax2.set_xlabel('Interaction Strength')
        ax2.set_title('Top Character Relationships')
        ax2.invert_yaxis()

        # Add value labels
        for bar, val in zip(bars, strengths):
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', va='center', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No relationships detected', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12, color='gray')
        ax2.set_title('Top Character Relationships')

    # 3. Interaction heatmap over narrative sections
    ax3 = fig.add_subplot(2, 2, 3)
    n_sections = 10
    section_size = len(trajectory.values) // n_sections
    section_means = []
    for i in range(n_sections):
        start = i * section_size
        end = start + section_size if i < n_sections - 1 else len(trajectory.values)
        section_means.append(np.mean(trajectory.values[start:end]))

    # Create heatmap-style bar
    colors = plt.cm.RdYlGn(np.array(section_means) / max(section_means) if max(section_means) > 0 else section_means)
    bars = ax3.bar(range(n_sections), section_means, color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_xticks(range(n_sections))
    ax3.set_xticklabels([f'{i*10}-{(i+1)*10}%' for i in range(n_sections)], rotation=45, ha='right')
    ax3.set_ylabel('Mean Interaction Density')
    ax3.set_title('Interaction Intensity by Narrative Section')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Statistics summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    stats_text = f"""
    Character Interaction Statistics
    ════════════════════════════════

    Total Unique Characters: {trajectory.metadata.get('total_characters', 0)}

    Mean Interaction Density: {trajectory.metadata.get('mean_interaction', 0):.3f}
    Interaction Variance: {trajectory.metadata.get('interaction_variance', 0):.4f}

    Peak Interaction at: {time_points[np.argmax(trajectory.values)]:.1f}%
    Minimum Interaction at: {time_points[np.argmin(trajectory.values)]:.1f}%

    Language: {language.upper()}
    Windows Analyzed: {len(windows)}
    """

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    console.print(f"[green]✓ Saved character interaction visualization: {output_path}[/green]")
    return trajectory


def visualize_narrative_events(
    text: str,
    language: str,
    title: str,
    output_path: Path,
    window_size: int = 1000
):
    """
    Visualize narrative events over the story.

    Creates:
    - Event density curve
    - Event type distribution
    - Major plot points timeline
    """
    # Create windows
    if language == 'ru':
        windows = create_windows_preserve_case(text, window_size, window_size // 2)
        functor = RussianNarrativeEventFunctor()
    else:
        windows = create_windows_english(text, window_size, window_size // 2)
        functor = NarrativeEventFunctor()

    console.print(f"[cyan]Processing {len(windows)} windows for narrative events...[/cyan]")

    # Apply functor
    trajectory = functor(windows)

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Narrative Event Analysis: {title}', fontsize=16, fontweight='bold')

    # 1. Event density over time with major events highlighted
    ax1 = fig.add_subplot(2, 2, 1)
    time_points = np.linspace(0, 100, len(trajectory.values))

    ax1.fill_between(time_points, trajectory.values, alpha=0.3, color='orange')
    ax1.plot(time_points, trajectory.values, 'darkorange', linewidth=2, label='Event Density')

    # Mark major plot points
    major_points = trajectory.metadata.get('major_plot_points', [])
    for point in major_points[:5]:  # Top 5
        pos = point['position'] * 100
        intensity = point['intensity']
        ax1.axvline(x=pos, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax1.annotate(f'Major Event\n({intensity:.2f})', xy=(pos, intensity),
                    xytext=(pos+3, intensity+0.05), fontsize=8,
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    ax1.set_xlabel('Narrative Progress (%)')
    ax1.set_ylabel('Event Density')
    ax1.set_title('Narrative Event Intensity (Major Events Marked)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Event type distribution (pie chart)
    ax2 = fig.add_subplot(2, 2, 2)
    event_types = trajectory.metadata.get('event_type_totals', {})
    if event_types:
        labels = list(event_types.keys())
        sizes = list(event_types.values())

        # Color map for event types
        event_colors = {
            'movement': '#3498db',
            'death': '#2c3e50',
            'marriage': '#e91e63',
            'conflict': '#e74c3c',
            'revelation': '#9b59b6',
            'emotional': '#f39c12',
        }
        colors = [event_colors.get(l, '#95a5a6') for l in labels]

        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90,
                                           explode=[0.02]*len(labels))
        ax2.set_title('Event Type Distribution')

        # Make labels readable
        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontsize(8)
    else:
        ax2.text(0.5, 0.5, 'No events detected', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12, color='gray')
        ax2.set_title('Event Type Distribution')

    # 3. Event type timeline (stacked area)
    ax3 = fig.add_subplot(2, 2, 3)

    # Reconstruct per-window event types (simplified)
    n_sections = 20
    section_size = len(trajectory.values) // n_sections

    # Create stacked data based on overall proportions applied to density
    if event_types:
        total_events = sum(event_types.values())
        proportions = {k: v/total_events for k, v in event_types.items()}

        section_time = np.linspace(0, 100, n_sections)
        section_densities = []
        for i in range(n_sections):
            start = i * section_size
            end = start + section_size if i < n_sections - 1 else len(trajectory.values)
            section_densities.append(np.mean(trajectory.values[start:end]))

        # Create stacked data
        stacked_data = {}
        for etype, prop in proportions.items():
            stacked_data[etype] = [d * prop for d in section_densities]

        # Plot stacked area
        colors_stack = [event_colors.get(k, '#95a5a6') for k in stacked_data.keys()]
        ax3.stackplot(section_time, *stacked_data.values(), labels=stacked_data.keys(),
                     colors=colors_stack, alpha=0.8)
        ax3.legend(loc='upper left', fontsize=8)

    ax3.set_xlabel('Narrative Progress (%)')
    ax3.set_ylabel('Event Contribution')
    ax3.set_title('Event Types Across Narrative')
    ax3.grid(True, alpha=0.3)

    # 4. Statistics and major events list
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    # Build stats text
    stats_lines = [
        "Narrative Event Statistics",
        "══════════════════════════════",
        "",
        f"Total Events Detected: {trajectory.metadata.get('total_events', 0)}",
        f"Mean Event Density: {trajectory.metadata.get('mean_event_density', 0):.3f}",
        f"Event Variance: {trajectory.metadata.get('event_variance', 0):.4f}",
        "",
        "Event Breakdown:",
    ]

    for etype, count in sorted(event_types.items(), key=lambda x: -x[1]):
        stats_lines.append(f"  • {etype}: {count}")

    stats_lines.extend([
        "",
        "Major Plot Points:",
    ])

    for i, point in enumerate(major_points[:5], 1):
        stats_lines.append(f"  {i}. {point['position']*100:.1f}% (intensity: {point['intensity']:.2f})")

    stats_text = '\n'.join(stats_lines)

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    console.print(f"[green]✓ Saved narrative events visualization: {output_path}[/green]")
    return trajectory


def visualize_plot_state(
    text: str,
    language: str,
    title: str,
    output_path: Path,
    window_size: int = 1000
):
    """
    Visualize plot state (tension) over the narrative.

    Creates:
    - Tension curve with state annotations
    - Freytag's pyramid overlay
    - State distribution
    """
    # Create windows
    if language == 'ru':
        windows = create_windows_preserve_case(text, window_size, window_size // 2)
        functor = RussianPlotStateFunctor()
    else:
        windows = create_windows_english(text, window_size, window_size // 2)
        functor = PlotStateFunctor()

    console.print(f"[cyan]Processing {len(windows)} windows for plot state...[/cyan]")

    # Apply functor
    trajectory = functor(windows)

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Plot State Analysis: {title}', fontsize=16, fontweight='bold')

    # 1. Main tension curve
    ax1 = fig.add_subplot(2, 2, 1)
    time_points = np.linspace(0, 100, len(trajectory.values))

    # Color the line by tension level
    for i in range(len(trajectory.values) - 1):
        color = plt.cm.RdYlGn_r(trajectory.values[i])
        ax1.plot(time_points[i:i+2], trajectory.values[i:i+2], color=color, linewidth=2)

    # Fill under curve with gradient
    ax1.fill_between(time_points, trajectory.values, alpha=0.3,
                     color='gray')

    # Mark climax
    climax_pos = trajectory.metadata.get('climax_position', 0.5) * 100
    climax_tension = trajectory.metadata.get('climax_tension', 0.5)
    ax1.axvline(x=climax_pos, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax1.scatter([climax_pos], [climax_tension], s=200, c='red', marker='*',
                zorder=5, label=f'Climax ({climax_pos:.0f}%)')

    # Add tension level zones
    ax1.axhspan(0, 0.3, alpha=0.1, color='green', label='Low Tension')
    ax1.axhspan(0.3, 0.6, alpha=0.1, color='yellow')
    ax1.axhspan(0.6, 1.0, alpha=0.1, color='red', label='High Tension')

    ax1.set_xlabel('Narrative Progress (%)')
    ax1.set_ylabel('Tension Level')
    ax1.set_title('Narrative Tension Arc')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Freytag's Pyramid overlay
    ax2 = fig.add_subplot(2, 2, 2)

    # Draw idealized Freytag's pyramid
    pyramid_x = [0, 20, 50, 75, 100]
    pyramid_y = [0.2, 0.4, 1.0, 0.4, 0.2]  # Classic shape
    ax2.plot(pyramid_x, pyramid_y, 'k--', linewidth=2, alpha=0.5, label="Freytag's Pyramid")
    ax2.fill_between(pyramid_x, pyramid_y, alpha=0.1, color='gray')

    # Overlay actual tension
    ax2.plot(time_points, trajectory.values, 'b-', linewidth=2, label='Actual Tension')

    # Add labels
    ax2.annotate('Exposition', xy=(10, 0.3), fontsize=9, ha='center')
    ax2.annotate('Rising\nAction', xy=(35, 0.7), fontsize=9, ha='center')
    ax2.annotate('Climax', xy=(50, 1.05), fontsize=9, ha='center', fontweight='bold')
    ax2.annotate('Falling\nAction', xy=(62.5, 0.7), fontsize=9, ha='center')
    ax2.annotate('Resolution', xy=(87.5, 0.3), fontsize=9, ha='center')

    ax2.set_xlabel('Narrative Progress (%)')
    ax2.set_ylabel('Tension Level')
    ax2.set_title("Comparison with Freytag's Pyramid")
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. State distribution over narrative
    ax3 = fig.add_subplot(2, 2, 3)

    state_dist = trajectory.metadata.get('state_distribution', {})
    if state_dist:
        states = list(state_dist.keys())
        counts = list(state_dist.values())

        state_colors = {
            'equilibrium': '#27ae60',
            'rising_action': '#f39c12',
            'climax': '#e74c3c',
            'falling_action': '#3498db',
            'resolution': '#2ecc71',
            'disruption': '#9b59b6',
        }
        colors = [state_colors.get(s, '#95a5a6') for s in states]

        bars = ax3.bar(states, counts, color=colors, edgecolor='black', linewidth=0.5)
        ax3.set_ylabel('Number of Windows')
        ax3.set_title('Plot State Distribution')
        ax3.tick_params(axis='x', rotation=45)

        # Add count labels
        for bar, count in zip(bars, counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', fontsize=9)

    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Statistics and structure classification
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    structure = trajectory.metadata.get('narrative_structure', 'unknown')

    # Structure descriptions
    structure_desc = {
        'classic_arc': 'Traditional narrative with clear exposition, rising action, climax, and resolution',
        'tragic_arc': 'Builds to climax but lacks traditional resolution (tragedy)',
        'in_medias_res': 'Starts in the middle of action, high initial tension',
        'episodic': 'Multiple smaller arcs, no single climax',
        'complex': 'Non-traditional structure with multiple tension peaks',
        'unknown': 'Structure could not be determined',
    }

    stats_text = f"""
    Plot State Statistics
    ═══════════════════════════════════

    NARRATIVE STRUCTURE: {structure.upper().replace('_', ' ')}

    {structure_desc.get(structure, '')}

    ───────────────────────────────────

    Climax Position: {climax_pos:.1f}%
    Climax Tension: {climax_tension:.2f}

    Mean Tension: {trajectory.metadata.get('mean_tension', 0):.3f}
    Tension Variance: {trajectory.metadata.get('tension_variance', 0):.4f}

    ───────────────────────────────────

    State Breakdown:
    """

    for state, count in sorted(state_dist.items(), key=lambda x: -x[1]):
        pct = count / sum(state_dist.values()) * 100
        stats_text += f"\n      {state}: {count} ({pct:.1f}%)"

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    console.print(f"[green]✓ Saved plot state visualization: {output_path}[/green]")
    return trajectory


def create_combined_dashboard(
    text: str,
    language: str,
    title: str,
    output_path: Path,
    window_size: int = 1000
):
    """
    Create a combined dashboard showing all semantic dimensions.
    """
    # Create windows - preserve case for character extraction
    windows = create_windows_preserve_case(text, window_size, window_size // 2)

    if language == 'ru':
        interaction_functor = RussianCharacterInteractionFunctor()
        event_functor = RussianNarrativeEventFunctor()
        plot_functor = RussianPlotStateFunctor()
    else:
        interaction_functor = CharacterInteractionFunctor()
        event_functor = NarrativeEventFunctor()
        plot_functor = PlotStateFunctor()

    console.print(f"[cyan]Creating combined semantic dashboard...[/cyan]")

    # Apply all functors
    interaction_traj = interaction_functor(windows)
    event_traj = event_functor(windows)
    plot_traj = plot_functor(windows)

    # Create figure
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'Semantic Analysis Dashboard: {title}', fontsize=18, fontweight='bold')

    time_points = np.linspace(0, 100, len(interaction_traj.values))

    # 1. Combined trajectory view (top)
    ax1 = fig.add_subplot(3, 1, 1)

    ax1.plot(time_points, interaction_traj.values, 'b-', linewidth=2,
             label='Character Interactions', alpha=0.8)
    ax1.plot(time_points, event_traj.values, 'orange', linewidth=2,
             label='Narrative Events', alpha=0.8)
    ax1.plot(time_points, plot_traj.values, 'g-', linewidth=2,
             label='Plot Tension', alpha=0.8)

    # Mark climax
    climax_pos = plot_traj.metadata.get('climax_position', 0.5) * 100
    ax1.axvline(x=climax_pos, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Climax ({climax_pos:.0f}%)')

    ax1.set_xlabel('Narrative Progress (%)', fontsize=12)
    ax1.set_ylabel('Normalized Score', fontsize=12)
    ax1.set_title('All Semantic Dimensions Over Narrative', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)

    # 2. Correlation heatmap (bottom left)
    ax2 = fig.add_subplot(3, 3, 4)

    # Calculate correlations
    corr_matrix = np.corrcoef([
        interaction_traj.values,
        event_traj.values,
        plot_traj.values
    ])

    # Handle NaN
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    labels = ['Interactions', 'Events', 'Tension']
    im = ax2.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_yticklabels(labels)

    # Add correlation values
    for i in range(3):
        for j in range(3):
            text = ax2.text(j, i, f'{corr_matrix[i, j]:.2f}',
                           ha='center', va='center', fontsize=10,
                           color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

    ax2.set_title('Dimension Correlations')
    plt.colorbar(im, ax=ax2, shrink=0.8)

    # 3. Event type breakdown (bottom middle)
    ax3 = fig.add_subplot(3, 3, 5)

    event_types = event_traj.metadata.get('event_type_totals', {})
    if event_types:
        labels = list(event_types.keys())
        sizes = list(event_types.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        ax3.pie(sizes, labels=labels, autopct='%1.0f%%', colors=colors, startangle=90)
    ax3.set_title('Event Types')

    # 4. Key statistics (bottom right)
    ax4 = fig.add_subplot(3, 3, 6)
    ax4.axis('off')

    structure = plot_traj.metadata.get('narrative_structure', 'unknown')

    stats = f"""
    KEY METRICS
    ═══════════════════════

    Structure: {structure.upper()}
    Climax: {climax_pos:.0f}%

    Characters: {interaction_traj.metadata.get('total_characters', 0)}
    Events: {event_traj.metadata.get('total_events', 0)}

    Mean Tension: {plot_traj.metadata.get('mean_tension', 0):.2f}
    Event Density: {event_traj.metadata.get('mean_event_density', 0):.2f}

    Language: {language.upper()}
    Windows: {len(windows)}
    """

    ax4.text(0.1, 0.9, stats, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    # 5. Divergence from each other (bottom row spanning)
    ax5 = fig.add_subplot(3, 1, 3)

    # Calculate rolling divergences
    div_int_evt = np.abs(interaction_traj.values - event_traj.values)
    div_int_plt = np.abs(interaction_traj.values - plot_traj.values)
    div_evt_plt = np.abs(event_traj.values - plot_traj.values)

    ax5.fill_between(time_points, div_int_evt, alpha=0.5, label='|Interactions - Events|')
    ax5.fill_between(time_points, div_int_plt, alpha=0.5, label='|Interactions - Tension|')
    ax5.fill_between(time_points, div_evt_plt, alpha=0.5, label='|Events - Tension|')

    ax5.set_xlabel('Narrative Progress (%)', fontsize=12)
    ax5.set_ylabel('Divergence', fontsize=12)
    ax5.set_title('Inter-Dimension Divergence', fontsize=14)
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    console.print(f"[green]✓ Saved combined dashboard: {output_path}[/green]")


@click.command()
@click.option('--input', '-i', 'input_file', required=True, type=click.Path(exists=True),
              help='Input text JSON file')
@click.option('--output', '-o', 'output_dir', default='output/visualizations/',
              help='Output directory for visualizations')
@click.option('--language', '-l', default='en', type=click.Choice(['en', 'ru']),
              help='Language of the text')
@click.option('--window-size', '-w', default=1000, help='Window size in words')
def main(input_file: str, output_dir: str, language: str, window_size: int):
    """
    Generate visualizations for semantic/plot tracking functors.

    Creates detailed visualizations for character interactions,
    narrative events, and plot state analysis.
    """
    # Load text
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    text = data.get('text', '')
    title = data.get('title', data.get('title_en', Path(input_file).stem))

    # Auto-detect language if specified in data
    if 'language' in data:
        language = data['language']

    console.print("=" * 70)
    console.print(f"[bold]SEMANTIC FUNCTOR VISUALIZATION[/bold]")
    console.print(f"Title: {title}")
    console.print(f"Language: {language.upper()}")
    console.print(f"Text length: {len(text):,} characters")
    console.print("=" * 70)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate stem for output files
    stem = Path(input_file).stem

    # Generate all visualizations
    console.print("\n[bold]1. Character Interactions[/bold]")
    visualize_character_interactions(
        text, language, title,
        output_path / f"{stem}_interactions.png",
        window_size
    )

    console.print("\n[bold]2. Narrative Events[/bold]")
    visualize_narrative_events(
        text, language, title,
        output_path / f"{stem}_events.png",
        window_size
    )

    console.print("\n[bold]3. Plot State[/bold]")
    visualize_plot_state(
        text, language, title,
        output_path / f"{stem}_plot_state.png",
        window_size
    )

    console.print("\n[bold]4. Combined Dashboard[/bold]")
    create_combined_dashboard(
        text, language, title,
        output_path / f"{stem}_semantic_dashboard.png",
        window_size
    )

    console.print("\n" + "=" * 70)
    console.print(f"[bold green]✓ All visualizations saved to {output_dir}[/bold green]")
    console.print("=" * 70)


if __name__ == "__main__":
    main()
