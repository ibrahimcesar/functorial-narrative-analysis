#!/usr/bin/env python3
"""
Visualize Dual Narrative Structure in Anna Karenina

Tolstoy originally planned to call the novel "Two Marriages" - it tells two
parallel stories:
1. Anna Karenina's tragic affair with Vronsky
2. Konstantin Levin's journey to find meaning through love and faith

This script creates visualizations showing:
- Events/facts for each storyline
- Where the two narratives converge and diverge
- Character presence for each protagonist

Usage:
    python scripts/visualize_dual_narrative.py \
        --input data/raw/tolstoy/anna_karenina.json \
        --output output/visualizations/anna_karenina/ \
        --language en
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import click
from rich.console import Console
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

console = Console()


# Character groups for each storyline
ANNA_STORYLINE_EN = {
    # Core characters
    'Anna', 'Karenina', 'Vronsky', 'Karenin', 'Alexei', 'Alexey',
    'Seryozha', 'Serezha',  # Anna's son
    # Extended
    'Betsy', 'Princess', 'Countess', 'Stiva', 'Dolly',
}

LEVIN_STORYLINE_EN = {
    # Core characters
    'Levin', 'Kitty', 'Konstantin', 'Kostya',
    'Nikolai', 'Nicholas',  # Levin's brother
    # Extended
    'Agafea', 'Koznyshev', 'Sergei', 'Varenka',
    'Oblonsky', 'Stiva',  # Shared
}

ANNA_STORYLINE_RU = {
    # Core Anna storyline
    'Анна', 'Каренина', 'Каренин', 'Вронский', 'Вронским', 'Вронского',
    'Алексей', 'Алексея', 'Алексею',
    'Сережа', 'Сережи',  # Anna's son
    # Extended
    'Бетси', 'Княгиня', 'Графиня',
}

LEVIN_STORYLINE_RU = {
    # Core Levin storyline
    'Левин', 'Левина', 'Левину', 'Левиным',
    'Кити', 'Китти',
    'Константин', 'Костя',
    'Николай', 'Николая',  # Brother
    # Extended
    'Кознышев', 'Сергей', 'Сергея',
    'Варенька', 'Агафья',
}

# Shared characters (appear in both storylines)
SHARED_CHARACTERS_EN = {'Dolly', 'Stiva', 'Oblonsky', 'Stepan'}
SHARED_CHARACTERS_RU = {'Долли', 'Стива', 'Облонский', 'Степан', 'Дарья'}


def create_windows(text: str, window_size: int = 1000, overlap: int = 500) -> List[str]:
    """Create overlapping windows preserving case."""
    words = text.split()
    step = window_size - overlap
    windows = []
    for i in range(0, len(words), step):
        window = ' '.join(words[i:i + window_size])
        if len(window.split()) >= window_size // 2:
            windows.append(window)
    return windows if windows else [text]


def count_character_mentions(text: str, character_set: Set[str]) -> int:
    """Count mentions of characters from a set."""
    count = 0
    for char in character_set:
        count += len(re.findall(rf'\b{char}\b', text, re.IGNORECASE))
    return count


def detect_storyline_events(text: str, language: str) -> Dict[str, int]:
    """
    Detect events specific to each storyline.

    Returns dict with counts of Anna events, Levin events, and shared events.
    """
    if language == 'ru':
        anna_chars = ANNA_STORYLINE_RU
        levin_chars = LEVIN_STORYLINE_RU
    else:
        anna_chars = ANNA_STORYLINE_EN
        levin_chars = LEVIN_STORYLINE_EN

    anna_count = count_character_mentions(text, anna_chars)
    levin_count = count_character_mentions(text, levin_chars)

    return {
        'anna': anna_count,
        'levin': levin_count,
        'ratio': anna_count / max(1, anna_count + levin_count)
    }


def analyze_dual_narrative(
    text: str,
    language: str,
    window_size: int = 1000
) -> Tuple[List[float], List[float], List[float]]:
    """
    Analyze which storyline dominates each section of the narrative.

    Returns:
        Tuple of (anna_presence, levin_presence, narrative_position)
    """
    windows = create_windows(text, window_size, window_size // 2)

    anna_presence = []
    levin_presence = []
    positions = []

    for i, window in enumerate(windows):
        result = detect_storyline_events(window, language)

        anna_presence.append(result['anna'])
        levin_presence.append(result['levin'])
        positions.append(i / max(1, len(windows) - 1))

    # Normalize to percentages
    max_count = max(max(anna_presence), max(levin_presence), 1)
    anna_norm = [a / max_count for a in anna_presence]
    levin_norm = [l / max_count for l in levin_presence]

    return anna_norm, levin_norm, positions


def detect_major_events(text: str, language: str) -> Dict[str, List[str]]:
    """
    Detect major events for each storyline.
    """
    if language == 'ru':
        # Anna's key events in Russian
        anna_events = {
            'affair': ['роман', 'любовник', 'измена', 'страсть'],
            'scandal': ['скандал', 'позор', 'общество', 'свет'],
            'jealousy': ['ревность', 'ревнив', 'подозрение'],
            'despair': ['отчаяние', 'безысход', 'невыносим'],
            'death': ['поезд', 'колеса', 'смерть', 'конец'],
        }
        # Levin's key events
        levin_events = {
            'rejection': ['отказ', 'отвергла', 'нет'],
            'farming': ['хозяйство', 'земля', 'крестьяне', 'урожай'],
            'proposal': ['предложение', 'согласие', 'свадьба'],
            'faith': ['вера', 'бог', 'смысл', 'душа'],
            'family': ['семья', 'ребенок', 'сын', 'жена'],
        }
    else:
        anna_events = {
            'affair': ['affair', 'lover', 'passion', 'love'],
            'scandal': ['scandal', 'shame', 'society', 'disgrace'],
            'jealousy': ['jealousy', 'jealous', 'suspicion'],
            'despair': ['despair', 'hopeless', 'unbearable'],
            'death': ['train', 'wheels', 'death', 'end'],
        }
        levin_events = {
            'rejection': ['refused', 'rejected', 'no'],
            'farming': ['farm', 'land', 'peasants', 'harvest', 'mowing'],
            'proposal': ['proposal', 'accepted', 'wedding', 'marriage'],
            'faith': ['faith', 'god', 'meaning', 'soul', 'believe'],
            'family': ['family', 'child', 'son', 'wife', 'baby'],
        }

    return {'anna': anna_events, 'levin': levin_events}


def find_event_occurrences(
    windows: List[str],
    events_dict: Dict[str, List[str]],
    positions: List[float]
) -> Dict[str, List[Tuple[float, str]]]:
    """Find where key events occur in the narrative."""
    occurrences = defaultdict(list)

    for i, (window, pos) in enumerate(zip(windows, positions)):
        window_lower = window.lower()
        for event_type, keywords in events_dict.items():
            for keyword in keywords:
                if keyword.lower() in window_lower:
                    occurrences[event_type].append((pos, keyword))
                    break

    return dict(occurrences)


def visualize_dual_narrative(
    text: str,
    language: str,
    title: str,
    output_path: Path,
    window_size: int = 1000
):
    """
    Create visualization showing Anna vs Levin storylines.
    """
    console.print("[cyan]Analyzing dual narrative structure...[/cyan]")

    # Analyze character presence
    anna_presence, levin_presence, positions = analyze_dual_narrative(
        text, language, window_size
    )

    # Create windows for event detection
    windows = create_windows(text, window_size, window_size // 2)
    time_points = np.linspace(0, 100, len(windows))

    # Get event patterns
    events = detect_major_events(text, language)
    anna_events = find_event_occurrences(windows, events['anna'], positions)
    levin_events = find_event_occurrences(windows, events['levin'], positions)

    # Create figure with two main plots
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f'Dual Narrative Structure: {title}\n"Two Marriages" - Anna Karenina & Konstantin Levin',
                 fontsize=16, fontweight='bold')

    # ==========================================================================
    # 1. TOP: Anna's storyline (inverted, going down from top)
    # ==========================================================================
    ax1 = fig.add_subplot(3, 1, 1)

    # Plot Anna's presence (inverted so it goes DOWN)
    anna_smooth = np.convolve(anna_presence, np.ones(10)/10, mode='same')
    ax1.fill_between(time_points, 0, -np.array(anna_smooth), alpha=0.6, color='#e74c3c',
                     label="Anna's Storyline")
    ax1.plot(time_points, -np.array(anna_smooth), color='#c0392b', linewidth=2)

    # Mark Anna's key events
    event_colors = {
        'affair': '#e91e63',
        'scandal': '#9c27b0',
        'jealousy': '#ff5722',
        'despair': '#795548',
        'death': '#212121',
    }

    for event_type, occurrences in anna_events.items():
        if occurrences:
            # Get first significant occurrence
            pos = occurrences[0][0] * 100
            ax1.axvline(x=pos, color=event_colors.get(event_type, 'gray'),
                       linestyle='--', alpha=0.7, linewidth=1.5)
            ax1.annotate(event_type.title(), xy=(pos, -0.3),
                        fontsize=8, rotation=45, ha='right', color=event_colors.get(event_type, 'gray'))

    ax1.set_xlim(0, 100)
    ax1.set_ylim(-1.2, 0.1)
    ax1.set_ylabel("Anna's Story\n(Tragedy →)")
    ax1.set_title("ANNA KARENINA - Passion, Scandal, and Tragedy", fontsize=12, color='#c0392b')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linewidth=2)

    # Add event legend for Anna
    anna_patches = [mpatches.Patch(color=color, label=name.title())
                   for name, color in event_colors.items() if name in anna_events]
    if anna_patches:
        ax1.legend(handles=anna_patches, loc='lower left', fontsize=8)

    # ==========================================================================
    # 2. MIDDLE: Convergence points and shared moments
    # ==========================================================================
    ax2 = fig.add_subplot(3, 1, 2)

    # Calculate where storylines converge (both present)
    convergence = np.array(anna_presence) * np.array(levin_presence)
    convergence_smooth = np.convolve(convergence, np.ones(10)/10, mode='same')

    # Show convergence areas
    ax2.fill_between(time_points, convergence_smooth, alpha=0.5, color='#9b59b6',
                     label='Storylines Converge')
    ax2.plot(time_points, convergence_smooth, color='#8e44ad', linewidth=2)

    # Calculate dominance (positive = Anna dominant, negative = Levin dominant)
    dominance = np.array(anna_presence) - np.array(levin_presence)
    dominance_smooth = np.convolve(dominance, np.ones(10)/10, mode='same')

    ax2_twin = ax2.twinx()
    ax2_twin.plot(time_points, dominance_smooth, color='gray', linewidth=1.5,
                  linestyle='--', alpha=0.7, label='Anna↑ / Levin↓')
    ax2_twin.axhline(y=0, color='gray', linewidth=1, alpha=0.5)
    ax2_twin.set_ylabel('Dominance (Anna↑ / Levin↓)', fontsize=9, color='gray')
    ax2_twin.set_ylim(-1, 1)

    ax2.set_xlim(0, 100)
    ax2.set_ylabel('Storyline Convergence')
    ax2.set_title('CONVERGENCE - Where the Two Marriages Meet', fontsize=12, color='#8e44ad')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    # Mark key convergence points
    # Find peaks in convergence
    for i in range(5, len(convergence_smooth) - 5):
        if convergence_smooth[i] > 0.3:
            if convergence_smooth[i] > convergence_smooth[i-1] and convergence_smooth[i] > convergence_smooth[i+1]:
                ax2.scatter([time_points[i]], [convergence_smooth[i]], s=100, c='purple',
                           marker='*', zorder=5)

    # ==========================================================================
    # 3. BOTTOM: Levin's storyline (going up from bottom)
    # ==========================================================================
    ax3 = fig.add_subplot(3, 1, 3)

    # Plot Levin's presence
    levin_smooth = np.convolve(levin_presence, np.ones(10)/10, mode='same')
    ax3.fill_between(time_points, 0, levin_smooth, alpha=0.6, color='#27ae60',
                     label="Levin's Storyline")
    ax3.plot(time_points, levin_smooth, color='#1e8449', linewidth=2)

    # Mark Levin's key events
    levin_event_colors = {
        'rejection': '#e74c3c',
        'farming': '#8bc34a',
        'proposal': '#e91e63',
        'faith': '#ffc107',
        'family': '#4caf50',
    }

    for event_type, occurrences in levin_events.items():
        if occurrences:
            pos = occurrences[0][0] * 100
            ax3.axvline(x=pos, color=levin_event_colors.get(event_type, 'gray'),
                       linestyle='--', alpha=0.7, linewidth=1.5)
            ax3.annotate(event_type.title(), xy=(pos, 0.3),
                        fontsize=8, rotation=45, ha='left', color=levin_event_colors.get(event_type, 'gray'))

    ax3.set_xlim(0, 100)
    ax3.set_ylim(-0.1, 1.2)
    ax3.set_xlabel('Narrative Progress (%)')
    ax3.set_ylabel("Levin's Story\n(← Redemption)")
    ax3.set_title("KONSTANTIN LEVIN - Doubt, Love, and Faith", fontsize=12, color='#1e8449')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linewidth=2)

    # Add event legend for Levin
    levin_patches = [mpatches.Patch(color=color, label=name.title())
                    for name, color in levin_event_colors.items() if name in levin_events]
    if levin_patches:
        ax3.legend(handles=levin_patches, loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    console.print(f"[green]✓ Saved dual narrative visualization: {output_path}[/green]")


def visualize_character_timelines(
    text: str,
    language: str,
    title: str,
    output_path: Path,
    window_size: int = 1000
):
    """
    Create timeline showing when each major character appears.
    """
    console.print("[cyan]Creating character timeline...[/cyan]")

    windows = create_windows(text, window_size, window_size // 2)
    time_points = np.linspace(0, 100, len(windows))

    # Define key characters to track
    if language == 'ru':
        characters = {
            'Анна': ['Анна', 'Анны', 'Анне', 'Анну'],
            'Вронский': ['Вронский', 'Вронского', 'Вронским', 'Вронскому'],
            'Каренин': ['Каренин', 'Каренина', 'Каренину', 'Карениным'],
            'Левин': ['Левин', 'Левина', 'Левину', 'Левиным'],
            'Кити': ['Кити', 'Китти'],
            'Стива': ['Стива', 'Степан', 'Облонский'],
            'Долли': ['Долли', 'Дарья'],
        }
    else:
        characters = {
            'Anna': ['Anna'],
            'Vronsky': ['Vronsky'],
            'Karenin': ['Karenin', 'Alexei Alexandrovitch'],
            'Levin': ['Levin', 'Konstantin'],
            'Kitty': ['Kitty'],
            'Stiva': ['Stiva', 'Stepan', 'Oblonsky'],
            'Dolly': ['Dolly', 'Darya'],
        }

    # Track each character's presence
    char_presence = {name: [] for name in characters}

    for window in windows:
        for char_name, variants in characters.items():
            count = sum(len(re.findall(rf'\b{v}\b', window, re.IGNORECASE)) for v in variants)
            char_presence[char_name].append(count)

    # Normalize
    for char_name in char_presence:
        max_val = max(char_presence[char_name]) if max(char_presence[char_name]) > 0 else 1
        char_presence[char_name] = [c / max_val for c in char_presence[char_name]]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(f'Character Presence Timeline: {title}', fontsize=14, fontweight='bold')

    # Color scheme
    colors = {
        'Anna': '#e74c3c', 'Анна': '#e74c3c',
        'Vronsky': '#e91e63', 'Вронский': '#e91e63',
        'Karenin': '#9c27b0', 'Каренин': '#9c27b0',
        'Levin': '#27ae60', 'Левин': '#27ae60',
        'Kitty': '#4caf50', 'Кити': '#4caf50',
        'Stiva': '#ff9800', 'Стива': '#ff9800',
        'Dolly': '#ff5722', 'Долли': '#ff5722',
    }

    # Plot each character
    y_offset = 0
    y_spacing = 1.5
    y_positions = {}

    for char_name, presence in char_presence.items():
        smooth = np.convolve(presence, np.ones(8)/8, mode='same')
        color = colors.get(char_name, 'gray')

        # Plot as filled area at specific y level
        ax.fill_between(time_points, y_offset, y_offset + np.array(smooth),
                       alpha=0.6, color=color, label=char_name)
        ax.plot(time_points, y_offset + np.array(smooth), color=color, linewidth=1.5)

        y_positions[char_name] = y_offset + 0.5
        y_offset += y_spacing

    # Add character labels on y-axis
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))

    # Add vertical lines for major plot points
    plot_points = [
        (8, 'Ball Scene'),
        (25, 'Anna leaves Moscow'),
        (50, 'Mid-novel crisis'),
        (75, 'Levin\'s proposal'),
        (95, 'Anna\'s death'),
    ]

    for pos, label in plot_points:
        ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.5)
        ax.text(pos, y_offset + 0.2, label, fontsize=8, rotation=90, va='bottom')

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, y_offset + 1)
    ax.set_xlabel('Narrative Progress (%)')
    ax.set_title('When Characters Appear in the Narrative')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    console.print(f"[green]✓ Saved character timeline: {output_path}[/green]")


@click.command()
@click.option('--input', '-i', 'input_file', required=True, type=click.Path(exists=True))
@click.option('--output', '-o', 'output_dir', default='output/visualizations/')
@click.option('--language', '-l', default='en', type=click.Choice(['en', 'ru']))
@click.option('--window-size', '-w', default=1000)
def main(input_file: str, output_dir: str, language: str, window_size: int):
    """
    Visualize the dual narrative structure of Anna Karenina.

    Shows Anna's tragic arc vs Levin's redemptive arc, their convergence
    points, and when they diverge.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    text = data.get('text', '')
    title = data.get('title', data.get('title_en', Path(input_file).stem))

    if 'language' in data:
        language = data['language']

    console.print("=" * 70)
    console.print("[bold]DUAL NARRATIVE ANALYSIS[/bold]")
    console.print(f"Title: {title}")
    console.print(f"Language: {language.upper()}")
    console.print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stem = Path(input_file).stem

    # Generate visualizations
    visualize_dual_narrative(
        text, language, title,
        output_path / f"{stem}_dual_narrative.png",
        window_size
    )

    visualize_character_timelines(
        text, language, title,
        output_path / f"{stem}_character_timeline.png",
        window_size
    )

    console.print("\n" + "=" * 70)
    console.print(f"[bold green]✓ Dual narrative visualizations saved to {output_dir}[/bold green]")


if __name__ == "__main__":
    main()
