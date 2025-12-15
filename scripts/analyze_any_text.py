#!/usr/bin/env python3
"""
Analyze Any Text - Dynamic Character and Narrative Analysis

This script automatically:
1. Extracts character definitions from any text
2. Builds character interaction networks
3. Creates visualizations (orbits, networks, events)
4. Calculates connectivity scores

No manual character definitions required!

Usage:
    python scripts/analyze_any_text.py -i data/raw/gutenberg/pg42671.json -o output/pride_and_prejudice
    python scripts/analyze_any_text.py -i data/raw/russian/texts/anna_karenina_ru.json -o output/anna_ru
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import to_rgba
import click
from rich.console import Console
from rich.table import Table

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.dynamic_characters import DynamicCharacterExtractor, ExtractedCharacter

console = Console()


@dataclass
class CharacterInteraction:
    """Represents interaction between two characters."""
    char1: str
    char2: str
    count: int
    first_occurrence: float  # normalized position
    last_occurrence: float


class DynamicNarrativeAnalyzer:
    """
    Analyze narrative structure using dynamically extracted characters.
    """

    def __init__(
        self,
        text: str,
        title: str = "Unknown",
        min_mentions: int = 5,
        max_characters: int = 25,
    ):
        self.text = text
        self.title = title
        self.text_length = len(text)

        # Extract characters dynamically
        self.extractor = DynamicCharacterExtractor(
            min_mentions=min_mentions,
            max_characters=max_characters,
        )
        self.characters = self.extractor.extract(text)
        self.language = self.extractor.language

        # Build interactions
        self.interactions = self._build_interactions()

        console.print(f"[bold]Analyzed: {title}[/bold]")
        console.print(f"Language: {self.language}")
        console.print(f"Characters found: {len(self.characters)}")

    def _build_interactions(self) -> Dict[str, CharacterInteraction]:
        """Build character interaction matrix from co-occurrences."""
        interactions = {}

        for name, char in self.characters.items():
            for other, count in char.co_occurrences.items():
                if other in self.characters and count > 0:
                    # Create canonical key (alphabetically sorted)
                    key = tuple(sorted([name, other]))
                    if key not in interactions:
                        interactions[key] = CharacterInteraction(
                            char1=key[0],
                            char2=key[1],
                            count=count,
                            first_occurrence=min(
                                self.characters[name].first_appearance,
                                self.characters[other].first_appearance
                            ),
                            last_occurrence=max(
                                self.characters[name].last_appearance,
                                self.characters[other].last_appearance
                            ),
                        )
                    else:
                        # Update count (take max since both characters track)
                        interactions[key].count = max(interactions[key].count, count)

        return interactions

    def calculate_connectivity(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate connectivity scores for each character.

        Returns dict with metrics:
        - mentions: Total mentions in text
        - degree: Number of unique characters they interact with
        - strength: Total interaction weight
        - weighted_degree: Average interaction strength
        - centrality: Bridge score (connects clusters)
        - connectivity_index: Composite score (0-100)
        """
        scores = {}

        for name, char in self.characters.items():
            # Degree: unique connections
            connections = set()
            total_weight = 0

            for (c1, c2), inter in self.interactions.items():
                if c1 == name:
                    connections.add(c2)
                    total_weight += inter.count
                elif c2 == name:
                    connections.add(c1)
                    total_weight += inter.count

            degree = len(connections)
            strength = total_weight
            weighted_degree = strength / degree if degree > 0 else 0

            # Centrality: approximate betweenness
            # Characters who connect otherwise disconnected groups
            centrality = 0
            for conn1 in connections:
                for conn2 in connections:
                    if conn1 < conn2:
                        # Check if conn1 and conn2 are connected
                        key = tuple(sorted([conn1, conn2]))
                        if key not in self.interactions:
                            centrality += 1

            scores[name] = {
                'mentions': char.mentions,
                'degree': degree,
                'strength': strength,
                'weighted_degree': weighted_degree,
                'centrality': centrality,
            }

        # Normalize and compute composite index
        if scores:
            max_mentions = max(s['mentions'] for s in scores.values()) or 1
            max_degree = max(s['degree'] for s in scores.values()) or 1
            max_strength = max(s['strength'] for s in scores.values()) or 1
            max_wd = max(s['weighted_degree'] for s in scores.values()) or 1
            max_centrality = max(s['centrality'] for s in scores.values()) or 1

            for name in scores:
                s = scores[name]
                s['mentions_norm'] = s['mentions'] / max_mentions
                s['degree_norm'] = s['degree'] / max_degree
                s['strength_norm'] = s['strength'] / max_strength
                s['wd_norm'] = s['weighted_degree'] / max_wd
                s['centrality_norm'] = s['centrality'] / max_centrality if max_centrality > 0 else 0

                # Composite index (weighted average)
                s['connectivity_index'] = (
                    0.25 * s['mentions_norm'] +
                    0.20 * s['degree_norm'] +
                    0.25 * s['strength_norm'] +
                    0.15 * s['wd_norm'] +
                    0.15 * s['centrality_norm']
                ) * 100

        return scores

    def print_connectivity_report(self):
        """Print connectivity analysis table."""
        scores = self.calculate_connectivity()

        table = Table(title=f"Character Connectivity - {self.title}")
        table.add_column("Rank", style="dim", justify="right")
        table.add_column("Character", style="cyan")
        table.add_column("Role", style="magenta")
        table.add_column("Index", justify="right", style="bold green")
        table.add_column("Mentions", justify="right")
        table.add_column("Connections", justify="right")
        table.add_column("Strength", justify="right")

        # Sort by connectivity index
        sorted_chars = sorted(
            scores.items(),
            key=lambda x: x[1]['connectivity_index'],
            reverse=True
        )

        for rank, (name, s) in enumerate(sorted_chars, 1):
            char = self.characters[name]
            table.add_row(
                str(rank),
                name,
                char.role,
                f"{s['connectivity_index']:.1f}",
                str(s['mentions']),
                str(s['degree']),
                str(s['strength']),
            )

        console.print(table)

        # Print most connected character
        if sorted_chars:
            top = sorted_chars[0]
            console.print(f"\n[bold green]Most Connected Character: {top[0]}[/bold green]")
            console.print(f"  Connectivity Index: {top[1]['connectivity_index']:.1f}/100")

    def visualize_network(self, output_path: Path, top_n: int = 15):
        """Create character network visualization."""
        # Select top N characters
        sorted_chars = sorted(
            self.characters.items(),
            key=lambda x: x[1].mentions,
            reverse=True
        )[:top_n]

        char_names = {c[0] for c in sorted_chars}

        # Filter interactions to only include these characters
        relevant_interactions = {
            k: v for k, v in self.interactions.items()
            if v.char1 in char_names and v.char2 in char_names
        }

        if not relevant_interactions:
            console.print("[yellow]No interactions found for network visualization[/yellow]")
            return

        # Layout: circular arrangement
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_aspect('equal')

        n_chars = len(sorted_chars)
        angles = np.linspace(0, 2 * np.pi, n_chars, endpoint=False)

        positions = {}
        for i, (name, char) in enumerate(sorted_chars):
            x = np.cos(angles[i]) * 0.8
            y = np.sin(angles[i]) * 0.8
            positions[name] = (x, y)

        # Draw edges (interactions)
        max_weight = max(i.count for i in relevant_interactions.values()) if relevant_interactions else 1

        for (c1, c2), inter in relevant_interactions.items():
            if c1 in positions and c2 in positions:
                x1, y1 = positions[c1]
                x2, y2 = positions[c2]

                # Line thickness based on interaction strength
                width = 0.5 + (inter.count / max_weight) * 3
                alpha = 0.3 + (inter.count / max_weight) * 0.5

                ax.plot([x1, x2], [y1, y2],
                       color='gray', alpha=alpha, linewidth=width, zorder=1)

        # Draw nodes (characters)
        for name, char in sorted_chars:
            x, y = positions[name]
            color = char.color

            # Size based on mentions
            max_mentions = sorted_chars[0][1].mentions
            size = 0.08 + (char.mentions / max_mentions) * 0.12

            circle = Circle((x, y), size, facecolor=color,
                           edgecolor='white', linewidth=2, zorder=2)
            ax.add_patch(circle)

            # Label
            label_y = y + size + 0.05
            ax.text(x, label_y, name, ha='center', va='bottom',
                   fontsize=10, fontweight='bold', zorder=3)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        ax.set_title(f'Character Network: {self.title}', fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        console.print(f"[green]Saved network: {output_path}[/green]")

    def visualize_timeline(self, output_path: Path, top_n: int = 10):
        """Create character appearance timeline."""
        sorted_chars = sorted(
            self.characters.items(),
            key=lambda x: x[1].mentions,
            reverse=True
        )[:top_n]

        fig, ax = plt.subplots(figsize=(14, 8))

        for i, (name, char) in enumerate(sorted_chars):
            y = len(sorted_chars) - i - 1

            # Draw appearance range
            start = char.first_appearance
            end = char.last_appearance
            ax.barh(y, end - start, left=start, height=0.6,
                   color=char.color, alpha=0.7, edgecolor='white', linewidth=1)

            # Label
            ax.text(-0.02, y, name, ha='right', va='center',
                   fontsize=10, fontweight='bold')

            # Mention count
            ax.text(1.02, y, f'{char.mentions}', ha='left', va='center',
                   fontsize=9, color='gray')

        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.5, len(sorted_chars) - 0.5)
        ax.set_xlabel('Narrative Progress', fontsize=12)
        ax.set_title(f'Character Appearance Timeline: {self.title}',
                    fontsize=14, fontweight='bold', pad=15)

        # Remove y-axis
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        console.print(f"[green]Saved timeline: {output_path}[/green]")

    def visualize_orbits(self, output_path: Path, top_n: int = 8):
        """Create character orbital system visualization."""
        sorted_chars = sorted(
            self.characters.items(),
            key=lambda x: x[1].mentions,
            reverse=True
        )[:top_n]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for idx, (name, char) in enumerate(sorted_chars):
            ax = axes[idx]
            ax.set_aspect('equal')

            # Draw central character
            main_circle = Circle((0, 0), 0.15, facecolor=char.color,
                                edgecolor='white', linewidth=2, zorder=3)
            ax.add_patch(main_circle)
            ax.text(0, 0, name[:12], ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white', zorder=4)

            # Draw satellites (connected characters)
            satellites = sorted(
                char.co_occurrences.items(),
                key=lambda x: x[1],
                reverse=True
            )[:6]

            if satellites:
                max_count = satellites[0][1]
                angles = np.linspace(0, 2 * np.pi, len(satellites), endpoint=False)

                for j, (sat_name, count) in enumerate(satellites):
                    # Position
                    dist = 0.4 + (1 - count/max_count) * 0.3
                    x = np.cos(angles[j]) * dist
                    y = np.sin(angles[j]) * dist

                    # Get color from character if exists
                    sat_color = '#888888'
                    if sat_name in self.characters:
                        sat_color = self.characters[sat_name].color

                    # Size based on interaction strength
                    size = 0.05 + (count / max_count) * 0.08

                    # Draw orbit path
                    orbit = Circle((0, 0), dist, fill=False,
                                  edgecolor='gray', linestyle='--',
                                  alpha=0.3, zorder=1)
                    ax.add_patch(orbit)

                    # Draw satellite
                    sat_circle = Circle((x, y), size, facecolor=sat_color,
                                       edgecolor='white', linewidth=1, zorder=2)
                    ax.add_patch(sat_circle)

                    # Label
                    ax.text(x, y + size + 0.05, sat_name[:10], ha='center', va='bottom',
                           fontsize=7, zorder=5)

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.axis('off')
            ax.set_title(f'{name}\n({char.mentions} mentions)', fontsize=10, pad=5)

        # Hide unused subplots
        for idx in range(len(sorted_chars), len(axes)):
            axes[idx].axis('off')

        fig.suptitle(f'Character Orbital Systems: {self.title}',
                    fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        console.print(f"[green]Saved orbits: {output_path}[/green]")

    def save_analysis(self, output_path: Path):
        """Save full analysis to JSON."""
        scores = self.calculate_connectivity()

        analysis = {
            'title': self.title,
            'language': self.language,
            'character_count': len(self.characters),
            'interaction_count': len(self.interactions),
            'characters': {
                name: {
                    **char.to_dict(),
                    'connectivity': scores.get(name, {}),
                }
                for name, char in self.characters.items()
            },
            'interactions': [
                {
                    'char1': k[0],
                    'char2': k[1],
                    'count': v.count,
                    'first': v.first_occurrence,
                    'last': v.last_occurrence,
                }
                for k, v in self.interactions.items()
            ],
            'most_connected': max(
                scores.items(),
                key=lambda x: x[1].get('connectivity_index', 0)
            )[0] if scores else None,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        console.print(f"[green]Saved analysis: {output_path}[/green]")


@click.command()
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=True), help='Input JSON file with text')
@click.option('--output', '-o', 'output_dir', required=True,
              type=click.Path(), help='Output directory for results')
@click.option('--min-mentions', '-m', default=5,
              help='Minimum mentions for character inclusion')
@click.option('--max-characters', '-n', default=25,
              help='Maximum number of characters to extract')
@click.option('--top-visualize', '-t', default=15,
              help='Number of characters in visualizations')
def main(
    input_path: str,
    output_dir: str,
    min_mentions: int,
    max_characters: int,
    top_visualize: int,
):
    """
    Analyze any text for character dynamics and narrative structure.

    Automatically extracts characters, builds interaction networks,
    and creates visualizations - no manual character definitions needed!
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load text
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    text = data.get('text', '')
    title = data.get('title', input_path.stem)

    if not text:
        console.print("[red]Error: No text found in input file[/red]")
        return

    console.print(f"\n[bold blue]═══ Dynamic Narrative Analysis ═══[/bold blue]\n")

    # Create analyzer
    analyzer = DynamicNarrativeAnalyzer(
        text=text,
        title=title,
        min_mentions=min_mentions,
        max_characters=max_characters,
    )

    # Print character summary
    analyzer.extractor.print_summary(analyzer.characters)

    # Print connectivity report
    console.print()
    analyzer.print_connectivity_report()

    # Create visualizations
    console.print(f"\n[bold]Creating visualizations...[/bold]")

    analyzer.visualize_network(
        output_dir / f"{input_path.stem}_network.png",
        top_n=top_visualize
    )

    analyzer.visualize_timeline(
        output_dir / f"{input_path.stem}_timeline.png",
        top_n=min(top_visualize, 12)
    )

    analyzer.visualize_orbits(
        output_dir / f"{input_path.stem}_orbits.png",
        top_n=8
    )

    # Save analysis JSON
    analyzer.save_analysis(output_dir / f"{input_path.stem}_analysis.json")

    console.print(f"\n[bold green]✓ Analysis complete![/bold green]")
    console.print(f"  Output: {output_dir}")


if __name__ == '__main__':
    main()
