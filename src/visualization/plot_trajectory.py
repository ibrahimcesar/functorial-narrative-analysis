"""
Basic visualization for narrative trajectories.

Creates plots showing emotional arcs over narrative time.
"""

import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import click
from rich.console import Console

console = Console()


def plot_single_trajectory(
    trajectory_file: Path,
    output_file: Optional[Path] = None,
    smooth_sigma: float = 3.0,
    title: Optional[str] = None
):
    """
    Plot a single trajectory.

    Args:
        trajectory_file: Path to trajectory JSON file
        output_file: Output path for the plot
        smooth_sigma: Gaussian smoothing sigma
        title: Optional title override
    """
    with open(trajectory_file, 'r') as f:
        data = json.load(f)

    values = np.array(data["values"])
    time_points = np.array(data["time_points"])
    metadata = data.get("metadata", {})

    # Smooth the trajectory
    smoothed = gaussian_filter1d(values, sigma=smooth_sigma)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot raw values with low alpha
    ax.plot(time_points, values, alpha=0.2, color='blue', linewidth=0.5, label='Raw')

    # Plot smoothed trajectory
    ax.plot(time_points, smoothed, color='blue', linewidth=2, label='Smoothed')

    # Add horizontal line at neutral
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Labels
    plot_title = title or metadata.get("title", "Sentiment Trajectory")
    ax.set_title(f"{plot_title}\n({metadata.get('author', 'Unknown')})", fontsize=14)
    ax.set_xlabel("Narrative Time", fontsize=12)
    ax.set_ylabel("Sentiment (negative ← → positive)", fontsize=12)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add stats annotation
    stats_text = f"Mean: {np.mean(values):.3f}\nStd: {np.std(values):.3f}\nWindows: {len(values)}"
    ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        console.print(f"[green]Saved plot to {output_file}[/green]")
    else:
        plt.show()

    plt.close()


def plot_comparison(
    trajectory_files: List[Path],
    output_file: Optional[Path] = None,
    smooth_sigma: float = 5.0,
    normalize_length: bool = True
):
    """
    Plot multiple trajectories for comparison.

    Args:
        trajectory_files: List of trajectory JSON files
        output_file: Output path for the plot
        smooth_sigma: Gaussian smoothing sigma
        normalize_length: Resample all to same length
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectory_files)))

    for i, traj_file in enumerate(trajectory_files):
        with open(traj_file, 'r') as f:
            data = json.load(f)

        values = np.array(data["values"])
        time_points = np.array(data["time_points"])
        metadata = data.get("metadata", {})

        # Smooth
        smoothed = gaussian_filter1d(values, sigma=smooth_sigma)

        # Normalize to same length if needed
        if normalize_length:
            new_time = np.linspace(0, 1, 100)
            smoothed = np.interp(new_time, time_points, smoothed)
            time_points = new_time

        label = metadata.get("title", traj_file.stem)
        if len(label) > 30:
            label = label[:27] + "..."

        ax.plot(time_points, smoothed, color=colors[i], linewidth=2, label=label)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title("Sentiment Trajectory Comparison", fontsize=14)
    ax.set_xlabel("Narrative Time", fontsize=12)
    ax.set_ylabel("Sentiment (negative ← → positive)", fontsize=12)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 1)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        console.print(f"[green]Saved comparison plot to {output_file}[/green]")
    else:
        plt.show()

    plt.close()


@click.command()
@click.option('--input', '-i', 'input_path', required=True, type=click.Path(exists=True),
              help='Input trajectory file or directory')
@click.option('--output', '-o', 'output_path', type=click.Path(),
              help='Output file for the plot')
@click.option('--smooth', '-s', default=3.0, help='Smoothing sigma')
@click.option('--compare', is_flag=True, help='Compare multiple trajectories')
def main(input_path: str, output_path: str, smooth: float, compare: bool):
    """Visualize narrative trajectories."""
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else None

    if input_path.is_dir():
        # Plot all trajectories in directory
        trajectory_files = list(input_path.glob("*_sentiment.json"))

        if compare:
            plot_comparison(trajectory_files, output_path, smooth_sigma=smooth)
        else:
            # Plot each individually
            for traj_file in trajectory_files:
                out = output_path.parent / f"{traj_file.stem}_plot.png" if output_path else None
                plot_single_trajectory(traj_file, out, smooth_sigma=smooth)
    else:
        plot_single_trajectory(input_path, output_path, smooth_sigma=smooth)


if __name__ == "__main__":
    main()
