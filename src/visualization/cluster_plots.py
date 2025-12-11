"""
Cluster Visualization for Story Shapes

Creates visualizations showing identified story shapes (centroids)
and the distribution of texts across clusters.
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
import click
from rich.console import Console

from .style import setup_style

console = Console()

# Apply project-wide styling (JetBrains Mono)
setup_style()


def plot_centroids(
    centroids: List[np.ndarray],
    shape_names: List[str],
    output_file: Optional[Path] = None,
    title: str = "Identified Story Shapes"
):
    """
    Plot cluster centroids as story shapes.

    Args:
        centroids: List of centroid arrays
        shape_names: Names for each shape
        output_file: Path to save plot
        title: Plot title
    """
    n_clusters = len(centroids)
    cols = 3
    rows = (n_clusters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.array(axes).flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))

    for i, (centroid, name) in enumerate(zip(centroids, shape_names)):
        ax = axes[i]
        x = np.linspace(0, 1, len(centroid))

        # Smooth for display
        smoothed = gaussian_filter1d(centroid, sigma=2)

        ax.plot(x, smoothed, color=colors[i], linewidth=3)
        ax.fill_between(x, smoothed, alpha=0.3, color=colors[i])
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Narrative Time')
        ax.set_ylabel('Sentiment')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.2)

    # Hide unused subplots
    for i in range(n_clusters, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        console.print(f"[green]Saved centroid plot to {output_file}[/green]")
    else:
        plt.show()

    plt.close()


def plot_cluster_distribution(
    cluster_sizes: List[int],
    shape_names: List[str],
    output_file: Optional[Path] = None
):
    """
    Plot distribution of texts across clusters.

    Args:
        cluster_sizes: Number of texts in each cluster
        shape_names: Names for each shape
        output_file: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(cluster_sizes)))

    bars = ax.barh(shape_names, cluster_sizes, color=colors)

    # Add value labels
    for bar, size in zip(bars, cluster_sizes):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{size}', va='center', fontsize=10)

    ax.set_xlabel('Number of Texts', fontsize=12)
    ax.set_title('Distribution of Story Shapes', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        console.print(f"[green]Saved distribution plot to {output_file}[/green]")
    else:
        plt.show()

    plt.close()


def plot_all_trajectories_by_cluster(
    trajectories_dir: Path,
    clustering_results: dict,
    output_file: Optional[Path] = None
):
    """
    Plot all trajectories colored by cluster assignment.

    Args:
        trajectories_dir: Directory with trajectory files
        clustering_results: Clustering results dict
        output_file: Path to save plot
    """
    n_clusters = clustering_results["n_clusters"]
    labels = np.array(clustering_results["labels"])
    trajectory_ids = clustering_results["trajectory_ids"]
    shape_names = clustering_results["shape_names"]

    # Load all trajectories
    trajectories = {}
    for traj_file in sorted(trajectories_dir.glob("*_sentiment.json")):
        with open(traj_file, 'r') as f:
            data = json.load(f)
        traj_id = data["metadata"].get("source_id", traj_file.stem.replace("_sentiment", ""))
        trajectories[traj_id] = np.array(data["values"])

    # Create subplots
    cols = 3
    rows = (n_clusters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = np.array(axes).flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))

    for k in range(n_clusters):
        ax = axes[k]
        cluster_mask = labels == k
        cluster_ids = [tid for tid, m in zip(trajectory_ids, cluster_mask) if m]

        for tid in cluster_ids:
            if tid in trajectories:
                traj = trajectories[tid]
                # Normalize and smooth
                traj_norm = (traj - traj.min()) / (traj.max() - traj.min() + 1e-8)
                traj_smooth = gaussian_filter1d(traj_norm, sigma=3)
                x = np.linspace(0, 1, len(traj_smooth))
                ax.plot(x, traj_smooth, alpha=0.3, color=colors[k], linewidth=0.8)

        # Plot centroid
        centroid = np.array(clustering_results["centroids"][k])
        x = np.linspace(0, 1, len(centroid))
        ax.plot(x, centroid, color='black', linewidth=3, label='Centroid')

        ax.set_title(f"{shape_names[k]}\n(n={sum(cluster_mask)})", fontsize=10)
        ax.set_xlabel('Narrative Time')
        ax.set_ylabel('Sentiment (normalized)')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.2)

    # Hide unused subplots
    for i in range(n_clusters, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Story Shapes: All Trajectories by Cluster', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        console.print(f"[green]Saved cluster trajectories plot to {output_file}[/green]")
    else:
        plt.show()

    plt.close()


def plot_reagan_comparison(
    centroids: List[np.ndarray],
    shape_names: List[str],
    output_file: Optional[Path] = None
):
    """
    Create a 2x3 plot matching Reagan et al.'s six story shapes layout.

    The six shapes from Reagan et al. (2016):
    1. Rags to Riches (rise)
    2. Riches to Rags (fall)
    3. Man in a Hole (fall-rise)
    4. Icarus (rise-fall)
    5. Cinderella (rise-fall-rise)
    6. Oedipus (fall-rise-fall)
    """
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig)

    # Reagan's canonical shapes (idealized)
    reagan_shapes = {
        "Rags to Riches": lambda x: x,
        "Riches to Rags": lambda x: 1 - x,
        "Man in a Hole": lambda x: 4 * (x - 0.5) ** 2,
        "Icarus": lambda x: 1 - 4 * (x - 0.5) ** 2,
        "Cinderella": lambda x: 0.5 + 0.5 * np.sin(2 * np.pi * x - np.pi/2),
        "Oedipus": lambda x: 0.5 - 0.5 * np.sin(2 * np.pi * x - np.pi/2),
    }

    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    colors = plt.cm.Set2(np.linspace(0, 1, 6))

    for idx, (canon_name, shape_func) in enumerate(reagan_shapes.items()):
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])

        x = np.linspace(0, 1, 100)

        # Plot canonical shape (dashed)
        canon_y = shape_func(x)
        ax.plot(x, canon_y, '--', color='gray', linewidth=2, label='Reagan et al.', alpha=0.7)

        # Find best matching centroid
        best_match = None
        best_score = -np.inf
        for i, centroid in enumerate(centroids):
            # Resample centroid to match
            centroid_resampled = np.interp(x, np.linspace(0, 1, len(centroid)), centroid)
            # Correlation
            corr = np.corrcoef(canon_y, centroid_resampled)[0, 1]
            if corr > best_score:
                best_score = corr
                best_match = (i, centroid_resampled, shape_names[i])

        if best_match:
            i, matched_centroid, matched_name = best_match
            ax.plot(x, matched_centroid, color=colors[i], linewidth=2.5,
                    label=f'Ours: {matched_name[:20]}...' if len(matched_name) > 20 else f'Ours: {matched_name}')
            ax.fill_between(x, matched_centroid, alpha=0.2, color=colors[i])

        ax.set_title(f'{canon_name}\n(r={best_score:.2f})', fontsize=11)
        ax.set_xlabel('Narrative Time')
        ax.set_ylabel('Sentiment')
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(0, 1)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.suptitle('Comparison with Reagan et al. Six Story Shapes', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        console.print(f"[green]Saved Reagan comparison to {output_file}[/green]")
    else:
        plt.show()

    plt.close()


@click.command()
@click.option('--results', '-r', 'results_dir', required=True, type=click.Path(exists=True),
              help='Clustering results directory')
@click.option('--trajectories', '-t', 'trajectories_dir', required=True, type=click.Path(exists=True),
              help='Trajectories directory')
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(),
              help='Output directory for plots')
def main(results_dir: str, trajectories_dir: str, output_dir: str):
    """Generate cluster visualization plots."""
    results_dir = Path(results_dir)
    trajectories_dir = Path(trajectories_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    with open(results_dir / "clustering_results.json", 'r') as f:
        results = json.load(f)

    with open(results_dir / "centroids.json", 'r') as f:
        centroid_data = json.load(f)

    centroids = [np.array(c) for c in centroid_data["centroids"]]
    shape_names = centroid_data["shape_names"]

    # Generate plots
    plot_centroids(centroids, shape_names, output_dir / "story_shapes.png")
    plot_cluster_distribution(results["cluster_sizes"], shape_names, output_dir / "distribution.png")
    plot_all_trajectories_by_cluster(trajectories_dir, results, output_dir / "trajectories_by_cluster.png")
    plot_reagan_comparison(centroids, shape_names, output_dir / "reagan_comparison.png")

    console.print(f"[bold green]✓ All plots saved to {output_dir}[/bold green]")


if __name__ == "__main__":
    main()
