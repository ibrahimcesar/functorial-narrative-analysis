"""
Visualizations for Information-Geometric Narrative Shapes.

Creates plots showing:
    - Surprisal trajectories with curvature coloring
    - Shape archetype comparisons
    - Cross-corpus shape distributions
    - Individual work geometric profiles
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches


# Style configuration
SHAPE_COLORS = {
    'geodesic_tragedy': '#8B0000',      # Dark red - inevitable fall
    'high_curvature_mystery': '#4169E1', # Royal blue - constant revelation
    'random_walk_comedy': '#FFD700',     # Gold - oscillating fortune
    'compression_progress': '#228B22',   # Forest green - learning/resolution
    'discontinuous_twist': '#9932CC',    # Purple - surprise ending
}

SHAPE_LABELS = {
    'geodesic_tragedy': 'Geodesic Tragedy',
    'high_curvature_mystery': 'High-Curvature Mystery',
    'random_walk_comedy': 'Random Walk Comedy',
    'compression_progress': 'Compression Progress',
    'discontinuous_twist': 'Discontinuous Twist',
}


def plot_trajectory_with_curvature(
    values: np.ndarray,
    positions: np.ndarray,
    curvature: np.ndarray,
    title: str = "Narrative Trajectory",
    ax: Optional[plt.Axes] = None,
    cmap: str = 'plasma',
) -> plt.Axes:
    """
    Plot trajectory colored by local curvature.

    High curvature = dramatic turns in information space.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    # Create line segments
    points = np.array([positions, values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color by curvature
    norm = Normalize(vmin=np.percentile(curvature, 5),
                     vmax=np.percentile(curvature, 95))
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.5)
    lc.set_array(curvature[:-1])

    ax.add_collection(lc)
    ax.set_xlim(0, 1)
    ax.set_ylim(values.min() - 0.1, values.max() + 0.1)

    # Colorbar
    cbar = plt.colorbar(lc, ax=ax, label='Curvature (κ)')

    ax.set_xlabel('Narrative Position', fontsize=11)
    ax.set_ylabel('Surprisal (bits)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    return ax


def plot_shape_archetype(
    shape_name: str,
    n_points: int = 100,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot idealized shape archetype trajectory.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    t = np.linspace(0, 1, n_points)

    if shape_name == 'geodesic_tragedy':
        # Smooth descent - Greek tragedy
        y = 1.0 - 0.8 * t + 0.1 * np.sin(2 * np.pi * t)

    elif shape_name == 'high_curvature_mystery':
        # Sustained high curvature with multiple peaks
        y = 0.5 + 0.3 * np.sin(6 * np.pi * t) + 0.15 * np.sin(10 * np.pi * t)

    elif shape_name == 'random_walk_comedy':
        # Oscillating, mean-reverting
        np.random.seed(42)
        y = 0.5 + 0.3 * np.cumsum(np.random.randn(n_points)) / n_points
        y = (y - y.mean()) / y.std() * 0.3 + 0.5

    elif shape_name == 'compression_progress':
        # Steady decrease in entropy/uncertainty
        y = 0.8 * np.exp(-2 * t) + 0.2 + 0.05 * np.sin(4 * np.pi * t)

    elif shape_name == 'discontinuous_twist':
        # Relatively flat then late spike
        y = 0.4 + 0.1 * np.sin(2 * np.pi * t)
        # Add late spike
        spike_center = 0.85
        spike = 0.5 * np.exp(-((t - spike_center) / 0.05) ** 2)
        y = y + spike
    else:
        y = np.zeros(n_points)

    color = SHAPE_COLORS.get(shape_name, '#333333')
    label = SHAPE_LABELS.get(shape_name, shape_name)

    ax.plot(t, y, color=color, linewidth=2.5, label=label)
    ax.fill_between(t, y, alpha=0.2, color=color)

    ax.set_xlim(0, 1)
    ax.set_xlabel('Narrative Position', fontsize=10)
    ax.set_ylabel('Information Content', fontsize=10)
    ax.set_title(label, fontsize=12, fontweight='bold', color=color)
    ax.grid(True, alpha=0.3)

    return ax


def plot_all_archetypes(
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot all five shape archetypes side by side.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    shapes = [
        'geodesic_tragedy',
        'compression_progress',
        'random_walk_comedy',
        'high_curvature_mystery',
        'discontinuous_twist',
    ]

    for ax, shape in zip(axes, shapes):
        plot_shape_archetype(shape, ax=ax)

    fig.suptitle('Information-Geometric Narrative Shapes', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    return fig


def plot_shape_distribution(
    distributions: Dict[str, Dict[str, int]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot shape distributions across multiple corpora.

    Args:
        distributions: {corpus_name: {shape_name: count}}
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    corpora = list(distributions.keys())
    shapes = list(SHAPE_COLORS.keys())

    x = np.arange(len(corpora))
    width = 0.15

    for i, shape in enumerate(shapes):
        counts = []
        for corpus in corpora:
            corpus_dist = distributions[corpus]
            total = sum(corpus_dist.values())
            count = corpus_dist.get(shape, 0)
            pct = 100 * count / total if total > 0 else 0
            counts.append(pct)

        offset = (i - len(shapes) / 2 + 0.5) * width
        bars = ax.bar(x + offset, counts, width,
                      label=SHAPE_LABELS[shape],
                      color=SHAPE_COLORS[shape],
                      edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Corpus', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Shape Distribution by Corpus', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(corpora, fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    return fig


def plot_geometric_profile(
    features: Dict[str, float],
    shape_scores: Dict[str, float],
    title: str = "Geometric Profile",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot radar chart of geometric features and shape scores.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    subplot_kw={'projection': 'polar'})

    # Feature radar
    feature_names = ['mean_curvature', 'n_peaks', 'arc_length',
                     'skewness', 'kurtosis', 'entropy_change']
    feature_labels = ['Mean κ', 'Peaks', 'Arc Length',
                      'Skewness', 'Kurtosis', 'ΔEntropy']

    # Normalize features to 0-1 range for radar
    normalized = []
    for name in feature_names:
        val = features.get(name, 0)
        # Simple min-max normalization with reasonable bounds
        if name == 'mean_curvature':
            val = min(1, val / 100)
        elif name == 'n_peaks':
            val = min(1, val / 15)
        elif name == 'arc_length':
            val = min(1, val / 10)
        elif name == 'skewness':
            val = (val + 3) / 6  # Assume range -3 to 3
        elif name == 'kurtosis':
            val = min(1, (val + 2) / 10)  # Assume range -2 to 8
        elif name == 'entropy_change':
            val = (val + 0.5) / 1  # Assume range -0.5 to 0.5
        normalized.append(max(0, min(1, val)))

    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
    normalized += normalized[:1]  # Close the polygon
    angles += angles[:1]

    ax1.plot(angles, normalized, 'o-', linewidth=2, color='#2E86AB')
    ax1.fill(angles, normalized, alpha=0.25, color='#2E86AB')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(feature_labels, fontsize=9)
    ax1.set_title('Geometric Features', fontsize=12, fontweight='bold', pad=20)

    # Shape scores radar
    shape_names = list(SHAPE_COLORS.keys())
    shape_values = [shape_scores.get(s, 0) for s in shape_names]
    shape_labels_short = ['Tragedy', 'Mystery', 'Comedy', 'Progress', 'Twist']

    angles2 = np.linspace(0, 2 * np.pi, len(shape_names), endpoint=False).tolist()
    shape_values += shape_values[:1]
    angles2 += angles2[:1]

    # Color by dominant shape
    best_shape = max(shape_scores, key=shape_scores.get)
    color = SHAPE_COLORS[best_shape]

    ax2.plot(angles2, shape_values, 'o-', linewidth=2, color=color)
    ax2.fill(angles2, shape_values, alpha=0.25, color=color)
    ax2.set_xticks(angles2[:-1])
    ax2.set_xticklabels(shape_labels_short, fontsize=9)
    ax2.set_title(f'Shape Scores\n(Best: {SHAPE_LABELS[best_shape]})',
                  fontsize=12, fontweight='bold', pad=20, color=color)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    return fig


def plot_corpus_comparison(
    corpus_data: Dict[str, Dict],
    metric: str = 'mean_curvature',
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Box plot comparing a metric across corpora.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    labels = []

    for corpus_name, corpus_info in corpus_data.items():
        works = corpus_info.get('works', [])
        values = []
        for work in works:
            gf = work.get('geometric_features', {})
            if metric in gf:
                values.append(gf[metric])
        if values:
            data.append(values)
            labels.append(corpus_name)

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        # Color boxes
        colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_xlabel('Corpus', fontsize=12)
    ax.set_title(f'{metric.replace("_", " ").title()} by Corpus',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    return fig


def plot_kl_trajectory(
    kl_values: np.ndarray,
    positions: np.ndarray,
    twists: Optional[List[Dict]] = None,
    title: str = "KL Divergence Trajectory",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None,
) -> plt.Axes:
    """
    Plot KL divergence trajectory with twist markers.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    # Main trajectory
    ax.plot(positions, kl_values, color='#2E86AB', linewidth=2, label='KL Divergence')
    ax.fill_between(positions, kl_values, alpha=0.2, color='#2E86AB')

    # Goldilocks zone (optimal engagement ~0.5-1.5 bits)
    ax.axhspan(0.5, 1.5, alpha=0.1, color='green', label='Goldilocks Zone')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1)

    # Mark twists
    if twists:
        for twist in twists:
            pos = twist.get('position', 0)
            kl = twist.get('kl_divergence', 0)
            ax.scatter([pos], [kl], color='#9932CC', s=100, zorder=5,
                      marker='*', edgecolor='white', linewidth=1)
            ax.annotate('twist', (pos, kl), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=8, color='#9932CC')

    ax.set_xlim(0, 1)
    ax.set_xlabel('Narrative Position', fontsize=11)
    ax.set_ylabel('KL Divergence (bits)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    return ax


def create_shape_legend() -> plt.Figure:
    """
    Create a standalone legend explaining all shapes.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    descriptions = {
        'geodesic_tragedy': 'Smooth descent through information space\n(Greek tragedy, inevitable fall)',
        'high_curvature_mystery': 'Sustained high information rate\n(Detective fiction, constant revelation)',
        'random_walk_comedy': 'Oscillating, mean-reverting trajectory\n(Comedy, picaresque, fortune\'s wheel)',
        'compression_progress': 'Steady entropy reduction\n(Bildungsroman, mystery resolution)',
        'discontinuous_twist': 'Late curvature spike\n(O. Henry, surprise ending)',
    }

    y_pos = 0.9
    for shape, color in SHAPE_COLORS.items():
        label = SHAPE_LABELS[shape]
        desc = descriptions[shape]

        # Color patch
        rect = mpatches.Rectangle((0.02, y_pos - 0.08), 0.04, 0.12,
                                   facecolor=color, edgecolor='white')
        ax.add_patch(rect)

        # Text
        ax.text(0.08, y_pos, label, fontsize=11, fontweight='bold',
                color=color, va='center')
        ax.text(0.08, y_pos - 0.08, desc, fontsize=9, color='#555555', va='top')

        y_pos -= 0.2

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Information-Geometric Narrative Shapes',
                 fontsize=14, fontweight='bold', y=0.98)

    return fig


if __name__ == "__main__":
    # Generate example visualizations
    import json

    output_dir = Path("assets/geometry")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. All archetypes
    print("Generating archetype plots...")
    fig = plot_all_archetypes(save_path=output_dir / "shape_archetypes.png")
    plt.close(fig)

    # 2. Legend
    print("Generating shape legend...")
    fig = create_shape_legend()
    fig.savefig(output_dir / "shape_legend.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # 3. Load corpus data and create comparison plots
    results_file = Path("data/results/information_geometry/corpus_analysis.json")
    if results_file.exists():
        print("Loading corpus data...")
        with open(results_file, 'r') as f:
            corpus_data = json.load(f)

        # Shape distributions
        distributions = {}
        for corpus_name, data in corpus_data.items():
            distributions[corpus_name] = data.get('shape_distribution', {})

        if distributions:
            print("Generating distribution plot...")
            fig = plot_shape_distribution(distributions,
                                          save_path=output_dir / "shape_distribution.png")
            plt.close(fig)

        # Metric comparisons
        for metric in ['mean_curvature', 'arc_length', 'n_peaks']:
            print(f"Generating {metric} comparison...")
            fig = plot_corpus_comparison(corpus_data, metric=metric,
                                         save_path=output_dir / f"{metric}_comparison.png")
            plt.close(fig)

    print(f"\nVisualizations saved to {output_dir}/")
