"""
Visualization module for Functorial Narrative Analysis.

Provides consistent styling with JetBrains Mono font across all visualizations.
"""

from .style import setup_style, get_sentiment_color, get_harmon_color, COLORS
from .plot_trajectory import plot_single_trajectory, plot_comparison
from .cluster_plots import (
    plot_centroids,
    plot_cluster_distribution,
    plot_all_trajectories_by_cluster,
    plot_reagan_comparison,
)

__all__ = [
    "setup_style",
    "get_sentiment_color",
    "get_harmon_color",
    "COLORS",
    "plot_single_trajectory",
    "plot_comparison",
    "plot_centroids",
    "plot_cluster_distribution",
    "plot_all_trajectories_by_cluster",
    "plot_reagan_comparison",
]
