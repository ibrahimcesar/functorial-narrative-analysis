# Information Geometry for Narrative Analysis

This document describes the information-geometric approach to narrative analysis implemented in the `src/geometry/` module.

## Overview

Information geometry treats narratives as **trajectories through statistical manifolds**. Instead of measuring sentiment (which is culturally biased), we measure how information flows through the text—surprisal, entropy, and belief updates.

This approach enables cross-cultural comparison because information-theoretic quantities are language-independent.

## Core Concepts

### Surprisal

**Surprisal** (or self-information) measures how unexpected each piece of text is:

```
I(x) = -log P(x | context)
```

High surprisal = unexpected information. Low surprisal = predictable continuation.

We compute surprisal trajectories using character/word-level entropy estimation:

```python
from src.geometry.surprisal import SurprisalExtractor

extractor = SurprisalExtractor(method='entropy', window_size=200)
trajectory = extractor.extract(text)

# trajectory.positions: normalized positions [0, 1]
# trajectory.values: surprisal values at each position
```

### Curvature

**Curvature** measures how sharply the narrative trajectory bends:

```
κ = |d²x/dt²| / (1 + (dx/dt)²)^(3/2)
```

High curvature = dramatic turns in information space.
Low curvature = smooth, gradual development.

```python
from src.geometry.curvature import NarrativeCurvature

analyzer = NarrativeCurvature()
features = analyzer.extract_features(trajectory.values, trajectory.positions)

# features.mean_curvature
# features.max_curvature
# features.n_peaks
# features.arc_length
# features.skewness
# features.kurtosis
# features.entropy_change
```

### KL Divergence

**KL Divergence** measures the "cost" of updating beliefs between narrative states:

```
D_KL(P || Q) = Σ P(x) log(P(x) / Q(x))
```

A spike in KL divergence indicates a **belief update**—the reader must revise their model of the story.

```python
from src.geometry.divergence import KLDivergenceAnalyzer

kl_analyzer = KLDivergenceAnalyzer()
belief_trajectory = kl_analyzer.compute_trajectory(text, n_states=20)

# Detect narrative twists
twists = kl_analyzer.detect_twists(belief_trajectory, threshold_sigma=2.0)
```

## Information-Geometric Shape Classification

Based on empirical analysis of 125 works across 5 corpora, we classify narratives into five **information-geometric shapes**:

| Shape | Description | Information Signature |
|-------|-------------|----------------------|
| **Geodesic Tragedy** | Smooth descent through information space | Low curvature, decreasing entropy, negative skew |
| **High-Curvature Mystery** | Sustained high information rate | High mean curvature, many peaks, sustained engagement |
| **Random Walk Comedy** | Oscillating, mean-reverting trajectory | High variance, many peaks, near-zero net change |
| **Compression Progress** | Steady entropy reduction | Decreasing entropy, low variance, learning/resolution |
| **Discontinuous Twist** | Late curvature spike | Late peaks, high max curvature, surprise ending |

### Usage

```python
from src.geometry.curvature import classify_information_shape

features = analyzer.extract_features(trajectory.values, trajectory.positions)
shape_scores = classify_information_shape(features)

# shape_scores = {
#     'geodesic_tragedy': 0.15,
#     'high_curvature_mystery': 0.45,
#     'random_walk_comedy': 0.20,
#     'compression_progress': 0.10,
#     'discontinuous_twist': 0.10,
# }
```

### Empirical Calibration

Shape classification thresholds were calibrated from corpus analysis (n=125 works):

| Metric | p25 | p50 | p75 |
|--------|-----|-----|-----|
| mean_curvature | 52 | 64 | 72 |
| max_curvature | 699 | 988 | 1337 |
| n_peaks | 7 | 11 | 13 |
| entropy_change | -0.08 | 0.0 | 0.10 |

## Cross-Cultural Findings

Analysis comparing Japanese (Aozora) and Western (Gutenberg) literature revealed:

| Metric | Japanese | Western | Interpretation |
|--------|----------|---------|----------------|
| Mean surprisal | 5.4 | 6.3 | Japanese texts more predictable locally |
| Mean curvature | 15.8 | 70.7 | Japanese narratives smoother |
| KL divergence | 1.74 bits | 0.85 bits | Japanese has larger belief updates |
| Goldilocks score | 0.74 | 0.98 | Western closer to "optimal engagement" zone |

This suggests Japanese narratives operate differently in information space—smoother local progression but larger discrete belief updates, consistent with kishōtenketsu structure.

## Goldilocks Principle

The **Goldilocks Principle** (Schmidhuber, 2009) suggests optimal engagement occurs at moderate surprise levels—not too boring (low KL), not too confusing (high KL).

We compute Goldilocks scores as proximity to ~1 bit of KL divergence per narrative state:

```python
goldilocks_score = exp(-|mean_kl - 1.0|)
```

## Visualizations

The `src/visualization/geometry_plots.py` module provides:

- `plot_trajectory_with_curvature()`: Trajectory colored by local curvature
- `plot_shape_archetype()`: Idealized shape trajectories
- `plot_shape_distribution()`: Cross-corpus shape comparison
- `plot_geometric_profile()`: Radar chart of geometric features

### 3D Visualizations

For interactive exploration:

- **Trajectory space**: Position × Surprisal × Curvature
- **Phase space**: Surprisal × Velocity × Curvature
- **Feature space**: Works plotted by geometric features
- **Manifold surface**: Curvature-weighted information density

See `assets/geometry/3d/` for generated visualizations.

## References

- Amari, S. (2016). *Information Geometry and Its Applications*. Springer.
- Schmidhuber, J. (2009). "Simple Algorithmic Theory of Subjective Beauty, Novelty, Surprise, Interestingness, Attention, Curiosity, Creativity, Art, Science, Music, Jokes." *JAGI* 1(1).
- Reagan, A. J., et al. (2016). "The emotional arcs of stories are dominated by six basic shapes." *EPJ Data Science* 5(1).
