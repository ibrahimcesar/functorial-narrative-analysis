# Functorial Narrative Analysis - Documentation

## Overview

This directory contains detailed documentation for the Functorial Narrative Analysis project.

## Documents

### Core Modules

| Document | Description |
|----------|-------------|
| [information-geometry.md](information-geometry.md) | Information-geometric approach to narrative analysis: surprisal, curvature, KL divergence, and shape classification |
| [kishotenketsu-detector.md](kishotenketsu-detector.md) | Redesigned kishōtenketsu detector using information geometry |

### Quick Links

- **Main README**: [../README.md](../README.md) - Project overview, theory, and research design
- **Replication Guide**: [../REPLICATION.md](../REPLICATION.md) - How to replicate experiments
- **Contributing**: [../CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute

## Module Structure

```
src/
├── geometry/                    # Information geometry module
│   ├── surprisal.py            # Surprisal extraction
│   ├── curvature.py            # Curvature and shape classification
│   └── divergence.py           # KL divergence analysis
│
├── detectors/                   # Narrative structure detectors
│   ├── kishotenketsu.py        # Kishōtenketsu (info-geometric)
│   └── harmon.py               # Harmon Story Circle
│
├── functors/                    # Analysis functors
│   ├── sentiment.py            # Sentiment functor (VADER)
│   ├── japanese_sentiment.py   # Japanese sentiment (SentiStrength)
│   └── entropy.py              # Entropy functor
│
├── visualization/               # Plotting utilities
│   └── geometry_plots.py       # 2D/3D geometry visualizations
│
└── corpus/                      # Corpus loaders
    ├── gutenberg.py            # Project Gutenberg
    └── aozora.py               # Aozora Bunko (Japanese)
```

## Key Concepts

### Information Geometry

Instead of measuring sentiment (culturally biased), we measure how information flows:

- **Surprisal**: How unexpected each part of the text is
- **Curvature**: How sharply the narrative "turns" in information space
- **KL Divergence**: The cost of updating beliefs between narrative states

### Shape Classification

Five information-geometric narrative shapes (calibrated on n=125 works):

1. **Geodesic Tragedy** - Smooth descent, low curvature
2. **High-Curvature Mystery** - Sustained high information rate
3. **Random Walk Comedy** - Oscillating, mean-reverting
4. **Compression Progress** - Steady entropy reduction
5. **Discontinuous Twist** - Late curvature spike

### Kishōtenketsu Detection

The kishōtenketsu detector identifies the four-act East Asian structure using:

- **Ki-Shō smoothness**: Low curvature in first half
- **Ten detection**: KL divergence spike at 55-75% position
- **Ketsu compression**: Entropy reduction in final act

## Generated Assets

Visualizations are saved to:

```
assets/
└── geometry/
    ├── shape_archetypes.png       # Five shape templates
    ├── shape_distribution.png     # Cross-corpus comparison
    ├── 3d/                        # 3D visualizations
    │   ├── trajectory_3d.png
    │   ├── phase_space_3d.png
    │   └── ...
    └── tolstoy/                   # Per-author analysis
        ├── War_and_Peace_trajectory.png
        └── ...
```

## Analysis Results

Results are saved to:

```
data/results/
├── information_geometry/
│   └── corpus_analysis.json      # Full corpus analysis
└── kishotenketsu/
    └── kishotenketsu_analysis.json
```
