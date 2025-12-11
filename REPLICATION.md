# Replication Guide

This document provides step-by-step instructions to replicate the Functorial Narrative Analysis research from scratch.

## Prerequisites

- Python 3.10 or higher
- Git
- 8GB RAM minimum (16GB recommended for BERT models)
- ~5GB disk space for corpora and models

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ibrahimcesar/functorial-narrative-analysis.git
cd functorial-narrative-analysis

# Run full replication
make replication
```

## Step-by-Step Replication

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"
```

### 2. Corpus Collection

The research uses two primary corpora:

#### Project Gutenberg (English, Public Domain)

```bash
# Download English literary texts
python -m src.corpus.gutenberg \
    --output data/raw/gutenberg \
    --sample-size 100

# Or use make:
make corpus-gutenberg SAMPLE_SIZE=100
```

**Note:** Project Gutenberg texts are public domain and freely redistributable.

#### Aozora Bunko (Japanese, Public Domain)

```bash
# Download Japanese literary texts
python -m src.corpus.aozora \
    --output data/raw/aozora \
    --sample-size 100
```

### 3. Preprocessing

```bash
# Segment texts into sentences
python -m src.corpus.preprocess segment \
    --input data/raw \
    --output data/processed/segmented

# Create sliding windows
python -m src.corpus.preprocess window \
    --input data/processed/segmented \
    --output data/processed/windowed \
    --window-size 1000 \
    --overlap 500

# Or use make:
make preprocess
```

### 4. Extract Functor Trajectories

Extract all five functor trajectories:

```bash
# Sentiment (F_sentiment)
python -m src.functors.sentiment \
    --input data/processed/windowed/gutenberg \
    --output data/results/trajectories/sentiment/en \
    --method vader

# Arousal (F_arousal)
python -m src.functors.arousal \
    --input data/processed/windowed/gutenberg \
    --output data/results/trajectories/arousal/en

# Entropy (F_entropy)
python -m src.functors.entropy \
    --input data/processed/windowed/gutenberg \
    --output data/results/trajectories/entropy/en

# Or extract all at once:
make extract-trajectories
```

### 5. Structural Detection

Detect narrative structures:

```bash
# Harmon Story Circle
python -m src.detectors.harmon_circle \
    --input data/results/trajectories/sentiment/en \
    --output data/results/structure/harmon

# Kishōtenketsu
python -m src.detectors.kishotenketsu \
    --input data/results/trajectories/sentiment/ja \
    --output data/results/structure/kishotenketsu

# Or use make:
make detect-all
```

### 6. Clustering & Analysis

```bash
# Cluster trajectories by shape
python -m src.clustering.dtw_clustering \
    --input data/results/trajectories \
    --output data/results/clusters \
    --n-clusters 6

# Or use make:
make cluster
make analyze-all
```

### 7. Generate Results

```bash
# Create visualizations
make visualize

# Generate report
make report

# Export results
make export-results
```

## Expected Results

After full replication, you should observe:

### Cross-Cultural Comparisons

| Metric | English (n≈50) | Japanese (n≈50) | p-value |
|--------|----------------|-----------------|---------|
| Mean Sentiment | ~0.15 | ~0.03 | <0.001 |
| Mean Arousal | ~0.52 | ~0.51 | ~0.5 |
| Mean Entropy | ~0.75 | ~0.85 | <0.001 |
| Harmon Conformance | ~0.55 | ~0.42 | <0.01 |
| Kishōtenketsu Conformance | ~0.38 | ~0.52 | <0.01 |

### Trajectory Clustering

- 6 shape clusters (following Reagan et al. 2016)
- Cultural distribution differences across clusters
- Higher Western concentration in "Rags to Riches" and "Man in a Hole"
- Higher Japanese concentration in "Steady" and "Descent" clusters

## Functor Reference

### F_sentiment (Emotional Valence)

Maps narrative states to happiness-sadness axis.

```python
from src.functors import SentimentFunctor

functor = SentimentFunctor(method="vader")
trajectory = functor.process_text(text, window_size=1000)
```

### F_arousal (Tension/Excitement)

Maps narrative states to calm-excited axis.

```python
from src.functors import ArousalFunctor

functor = ArousalFunctor()
trajectory = functor.process_text(text)
```

### F_entropy (Complexity/Predictability)

Maps narrative states to information-theoretic complexity.

```python
from src.functors import EntropyFunctor

functor = EntropyFunctor(method="combined")
trajectory = functor.process_text(text)
```

### F_thematic (Semantic Drift)

Measures thematic coherence and drift over narrative time.

```python
from src.functors import ThematicFunctor

functor = ThematicFunctor()
trajectory = functor(windows)
```

### F_epistemic (Certainty/Uncertainty)

Measures narrative epistemic stance.

```python
from src.functors import EpistemicFunctor

functor = EpistemicFunctor()
trajectory = functor(windows)
```

## Detector Reference

### Harmon Story Circle

Detects 8-stage Western narrative structure.

```python
from src.detectors import HarmonCircleDetector

detector = HarmonCircleDetector()
match = detector.detect(trajectory.values, "id", "Title")
print(f"Conformance: {match.conformance_score:.2f}")
print(f"Pattern: {match.pattern_type}")
```

### Kishōtenketsu

Detects 4-act East Asian narrative structure.

```python
from src.detectors import KishotenketsuDetector

detector = KishotenketsuDetector()
match = detector.detect(trajectory.values, "id", "Title")
print(f"Conformance: {match.conformance_score:.2f}")
print(f"Has twist: {match.has_twist}")
```

## Troubleshooting

### Memory Issues

If you encounter memory errors with BERT models:

```python
# Use VADER only (faster, less memory)
functor = SentimentFunctor(method="vader")
```

### Japanese Text Processing

Ensure proper encoding:

```python
with open(file, 'r', encoding='utf-8') as f:
    text = f.read()
```

### Missing Dependencies

```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

## Data Availability

| Corpus | Status | License | Access |
|--------|--------|---------|--------|
| Project Gutenberg | Included | Public Domain | Direct download |
| Aozora Bunko | Included | Public Domain | Direct download |
| Analysis Results | Included | MIT | `data/results/` |

## Citation

If you use this replication package, please cite:

```bibtex
@software{cesar2024functorial,
  author = {Cesar, Ibrahim},
  title = {Functorial Narrative Analysis: A Category-Theoretic Framework for Cross-Cultural Story Structure},
  year = {2024},
  url = {https://github.com/ibrahimcesar/functorial-narrative-analysis}
}
```

## Contact

For questions about replication:
- Open an issue: https://github.com/ibrahimcesar/functorial-narrative-analysis/issues
- Email: ibrahim@ibrahimcesar.com
