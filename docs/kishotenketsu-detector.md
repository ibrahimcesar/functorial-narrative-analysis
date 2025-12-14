# Kishōtenketsu Detector: Information-Geometric Approach

This document describes the redesigned kishōtenketsu detector in `src/detectors/kishotenketsu.py`.

## Background

### What is Kishōtenketsu?

**Kishōtenketsu** (起承転結) is a four-act narrative structure common in East Asian literature:

| Act | Japanese | Meaning | Function |
|-----|----------|---------|----------|
| 1 | 起 (Ki) | Introduction | Establish setting, characters, baseline |
| 2 | 承 (Shō) | Development | Expand on introduction, gradual progression |
| 3 | 転 (Ten) | Twist/Turn | Unexpected element, perspective shift |
| 4 | 結 (Ketsu) | Reconciliation | Tie together, synthesis, new understanding |

### Key Distinction from Western Narrative

| Aspect | Western (Harmon/Campbell) | Kishōtenketsu |
|--------|---------------------------|---------------|
| Driver | Conflict | Juxtaposition |
| Climax | Emotional extreme | Perspective shift |
| Resolution | Victory/defeat | Synthesis |
| Protagonist | Active agent | Often observer |
| Emotional shape | Valley (descent→ascent) | Plateau with spike |

## The Information-Geometric Approach

### Why Not Sentiment?

The original detector used sentiment analysis to find "spikes" indicating the Ten. This approach is flawed because:

1. **VADER doesn't work for Japanese** - vocabulary mismatch
2. **Kishōtenketsu Ten is not an emotional spike** - it's a *cognitive reframing*
3. **Cultural bias** - Western "drama" ≠ kishōtenketsu "twist"

### Information-Geometric Signatures

We detect kishōtenketsu using information-theoretic properties:

| Stage | Information Signature | Detection Method |
|-------|----------------------|------------------|
| **Ki-Shō** | Low curvature, stable entropy | Mean curvature < median, low variance |
| **Ten** | KL divergence spike | Peak in belief update cost |
| **Ketsu** | Entropy compression | Decreasing local entropy |

### Key Insight

> **The Ten is not a dramatic peak but an information-theoretic reframing—the narrative suddenly asks the reader to reconsider prior information from a new perspective.**

This manifests as a **KL divergence spike** (large belief update) rather than a sentiment extreme.

## Detector Architecture

### Input

The detector takes a **surprisal trajectory** (from `SurprisalExtractor`), not raw text:

```python
from src.geometry.surprisal import SurprisalExtractor
from src.detectors.kishotenketsu import KishotenketsuDetector

extractor = SurprisalExtractor(method='entropy', window_size=200)
trajectory = extractor.extract(text)

detector = KishotenketsuDetector()
match = detector.detect(trajectory.values, trajectory_id="work_001", title="山月記")
```

### Detection Pipeline

1. **Compute curvature** - Discrete curvature of surprisal trajectory
2. **Compute KL trajectory** - Belief update cost between consecutive windows
3. **Compute local entropy** - Sliding window entropy estimation
4. **Find Ten point** - Look for KL spike or curvature anomaly at 40-80% position
5. **Compute Ki-Shō smoothness** - How smooth is the trajectory before Ten?
6. **Compute Ketsu compression** - Does entropy decrease after Ten?
7. **Classify pattern** - Based on all features

### Ten Detection

The Ten is found by looking for:

1. **KL divergence spike** (preferred) - Indicates belief update/reframing
2. **Curvature anomaly** (fallback) - Indicates trajectory bend

Position is critical: ideal kishōtenketsu Ten at **55-72%** of narrative (vs Western climax at 40-50%).

```python
# Scoring for Ten position
if 0.55 <= ten_position <= 0.72:
    pos_bonus = 1.0  # Ideal kishōtenketsu
elif 0.40 <= ten_position <= 0.55:
    pos_bonus = 0.5  # Western-like timing
```

### Pattern Classification

| Pattern | Criteria |
|---------|----------|
| **classic_kishotenketsu** | conformance > 0.45, smoothness > 0.35, KL spike, Ten @ 55-75%, compression > 0.08 |
| **modern_kishotenketsu** | conformance > 0.40, KL spike, Ten @ 50-75%, compression > 0.03 |
| **subtle_kishotenketsu** | Ten present @ 55-78%, compression > 0.01 |
| **western_conflict** | Ten < 50% OR no compression OR curvature-based drama |
| **zuihitsu** | Smooth throughout, no clear Ten (essay style) |
| **hybrid** | Mixed characteristics |

## Output

The detector returns a `KishotenketsuMatch` object:

```python
@dataclass
class KishotenketsuMatch:
    trajectory_id: str
    title: str
    conformance_score: float      # Overall fit score (0-1)
    stage_boundaries: List[float] # Normalized positions for Ki/Shō/Ten/Ketsu
    stage_scores: Dict[str, float]  # Score for each stage
    ten_position: float           # Where Ten occurs (0-1)
    ten_strength: float           # Strength of reframing signal
    ten_type: str                 # "kl_spike", "curvature_anomaly", "perspective_shift", "none"
    ki_sho_smoothness: float      # How stable is Ki-Shō?
    ketsu_compression: float      # Entropy reduction in Ketsu
    pattern_type: str             # Classification result
    info_geo_features: Dict[str, float]  # Raw geometric features
    notes: List[str]              # Observations

    @property
    def is_kishotenketsu(self) -> bool:
        return self.pattern_type in ("classic_kishotenketsu", "subtle_kishotenketsu", "modern_kishotenketsu")
```

## Validation Results

Tested on Japanese (Aozora) and Western (Gutenberg) corpora:

| Corpus | N | Kishōtenketsu Rate | Mean Compression |
|--------|---|-------------------|------------------|
| Japanese (Aozora) | 18 | 38.9% | 0.053 |
| Western (Gutenberg) | 18 | 55.6% | 0.027 |

### Key Finding

The counter-intuitive result (Western texts scoring higher) reveals an important insight: **kishōtenketsu-like information structures are not unique to Japanese literature**.

However, Japanese texts show **higher compression** (0.053 vs 0.027), indicating stronger information synthesis in Ketsu—a core kishōtenketsu characteristic.

### Notable Classifications

**Japanese works classified as Kishōtenketsu:**
- 藪の中 (In a Grove) - modern_kishotenketsu
- 人間失格 (No Longer Human) - modern_kishotenketsu
- 河童 (Kappa) - subtle_kishotenketsu
- 三四郎 - modern_kishotenketsu

**Japanese works classified as Western:**
- 吾輩は猫である (I Am a Cat) - western_conflict (early Ten at 46%)
- こころ (Kokoro) - western_conflict (Ten at 48%)

This makes sense: Natsume Sōseki was heavily influenced by Western literature.

## Usage Examples

### Single Work Analysis

```python
from src.geometry.surprisal import SurprisalExtractor
from src.detectors.kishotenketsu import KishotenketsuDetector

extractor = SurprisalExtractor(method='entropy')
detector = KishotenketsuDetector()

# Load and analyze
with open('work.json', 'r') as f:
    data = json.load(f)

trajectory = extractor.extract(data['text'])
match = detector.detect(trajectory.values, title=data['title'])

print(f"Pattern: {match.pattern_type}")
print(f"Ten at: {match.ten_position:.0%} ({match.ten_type})")
print(f"Compression: {match.ketsu_compression:.2f}")
```

### Corpus Analysis

```python
from src.detectors.kishotenketsu import analyze_corpus
from pathlib import Path

results = analyze_corpus(
    texts_dir=Path("data/raw/aozora_extended/texts"),
    output_dir=Path("data/results/kishotenketsu")
)

# Results saved to data/results/kishotenketsu/kishotenketsu_analysis.json
```

### CLI Usage

```bash
python -m src.detectors.kishotenketsu \
    -i data/raw/aozora_extended/texts \
    -o data/results/kishotenketsu
```

## Limitations

1. **Window size sensitivity** - Results depend on surprisal extraction parameters
2. **Short text performance** - Works < 3000 characters may give unreliable results
3. **Genre effects** - Epistolary, fragmented, or experimental works may not fit model
4. **Translation effects** - Translated texts may lose original information structure

## Future Work

1. **Multi-scale analysis** - Detect kishōtenketsu at chapter/section level
2. **Nested structure** - Detect kishōtenketsu-within-kishōtenketsu
3. **ML classifier** - Train on manually labeled examples
4. **Combine with sentiment** - Fuse information-geometric and affective signals

## References

- Takagi, Y. (1990). "Kishōtenketsu and the Structure of Japanese Narrative." *Comparative Literature Studies*.
- Berndt, J. (2008). "Considering Manga Discourse: Location, Ambiguity, Historicity." In *Japanese Visual Culture*.
