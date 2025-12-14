# Information Complexity Classes (ICC) Model

## Overview

The ICC model is a **data-driven, falsifiable** narrative classification system developed from unsupervised pattern discovery on 2,600+ fiction texts. Unlike traditional Western narrative models (Aristotle, Freytag, Campbell, Harmon), ICC uses specific numerical thresholds that can be empirically tested and falsified.

## ICC Class Names at a Glance

| Class | Name | Full Name | Cultural Prediction |
|-------|------|-----------|---------------------|
| **ICC-0** | Wandering Mist | The Uncharted Path | Neutral |
| **ICC-1** | Quiet Ascent | The Contemplative Rise | Japanese-typical |
| **ICC-2** | Gentle Descent | The Elegiac Fall | Japanese-typical |
| **ICC-3** | Eternal Return | The Cyclical Journey | Western-typical |
| **ICC-4** | Triumphant Climb | The Dramatic Ascent | Western-typical |
| **ICC-5** | Tragic Fall | The Dramatic Descent | Western-typical |

## The Problem with Traditional Models

Our falsifiability analysis revealed that traditional narrative models are **epistemologically weak**:

| Model | Random Rejection | Shuffled Pass Rate | Scientific Rigor |
|-------|-----------------|-------------------|------------------|
| Aristotle | 75% | 98% | Very Low |
| Freytag | 82% | 78% | Low |
| Campbell | 68% | 92% | Very Low |
| Harmon | 65% | 95% | Very Low |
| Kishōtenketsu | 97.5% | 4% | High |

**Key insight**: Western models accept 78-98% of randomly shuffled text as "fitting" the pattern. They're so broad that they function more like Rorschach tests than scientific instruments.

## ICC Class Definitions

Based on information-theoretic features extracted from narrative trajectories:

### ICC-1: "Quiet Ascent" (Japanese-typical)
**The Contemplative Rise**
- **Archetype**: Bildungsroman without drama; kishōtenketsu enlightenment
- **Net change**: > +0.15 (rising arc)
- **Peaks**: ≤ 3
- **Volatility**: < 0.07
- **Description**: Gradual, steady rise with minimal turbulence. The narrative builds quietly toward revelation or growth.
- **Examples**: Soseki's *Kokoro*, Kawabata's *Snow Country*, quiet literary fiction
- **Random acceptance**: 0.0%

### ICC-2: "Gentle Descent" (Japanese-typical)
**The Elegiac Fall**
- **Archetype**: Mono no aware; beautiful sadness without melodrama
- **Net change**: < -0.15 (falling arc)
- **Peaks**: ≤ 4
- **Volatility**: < 0.07
- **Description**: Gradual, controlled decline toward loss or ending. The tragedy unfolds with restraint rather than dramatic reversals.
- **Examples**: *Tale of Genji*'s later chapters, Tanizaki's *The Makioka Sisters*
- **Random acceptance**: 2.5%

### ICC-3: "Eternal Return" (Western-typical)
**The Cyclical Journey**
- **Archetype**: Episodic adventure; the hero returns transformed
- **Net change**: ±0.2 (stable overall)
- **Peaks**: ≥ 4
- **Volatility**: ≥ 0.05
- **Symmetry**: ≥ 0.3 (beginning mirrors end)
- **Description**: Oscillating pattern that returns to origin. Many dramatic peaks but overall equilibrium. The journey is the destination.
- **Examples**: *The Odyssey*, *Don Quixote*, adventure serials, Harmon's Story Circle
- **Random acceptance**: 8.5%

### ICC-4: "Triumphant Climb" (Western-typical)
**The Dramatic Ascent**
- **Archetype**: Rags-to-riches with setbacks; the hero overcomes
- **Net change**: > +0.15
- **Peaks**: ≥ 4
- **Volatility**: ≥ 0.05
- **Trend R²**: ≥ 0.15 (follows discernible upward trend)
- **Description**: Rising trajectory with dramatic complications. Many setbacks but ultimate triumph. The Hollywood success story.
- **Examples**: *Rocky*, *Pride and Prejudice*, *The Count of Monte Cristo*, most Hollywood films
- **Random acceptance**: 2.5%

### ICC-5: "Tragic Fall" (Western-typical)
**The Dramatic Descent**
- **Archetype**: Shakespearean tragedy; the mighty are brought low
- **Net change**: < -0.15
- **Peaks**: ≥ 4
- **Volatility**: ≥ 0.05
- **Trend R²**: ≥ 0.15 (follows discernible downward trend)
- **Description**: Falling trajectory with dramatic reversals. Many moments of hope before final doom. Operatic, not quiet.
- **Examples**: *Macbeth*, *Breaking Bad*, *The Great Gatsby*, *Crime and Punishment*
- **Random acceptance**: 3.0%

### ICC-0: "Wandering Mist" (Universal)
**The Uncharted Path**
- **Archetype**: Polyphonic narrative; multiple interweaving arcs; experimental structure
- **Net change**: Any (no specific pattern)
- **Description**: Does not fit standard patterns. This is NOT a failure—it indicates genuinely unique structures that transcend simple classification. May represent: (1) Complex multi-plot structures where parallel arcs interfere (like *Anna Karenina*'s dual storylines), (2) Experimental or avant-garde narratives, (3) Genuinely innovative structures, or (4) Random/incoherent text. **The absence of pattern IS the pattern.**
- **Examples**: *Anna Karenina* (dual plots), *Ulysses*, *Cloud Atlas*, *2666*, Multi-POV epics, Experimental fiction
- **Random acceptance**: 83.5% (functions as reject category for noise, but also captures genuinely unique narratives)

## Falsifiability Results (v2 - Strengthened)

The ICC model now includes **structural requirements** for high-complexity classes to distinguish meaningful complexity from random noise:

```
============================================================
ICC MODEL FALSIFIABILITY TEST (v2)
============================================================

Random trajectory classification:
  ICC-0: 83.5%  <- Random noise correctly rejected
  ICC-1: 0.0%   <- Japanese-typical, VERY HIGH falsifiability
  ICC-2: 2.5%   <- Japanese-typical, VERY HIGH falsifiability
  ICC-3: 8.5%   <- Western-typical, HIGH falsifiability
  ICC-4: 2.5%   <- Western-typical, VERY HIGH falsifiability
  ICC-5: 3.0%   <- Western-typical, VERY HIGH falsifiability

✓ PASS: Low-complexity classes reject >95% of random
✓ PASS: High-complexity classes reject >86% of random
```

### Interpretation

| Class | Random Acceptance | Random Rejection | Falsifiability |
|-------|------------------|------------------|----------------|
| ICC-0 (Wandering Mist) | 83.5% | - | (catch-all category) |
| ICC-1 (Japanese) | **0.0%** | **100%** | **Very High** |
| ICC-2 (Japanese) | **2.5%** | **97.5%** | **Very High** |
| ICC-3 (Western) | **8.5%** | **91.5%** | **High** |
| ICC-4 (Western) | **2.5%** | **97.5%** | **Very High** |
| ICC-5 (Western) | **3.0%** | **97.0%** | **Very High** |

**Key achievement**: ALL ICC classes now have >90% random rejection, making them scientifically rigorous.

## Cultural Predictions

The ICC model makes testable predictions:

| Prediction | Testable Hypothesis |
|------------|---------------------|
| ICC-1/ICC-2 → Japanese | Japanese literature should show higher rates of ICC-1 and ICC-2 classification |
| ICC-3/ICC-4/ICC-5 → Western | Western literature should show higher rates of ICC-3, ICC-4, ICC-5 classification |
| Complexity difference | Japanese narratives should have lower average peak counts and volatility |

## Features Used

The ICC model classifies based on three primary features:

1. **Net Change**: `trajectory[-1] - trajectory[0]`
   - Measures overall direction (rising vs falling)

2. **Peak Count**: Number of significant peaks in trajectory
   - Detected using scipy's `find_peaks` with prominence threshold
   - Measures structural complexity

3. **Volatility**: Standard deviation of first differences
   - `np.std(np.diff(trajectory))`
   - Measures smoothness vs turbulence

## Comparison to Traditional Models

| Aspect | Traditional Models | ICC Model |
|--------|-------------------|-----------|
| Origin | Theoretical/philosophical | Data-driven discovery |
| Thresholds | Vague ("rising action") | Specific numbers (≥4 peaks) |
| Falsifiability | Low (fits anything) | High (rejects noise) |
| Cultural bias | Western-centric | Acknowledges cultural difference |
| Testability | Subjective interpretation | Objective measurement |

## Implementation

```python
from src.detectors.icc import ICCDetector, ICCResult

detector = ICCDetector()
result = detector.detect(trajectory, trajectory_id="book_1", title="My Novel")

print(f"Class: {result.icc_class}")  # e.g., "ICC-1"
print(f"Name: {result.class_name}")   # e.g., "Low Complexity Rise"
print(f"Confidence: {result.confidence}")
print(f"Cultural prediction: {result.cultural_prediction}")  # "japanese" or "western"
```

## Theoretical Implications

1. **"Universal narrative structure" may be a myth**
   - What we call "universal" may be Western structures imposed globally through Hollywood and publishing dominance

2. **Cultural encoding is measurable**
   - Japanese and Western narratives have statistically different information-theoretic profiles

3. **Complexity as cultural marker**
   - Western narratives favor high complexity (many peaks, high volatility)
   - Japanese narratives favor low complexity (gradual arcs, smooth transitions)

4. **Scientific narrative analysis is possible**
   - But requires abandoning loose theoretical models for measurable, falsifiable criteria

## Multilingual Support

### Does ICC work with any language?

**Short answer**: Yes, but with caveats depending on the sentiment analysis method.

### Language Support by Method

| Method | Languages | Quality | Notes |
|--------|-----------|---------|-------|
| **VADER** | English only | High for English | Rule-based, English-specific lexicon |
| **BERT Multilingual** | 100+ languages | Good | `nlptown/bert-base-multilingual-uncased-sentiment` |
| **Japanese-specific** | Japanese only | High for Japanese | Uses `daigo/bert-base-japanese-sentiment` |

### How It Works

The ICC model operates on **sentiment trajectories**, not raw text. The language dependency is in how those trajectories are extracted:

```
Text (any language) → Sentiment Functor → Trajectory → ICC Classification
                      ↑
                      Language-dependent step
```

Once you have a trajectory (array of sentiment values over narrative time), the ICC classification is **language-agnostic**—it only measures:
- Net change (first → last value)
- Peak count
- Volatility
- Trend R²

### Using ICC with Different Languages

#### English (default)
```bash
python scripts/analyze_narrative.py your_text.txt
python scripts/visualize_narrative.py your_text.txt
```

#### Japanese
```bash
# Uses dedicated Japanese sentiment analyzer
python -m src.functors.sentiment_ja -i input/ -o output/
```

#### Other Languages (Multilingual BERT)
```bash
# Use BERT with multilingual model
python scripts/analyze_narrative.py your_text.txt --method bert
```

For best results with non-English text:
```python
from src.functors.sentiment import SentimentFunctor

# Use multilingual BERT
functor = SentimentFunctor(
    method="bert",
    bert_model="nlptown/bert-base-multilingual-uncased-sentiment"
)
```

### Supported Languages (via Multilingual BERT)

The multilingual BERT model supports 104 languages including:

| Well-Supported | Moderate Support | Limited Support |
|----------------|------------------|-----------------|
| English, German, French, Spanish, Italian, Portuguese, Dutch, Russian, Chinese, Japanese, Korean, Arabic | Polish, Turkish, Vietnamese, Indonesian, Thai, Hebrew, Greek, Czech | Low-resource languages, code-switching text |

### Accuracy Considerations

| Scenario | Expected Quality | Recommendation |
|----------|------------------|----------------|
| English fiction | ★★★★★ | Use VADER (fast) or ensemble |
| Japanese fiction | ★★★★☆ | Use Japanese-specific analyzer |
| Major European languages | ★★★★☆ | Use multilingual BERT |
| Chinese/Korean | ★★★☆☆ | Use multilingual BERT, verify manually |
| Low-resource languages | ★★☆☆☆ | Validate on known texts first |

### Known Limitations

1. **Cultural sentiment expression differs**: What's "positive" in one culture may be neutral in another
2. **Irony/sarcasm**: Most models fail on subtle irony across all languages
3. **Historical texts**: Archaic language may confuse modern models
4. **Poetry**: Highly figurative language reduces accuracy

### Extending to New Languages

To add first-class support for a new language:

1. Create a language-specific sentiment functor (see `src/functors/sentiment_ja.py`)
2. Use a language-specific transformer model from HuggingFace
3. Validate on known texts with expected emotional arcs

The ICC classification layer needs no changes—it works on any properly extracted trajectory.

## Future Directions

1. **Expand Japanese corpus**: Current imbalance (55 Japanese vs 2,600+ Western) limits cross-cultural validation

2. **Historical analysis**: Track ICC distributions across literary periods

3. **Genre-specific patterns**: Do certain genres cluster in specific ICC classes?

4. **Author fingerprinting**: Can ICC profiles distinguish individual authors?

5. **Cross-linguistic validation**: Test ICC on other non-Western traditions (Chinese, Arabic, African)

6. **Language-specific models**: Develop dedicated sentiment analyzers for Chinese, Korean, Arabic, and other literary traditions

## References

- Popper, K. (1959). The Logic of Scientific Discovery
- Reagan et al. (2016). "The emotional arcs of stories" - PNAS
- Original pattern discovery analysis: `docs/discovered_patterns.md`
- Falsifiability analysis: `docs/falsifiability.md`
