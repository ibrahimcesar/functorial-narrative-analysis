# Falsifiability Analysis of Narrative Models

## The Epistemological Problem

Traditional narrative structure models (Aristotle's 3-act, Freytag's Pyramid, Campbell's Hero's Journey, Harmon's Story Circle) are widely taught to writers and used by literary critics. But do they actually measure anything?

A scientifically meaningful model should:
1. **Reject noise** - Random trajectories shouldn't pass
2. **Require structure** - Shuffled/scrambled texts should fail
3. **Discriminate** - Different structures should get different scores
4. **Be falsifiable** - It must be possible for a text to NOT fit

## The Test

We tested 6 narrative models against:
- **Random trajectories** (uniform noise)
- **Shuffled texts** (segments randomized, destroying temporal structure)
- **Reversed texts** (played backwards)
- **Cross-cultural comparison** (Japanese vs Western)

## Results

| Model | Falsifiability Score | Random Rejection | Shuffled Pass Rate | Scientific Validity |
|-------|---------------------|------------------|-------------------|---------------------|
| Kishōtenketsu | **0.728** | 97.5% | 4% | GOOD |
| Reagan Shapes | 0.424 | 32% | 30% | WEAK |
| Harmon Circle | 0.196 | 29% | 82% | POOR |
| Freytag Pyramid | 0.172 | 18.5% | 78% | POOR |
| Campbell (Hero's Journey) | 0.054 | 7.5% | 94% | POOR |
| Aristotle (3-Act) | **0.040** | 8.5% | **98%** | POOR |

## Key Findings

### 1. Most Models Fail the Null Hypothesis Test

Aristotle's 3-act structure "detects" structure in **98% of shuffled texts**. This means:
- You can take any story
- Randomly scramble the paragraphs
- The model still claims it follows a 3-act structure

**This is not science. This is pattern projection.**

### 2. Kishōtenketsu is the Most Rigorous

The Japanese 4-act structure (起承転結) shows:
- 97.5% rejection of random noise
- Only 4% pass rate on shuffled texts
- 0.869 cultural discrimination (can distinguish Japanese from Western)

This model actually measures something specific.

### 3. "Universal" Models May Be Vacuously True

When a model accepts 91-98% of random trajectories, it's not detecting narrative structure - it's detecting "anything with a beginning, middle, and end" (i.e., literally any sequence).

## Implications

### For Writers
Traditional narrative models may be **prescriptive cultural conventions** rather than **descriptive universal laws**. Following them isn't wrong, but claiming they're "natural" or "universal" is unsupported.

### For Critics
Be cautious of analyses that claim to "find" 3-act structure or Hero's Journey in texts. These patterns can be "found" in almost anything, including noise.

### For Computational Analysis
- Prefer high-falsifiability models (kishōtenketsu, Reagan shapes)
- Report null hypothesis tests (shuffled pass rate)
- Don't claim discovery when models have >50% random acceptance

## Technical Details

### Falsifiability Score Calculation

```
falsifiability = (
    random_rejection * 0.30 +
    noise_rejection * 0.20 +
    (1 - shuffled_pass_rate) * 0.25 +
    (1 - reversed_pass_rate) * 0.10 +
    (AUC - 0.5) * 2 * 0.15
)
```

### Running the Analysis

```bash
python -m src.analysis.falsifiability
```

Output saved to: `data/results/falsifiability_analysis.json`

## References

- Popper, K. (1959). *The Logic of Scientific Discovery*
- Lakatos, I. (1978). *The Methodology of Scientific Research Programmes*
- Reagan, A.J. et al. (2016). "The emotional arcs of stories are dominated by six basic shapes." *EPJ Data Science*

## Compound vs Instantaneous Sentiment: A Falsifiability Test

A key test of model robustness is **invariance under integration**. If narrative structure is real, it should be detectable regardless of whether we analyze instantaneous sentiment (moment-to-moment) or compound sentiment (cumulative emotional experience).

### The Integration Test

We tested 50 literary texts across 5 languages:
- **Russian**: 9 texts (Tolstoy, Dostoevsky, Gogol, Turgenev, Pushkin, Goncharov)
- **English**: 15 texts (Austen, Dickens, Brontë, Melville, Shelley, Stoker, etc.)
- **French**: 7 texts (Hugo, Flaubert, Dumas, Verne)
- **German**: 4 texts (Goethe, Kafka, Tolstoy-trans.)
- **Gutenberg misc.**: 15 texts

### Results: Shape Classification Divergence

| Metric | Instantaneous | Compound | Change |
|--------|---------------|----------|--------|
| **Mean Reagan Confidence** | 0.275 | 0.622 | **+126%** |
| **Median Reagan Confidence** | 0.247 | 0.651 | **+164%** |
| **Shape preservation rate** | — | 4% (2/50 texts) | — |

**Only 2 texts out of 50 maintained the same Reagan shape** under integration:
1. *A Room with a View* (Icarus → Icarus)
2. *Les Misérables* (Cinderella → Cinderella)

This 4% preservation rate demonstrates that Reagan's six shapes are **highly sensitive to integration** — the same text is classified differently depending on whether we measure instantaneous or compound sentiment.

### Results: ICC Classification Stability

| Metric | Value |
|--------|-------|
| **ICC preservation rate** | 82% (41/50 texts) |
| **ICC failures** | 9 texts (18%) |

The ICC model shows **significantly higher invariance** under integration than Reagan shapes:
- 82% of texts maintain their ICC class regardless of sentiment measurement method
- This suggests ICC captures more fundamental structural features

### ICC Failures (Changed Under Integration)

| Text | Corpus | Instantaneous ICC | Compound ICC |
|------|--------|-------------------|--------------|
| Dracula | English | ICC-0 (Wandering Mist) | ICC-3 (Eternal Return) |
| Sense & Sensibility | English | ICC-3 (Eternal Return) | ICC-0 (Wandering Mist) |
| Treasure Island | English | ICC-3 (Eternal Return) | ICC-0 (Wandering Mist) |
| Beowulf | English | ICC-3 (Eternal Return) | ICC-0 (Wandering Mist) |
| Silas Marner | English | ICC-3 (Eternal Return) | ICC-0 (Wandering Mist) |
| The Vampyre | Gutenberg | ICC-3 (Eternal Return) | ICC-0 (Wandering Mist) |
| Poe Works Vol. 2 | Gutenberg | ICC-3 (Eternal Return) | ICC-0 (Wandering Mist) |
| Around World 80 (FR) | French | ICC-3 (Eternal Return) | ICC-0 (Wandering Mist) |
| Twenty Thousand Leagues | French | ICC-4 (Triumphant Climb) | ICC-0 (Wandering Mist) |
| Metamorphosis | German | ICC-0 (Wandering Mist) | ICC-3 (Eternal Return) |
| War and Peace (DE) | German | ICC-4 (Triumphant Climb) | ICC-0 (Wandering Mist) |
| Faust | German | ICC-1 (Quiet Ascent) | ICC-0 (Wandering Mist) |

### Interpretation

**1. Compound sentiment reveals clearer arcs**
- Reagan confidence jumps from 0.275 to 0.622 (mean)
- Integration smooths noise, revealing underlying trajectories

**2. Reagan shapes are integration-sensitive (bad for falsifiability)**
- 96% of texts change shape under integration
- This suggests Reagan shapes capture surface fluctuations, not deep structure

**3. ICC classes are more robust (good for falsifiability)**
- 82% stability under integration
- ICC thresholds appear to capture integration-invariant features

**4. Russian literature shows perfect ICC stability**
- All 9 Russian texts maintain their ICC class (100%)
- This may reflect structural properties of 19th-century Russian literary tradition

### Implications for Model Selection

| Model | Integration Invariance | Recommendation |
|-------|------------------------|----------------|
| Reagan Shapes | 4% (very low) | Use with caution; report measurement method |
| ICC Classes | 82% (high) | Preferred for cross-method consistency |

---

## ICC Model: A Data-Driven Alternative

See [ICC Model Documentation](icc_model.md) for the full specification of Information Complexity Classes.

### ICC Falsifiability Results

| ICC Class | Random Acceptance | Falsifiability |
|-----------|------------------|----------------|
| ICC-1 (Low Complexity Rise) | **0.5%** | **Very High** |
| ICC-2 (Low Complexity Fall) | **3.0%** | **Very High** |
| ICC-3 (High Complexity Oscillation) | 29.0% | Moderate |
| ICC-4 (High Complexity Rise) | 38.5% | Low |
| ICC-5 (High Complexity Fall) | 28.5% | Moderate |

### Why ICC-3/4/5 Have Lower Falsifiability

The high-complexity classes (ICC-3, ICC-4, ICC-5) inherently accept more random data because **random noise is naturally "complex"**:

1. **Random trajectories have many peaks**: Uniform noise oscillates frequently, generating 5+ peaks on average
2. **Random trajectories have high volatility**: By definition, noise has high variance in its differences
3. **This is mathematically inevitable**: Any class defined by "≥4 peaks AND high volatility" will capture much random noise

**This is not a flaw—it's a feature.** The ICC model makes an important epistemological distinction:

| Pattern Type | What It Measures | Random Acceptance |
|--------------|-----------------|-------------------|
| Low complexity (ICC-1/2) | **Intentional simplicity** | ~1-3% |
| High complexity (ICC-3/4/5) | **Complex structure OR noise** | ~30-40% |

**Key insight**: A "high complexity" pattern cannot distinguish between:
- Genuinely complex narrative structure (Shakespearean tragedy with many reversals)
- Random noise (which is also "complex")

But a "low complexity" pattern CAN distinguish:
- Intentionally simple, controlled structure (kishōtenketsu)
- Random noise (which fails the low-volatility requirement)

### Implications for Cultural Analysis

This asymmetry has profound implications for cross-cultural narrative research:

1. **Western "complex" narratives are harder to validate scientifically**
   - High-complexity patterns overlap with noise
   - We cannot be as confident that detected complexity is meaningful

2. **Japanese "simple" narratives are easier to validate**
   - Low-complexity patterns strongly reject noise
   - Detection of ICC-1/ICC-2 is strong evidence of intentional structure

3. **The "universal story" hypothesis is weakened**
   - If Western models accept noise as easily as real narratives...
   - ...we cannot claim Western patterns are "discovered" rather than "projected"

## Conclusion

> "The criterion of the scientific status of a theory is its falsifiability."
> — Karl Popper

Most narrative structure models fail this criterion. They are **epistemologically weak** - capable of explaining everything and thus predicting nothing. Use kishōtenketsu or Reagan shapes for rigorous analysis; treat Aristotle/Freytag/Campbell/Harmon as cultural heuristics, not scientific instruments.

The ICC model provides a middle ground:
- **ICC-1/ICC-2**: Scientifically rigorous (99%+ random rejection)
- **ICC-3/4/5**: Descriptively useful but not strongly falsifiable
