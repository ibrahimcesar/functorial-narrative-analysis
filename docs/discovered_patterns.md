# Data-Driven Narrative Pattern Discovery

## Methodology

Instead of testing pre-existing models (which fail falsifiability tests), we used unsupervised clustering to discover natural patterns in 281 narrative trajectories:
- 28 Japanese (Aozora Bunko)
- 253 Western (HuggingFace fiction + Gutenberg)

### Features Extracted
- Net change (start to end)
- Number of peaks/valleys
- Peak positions (early/middle/late)
- Volatility (variability)
- Complexity (direction changes)
- Skewness

### Clustering
- Method: Hierarchical clustering
- Optimal clusters: 13 (determined by silhouette analysis)
- Silhouette score: 0.250

## Discovered Patterns

### Culture-Specific Patterns

#### Japanese-Specific (P4, P8)

**P4: "Gradual Rise"**
- 12 Japanese, 1 Western (92% Japanese)
- Rising trajectory with fewer peaks (avg 1.7)
- Smooth, contemplative arc
- Resembles kishōtenketsu's subtle development

**P8: "Tragic Descent"**
- 7 Japanese, 1 Western (88% Japanese)
- Falling trajectory with 3.5 peaks
- Oscillating decline
- Distinct from Western tragedy patterns

#### Western-Dominant Patterns

**P1: "Rising Multi-Peak"** (40 Western, 0 Japanese)
- Continuous rise with 5+ peaks
- High complexity
- Classic "progress with setbacks" narrative

**P2: "Falling Multi-Peak"** (45 Western, 2 Japanese)
- Sustained decline with many oscillations
- Complex tragedy structure

**P3, P5, P6: "Oscillating Complex"** (103 Western, 5 Japanese)
- Maintains level but with high volatility
- Many peaks and direction changes
- Episodic or cyclical narratives

### Universal Patterns

No patterns showed truly balanced distribution (>30% from both cultures), suggesting:
1. Cultural encoding in narrative structure is stronger than expected
2. Or sample imbalance (28 JP vs 253 WE) limits detection

## Key Findings

### 1. Japanese Narratives are Simpler in Structure

| Metric | Japanese (avg) | Western (avg) |
|--------|---------------|---------------|
| Peak count | 2.1 | 5.0 |
| Complexity | 8.3 | 14.7 |
| Volatility | 0.05 | 0.08 |

Japanese narratives show:
- Fewer dramatic reversals
- Smoother information flow
- Less "roller coaster" dynamics

### 2. Western Narratives Favor High Complexity

Western patterns (P1-P3, P5-P7, P9-P13) consistently show:
- 5+ peaks on average
- High complexity scores
- Greater volatility

This aligns with:
- Hollywood structure (constant tension/release)
- Page-turner dynamics
- Conflict-driven plotting

### 3. The "Gradual Rise" (P4) May Be a Kishōtenketsu Signature

P4 characteristics:
- Starts low, ends high
- Few dramatic peaks
- Smooth transitions
- 92% Japanese

This could represent the kishōtenketsu structure:
- Ki (起): Low start
- Shō (承): Gradual development
- Ten (転): Subtle twist (single peak)
- Ketsu (結): Resolution at higher level

## Proposed New Model: Information Complexity Classes (ICC)

Based on findings, we propose categorizing narratives by information-theoretic properties:

### ICC-1: Low Complexity Rise (Japanese-typical)
- Net change: > +0.2
- Peaks: < 3
- Volatility: < 0.06
- Example: Kishōtenketsu, contemplative fiction

### ICC-2: Low Complexity Fall (Japanese-typical)
- Net change: < -0.2
- Peaks: < 4
- Volatility: < 0.06
- Example: Mono no aware tragedies

### ICC-3: High Complexity Oscillation (Western-typical)
- Net change: ±0.2
- Peaks: > 4
- Volatility: > 0.07
- Example: Episodic adventures, soap operas

### ICC-4: High Complexity Rise (Western-typical)
- Net change: > +0.2
- Peaks: > 4
- Volatility: > 0.07
- Example: Rags-to-riches with complications

### ICC-5: High Complexity Fall (Western-typical)
- Net change: < -0.2
- Peaks: > 4
- Volatility: > 0.07
- Example: Shakespearean tragedy

## Falsifiability of ICC Model

The ICC model is falsifiable because:

1. **Specific thresholds**: Numerical cutoffs for peaks, volatility
2. **Rejects random**: Low-complexity patterns (ICC-1, ICC-2) reject random noise
3. **Cultural predictions**: ICC-1/2 should appear more in Japanese; ICC-3/4/5 in Western
4. **Testable on new data**: Clear classification criteria

| Test | ICC Model | Traditional Models |
|------|-----------|-------------------|
| Random rejection | High (for ICC-1/2) | Low |
| Shuffled rejection | High | Very Low |
| Cultural discrimination | Measurable | Claimed but untested |

## Conclusion

Traditional narrative models (Aristotle, Freytag, Campbell, Harmon) are too loose to be scientific instruments. Our data-driven discovery reveals:

1. **Japanese and Western narratives have measurably different structures**
2. **Complexity (peaks, volatility) is a key differentiator**
3. **The ICC model provides testable, falsifiable categories**
4. **Cultural patterns exist but require culture-specific models**

The "universal story structure" may be a myth. What we call "universal" may simply be Western structures imposed globally through Hollywood and Western publishing dominance.
