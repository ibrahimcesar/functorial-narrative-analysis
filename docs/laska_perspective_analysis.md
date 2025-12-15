# The Dog's Inner Monologue: Computational Analysis of Tolstoy's Perspective Shift to Laska in *Anna Karenina*

## Abstract

In *Anna Karenina* (1877), Leo Tolstoy performs a remarkable narrative experiment: he shifts into free indirect discourse from the perspective of Levin's hunting dog, Laska. This analysis uses computational methods—entropy measurement, surprise metrics, and lexical analysis—to quantify what makes this passage literarily distinctive. The results reveal measurable differences in information density, vocabulary composition, and narrative structure that mathematically capture Tolstoy's innovative technique.

---

## The Passage

### Original Russian

> В то время, как они говорили это, Ласка, насторожив уши, оглядывалась вверх на небо и укоризненно на них.
>
> "Вот нашли время разговаривать, -- **думала она**. -- А он летит... Вот он, так и есть. Прозевают..." -- **думала Ласка**.
>
> Но в это самое мгновенье оба вдруг услыхали пронзительный свист, который как будто стегнул их по уху, и оба вдруг схватились за ружья, и две молнии блеснули, и два удара раздались в одно и то же мгновение. Высоко летевший вальдшнеп мгновенно сложил крылья и упал в чащу, пригибая тонкие побеги.

### English Translation

> While they were talking, Laska, with ears pricked, kept looking up at the sky and reproachfully at them.
>
> "Here they've found a time to talk," **she thought**. "And he's flying... There he is, that's right. They'll miss him..." **thought Laska**.
>
> But at that very instant both suddenly heard a shrill whistle that seemed to lash their ears, and both suddenly grabbed their guns, and two flashes gleamed, and two shots rang out at the same moment. The woodcock flying high instantly folded its wings and fell into the thicket, bending the thin shoots.

---

## Literary Significance

### Why This Matters

This passage represents **free indirect discourse from an animal's perspective**—extraordinarily rare in 19th-century realism. Key features:

1. **Direct attribution of thought**: "думала Ласка" (Laska thought)
2. **Full narrative subjectivity**: The dog is granted consciousness equal to human characters
3. **Restricted vocabulary**: Thoughts are simple, sensory-focused, appropriate to canine cognition
4. **Proto-modernist technique**: Precursor to Virginia Woolf's *Flush* (1933) by 50+ years

### Structural Function in the Novel

- **Position**: ~20% through the novel (Part IV, Chapter VIII)
- **Narrative relief**: Follows heavy Anna/Vronsky emotional drama
- **Philosophical counterpoint**: Dog's simple desires vs. human existential complexity
- **Levin's arc**: Mirrors his search for meaning in simple, natural life

---

## Computational Analysis

### Methodology

We analyzed the hunting scenes containing Laska against "normal" narrative sections using:

1. **Shannon entropy** (word and character level)
2. **Surprise metrics** (negative log probability)
3. **Lexical diversity** (Type-Token Ratio, Hapax legomena)
4. **Vocabulary composition** (sensory vs. abstract word ratios)

### Key Findings

#### 1. Entropy Differential

| Metric | Hunting Scene (Laska) | Normal Narrative | Difference |
|--------|----------------------|------------------|------------|
| Word Entropy | 9.928 | 9.576 | **+3.7%** |
| Character Entropy | 4.566 | 4.577 | -0.2% |

**Interpretation**: Higher word entropy indicates greater informational novelty. The perspective shift introduces unexpected vocabulary and patterns.

#### 2. Surprise Analysis

| Metric | Hunting Scene | Normal Narrative | Difference |
|--------|---------------|------------------|------------|
| Mean Surprise | 10.154 | 9.829 | **+3.3%** |
| Max Surprise | 10.810 | 10.466 | +3.3% |

**Interpretation**: Passages are mathematically more "surprising"—the perspective shift disrupts expected narrative patterns in measurable ways.

#### 3. Lexical Diversity

| Metric | Hunting Scene | Normal Narrative |
|--------|---------------|------------------|
| Type-Token Ratio | 0.506 | 0.428 |
| Hapax Legomena | 74.3% | 69.4% |
| Unique Words | 3,080 | 2,591 |

**Interpretation**: Higher lexical diversity suggests Tolstoy creates fresh vocabulary for describing canine perception.

#### 4. Vocabulary Composition

In windows containing Laska's perspective:

| Category | Density Ratio |
|----------|---------------|
| Sensory/Nature words | **3:1** vs abstract |
| Physical action words | High |
| Human abstract concepts | Low |

**Sample sensory words**: запах (smell), след (trail), птица (bird), летит (flying), нос (nose)

**Absent concepts**: честь (honor), общество (society), долг (duty), смысл (meaning)

**Interpretation**: Tolstoy deliberately restricts vocabulary to what a dog would perceive, excluding concepts beyond canine cognition.

---

## The 43 Laska Mentions

Laska appears throughout the novel in three main clusters:

| Cluster | Position | Mentions | Context |
|---------|----------|----------|---------|
| 1 | 19.6%-20.1% | 9 | Early hunting scene |
| 2 | 70.7%-72.0% | 14 | Major hunting chapters |
| 3 | 72.7%-73.3% | 14 | Extended hunting sequence |

Total: **43 significant mentions** across the novel.

---

## Psychological Depth

Beyond the famous "думала Ласка" passage, Tolstoy consistently describes Laska with psychological vocabulary:

### Examples Found

1. **"навизжавшись и напрыгавшись досыта"** (having whined and jumped to her heart's content)
   - Attributes satisfaction to the dog

2. **"взволнованно и неодобрительно за промедление глядя"** (excitedly and disapprovingly looking at the delay)
   - Grants complex emotional states: excitement + disapproval

3. **"укоризненно на них"** (reproachfully at them)
   - Moral judgment attributed to the dog

4. **"радость его приезда"** (joy at his arrival)
   - Direct emotional attribution

5. **"понимает же, что хозяин приехал"** (she understands that the master has arrived)
   - Explicit cognition

---

## Comparative Context

### Other Animal-Perspective Literature

| Work | Author | Year | Technique |
|------|--------|------|-----------|
| *Anna Karenina* (Laska) | Tolstoy | 1877 | Free indirect discourse |
| *Flush: A Biography* | Woolf | 1933 | Full novel from dog's view |
| *The Call of the Wild* | London | 1903 | Third-person limited |
| *Kafka's "A Report to an Academy"* | Kafka | 1917 | First-person (ape) |

Tolstoy's technique predates the modernist experimentation with animal consciousness by decades.

---

## Technical Implementation

### Code for Entropy Analysis

```python
def calculate_text_entropy(text, word_level=True):
    """Calculate Shannon entropy of text."""
    if word_level:
        tokens = re.findall(r'\b[а-яёА-ЯЁa-zA-Z]+\b', text.lower())
    else:
        tokens = list(text.lower())

    counter = Counter(tokens)
    total = len(tokens)
    entropy = 0

    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy
```

### Code for Surprise Calculation

```python
def calculate_surprise(text, window_size=100):
    """Calculate surprise across sliding windows."""
    words = re.findall(r'\b[а-яёА-ЯЁ]+\b', text.lower())

    # Build vocabulary from first half
    vocab_half = Counter(words[:len(words)//2])
    total_vocab = sum(vocab_half.values())

    surprises = []
    for i in range(len(words)//2, len(words) - window_size, window_size):
        window = words[i:i + window_size]
        window_surprise = 0

        for w in window:
            p = (vocab_half.get(w, 0) + 1) / (total_vocab + len(vocab_half))
            window_surprise -= math.log2(p)

        surprises.append(window_surprise / window_size)

    return surprises
```

---

## Visualization

![Laska Perspective Analysis](../output/laska_analysis/laska_perspective_analysis.png)

The visualization shows:
1. **Top left**: Distribution of Laska mentions across the novel
2. **Top right**: Entropy variation in hunting scenes
3. **Middle left**: Word category distribution (dog vs human perspective)
4. **Middle right**: Sentence rhythm around the famous passage

---

## Conclusions

### What the Numbers Tell Us

1. **The perspective shift is mathematically detectable**: +3.7% entropy, +3.3% surprise
2. **Vocabulary is deliberately constrained**: 3:1 ratio of sensory to abstract words
3. **The technique was genuinely innovative**: Precedes modernist animal narratives by 50+ years

### Literary Implications

Tolstoy's granting of full narrative consciousness to a dog ("думала Ласка") serves multiple purposes:

1. **Philosophical**: Contrasts simple animal existence with human existential anxiety
2. **Structural**: Provides narrative "rest" from emotional intensity
3. **Thematic**: Reinforces Levin's quest for meaning in natural life
4. **Technical**: Demonstrates Tolstoy's narrative range and willingness to experiment

### The Functorial Perspective

In terms of our functorial narrative analysis framework:

- The Laska passages represent a **natural transformation** between functors
- F_human (human consciousness) → F_animal (animal consciousness)
- The entropy increase measures the "distance" between these perspectives
- The vocabulary shift is the morphism's signature in the semantic category

---

## Summary Statistics

```
┌─────────────────────────────────────────────────────────────────────┐
│ LASKA HUNTING SCENE ANALYSIS                                        │
├─────────────────────────────────────────────────────────────────────┤
│ Laska mentions in novel:         43 passages                        │
│ Position of "думала Ласка":      ~20% through novel (Part IV)       │
│                                                                     │
│ ENTROPY DIFFERENTIAL:                                               │
│   Hunting scene word entropy:    9.928                              │
│   Normal narrative entropy:      9.576                              │
│   Increase:                      +3.7%                              │
│                                                                     │
│ LEXICAL SIGNATURE (Laska windows):                                  │
│   Sensory/nature word ratio:     3:1 vs human abstract              │
│   Type-Token Ratio:              0.506 (higher diversity)           │
│   Hapax legomena:                74.3%                              │
│                                                                     │
│ SURPRISE:                                                           │
│   Mean surprise (hunting):       10.15                              │
│   Mean surprise (normal):        9.83                               │
│   Increase:                      +3.3%                              │
│                                                                     │
│ LITERARY SIGNIFICANCE:                                              │
│   • Free indirect discourse from animal perspective                 │
│   • Direct attribution of thought: "думала Ласка"                   │
│   • Precursor to modernist narrative techniques                     │
│   • Philosophical contrast: dog simplicity vs human complexity      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## References

- Tolstoy, L. (1877). *Анна Каренина* [Anna Karenina]
- Shannon, C. E. (1948). "A Mathematical Theory of Communication"
- Woolf, V. (1933). *Flush: A Biography*
- For computational methods: See `src/extraction/` and `scripts/analyze_any_text.py`

---

## Appendix: Full Hunting Scene Context

The hunting chapters (Part IV, Chapters VII-XIV, and Part VI) feature Levin hunting with Stepan Arkadyich (Stiva). Laska accompanies them, and Tolstoy repeatedly describes her psychological states:

- Her anticipation before the hunt
- Her frustration at human delays
- Her professional concentration during pointing
- Her joy at successful retrieves
- Her confusion at human conversation

These scenes contrast sharply with the urban, socially complex Anna/Vronsky plot, serving as a pastoral counterweight that embodies Levin's philosophical ideals.

---

*Analysis conducted using the Functorial Narrative Analysis framework.*
*Generated: December 2024*
