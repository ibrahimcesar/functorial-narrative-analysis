# Functorial Narrative Analysis: A Cross-Cultural Study of Story Shapes

---

## Abstract

Reagan et al. (2016) identified six emotional arc shapes dominating English-language fiction in the Project Gutenberg corpus. We argue this finding is potentially an artifact of temporal, cultural, and editorial selection biases. We propose a category-theoretic framework for narrative analysis that (1) formalizes story structure as morphisms in a category **Narr**, (2) defines multiple observation functors capturing different aspects of narrative transformation, and (3) enables principled cross-cultural comparison by treating different narrative traditions as different subcategories or as optimizing different functorial objectives. We outline a concrete research design using modern corpora (fan fiction, web novels, film subtitles) across Western, Japanese, Korean, and Chinese sources.

---

## 1. The Problem: Are Six Shapes Universal or Provincial?

### 1.1 The Reagan et al. Finding

The 2016 study analyzed ~1,700 stories from Project Gutenberg using sentiment analysis, identifying six core emotional trajectories:

1. **Rags to Riches** (steady rise)
2. **Tragedy** (steady fall)  
3. **Man in a Hole** (fall → rise)
4. **Icarus** (rise → fall)
5. **Cinderella** (rise → fall → rise)
6. **Oedipus** (fall → rise → fall)

These account for ~80% of the corpus, with more complex stories decomposable into combinations.

### 1.2 The Bias Problem

The Project Gutenberg corpus has severe selection biases:

| Dimension | Bias |
|-----------|------|
| **Temporal** | Pre-1923 (US copyright threshold) |
| **Linguistic** | Predominantly English |
| **Cultural** | Western publishing norms |
| **Editorial** | Victorian/Edwardian gatekeeping |
| **Generic** | Novel-length prose fiction |
| **Class** | Published = elite access |

**Hypothesis**: The "six shapes" may be *Victorian English publishing shapes*, not *human narrative shapes*.

### 1.3 The Cross-Cultural Challenge

East Asian narrative traditions (Chinese, Japanese, Korean) employ fundamentally different structures. **Kishōtenketsu** (起承転結) is a four-act structure:

- **Ki** (起): Introduction
- **Shō** (承): Development  
- **Ten** (転): Twist/Turn
- **Ketsu** (結): Reconciliation

Crucially, kishōtenketsu does not center **conflict** as the driver of narrative tension. Instead, tension arises from **juxtaposition** and **reframing**—the "ten" (twist) changes how we understand the preceding acts.

This suggests that measuring emotional valence (the Reagan approach) may be measuring the *wrong observable* for non-Western narrative. A story optimized for kishōtenketsu might show a flat emotional arc but a highly structured *epistemic* arc.

### 1.4 The Harmon Story Circle: A Western Generator

Dan Harmon's Story Circle is an 8-step structure derived from Campbell's Hero's Journey, widely used in contemporary television writing (Community, Rick and Morty) and screenwriting pedagogy:

```
            1. COMFORT
         (A character is in a zone of comfort)
                 ↓
            2. NEED
         (But they want something)
                 ↓
            3. GO
         (They enter an unfamiliar situation)
                 ↓
            4. SEARCH
         (Adapt to it)
                 ↓
            5. FIND
         (Get what they wanted)
                 ↓
            6. TAKE
         (Pay a heavy price for it)
                 ↓
            7. RETURN
         (Return to their familiar situation)
                 ↓
            8. CHANGE
         (Having changed)
```

Geometrically, Harmon represents this as a circle bisected horizontally:

```
           ORDER (Known World)
        ___1_________8___
       /                  \
      2                    7
     |                      |
     |    ←—— DESCENT ——→   |
     |                      |
      3                    6
       \____4_____5______ /
           CHAOS (Unknown)
```

The top half is the "ordinary world" (order, comfort, known). The bottom half is the "special world" (chaos, trial, unknown). Steps 1-4 are descent; steps 5-8 are ascent.

**Key Insight**: The Harmon Circle maps directly onto Reagan's "man-in-a-hole" shape when projected through F_sentiment:

| Harmon Steps | Position | F_sentiment |
|--------------|----------|-------------|
| 1-2 (Comfort/Need) | Top | High (stable) |
| 3-4 (Go/Search) | Descending | Falling |
| 5 (Find) | Bottom | Low point |
| 6 (Take) | Ascending | Rising (with cost) |
| 7-8 (Return/Change) | Top | High (transformed) |

But the Circle is *more structured* than the arc—it prescribes *what happens at each position*, not just emotional valence. This suggests the Circle operates at a higher categorical level than the arc.

### 1.5 Structural Relationships: Circle, Shapes, and Kishōtenketsu

We now have three levels of narrative structure to relate:

| Structure | Acts/Steps | Primary Driver | Cultural Origin |
|-----------|------------|----------------|-----------------|
| Reagan Shapes | Continuous curve | Emotional valence | Empirical (Western corpus) |
| Harmon Circle | 8 discrete steps | Transformation via trial | Western (Campbell tradition) |
| Kishōtenketsu | 4 acts | Epistemic reframing | East Asian |

**Hypothesis 1**: The Harmon Circle is a *generator* for Reagan shapes.

Different weightings and iterations of the Circle produce different shapes:
- One full traversal → Man-in-a-hole
- Truncated at step 5 → Tragedy (descent without return)
- Starting at step 5 → Rags-to-riches (ascent only)
- Two traversals (varied intensity) → Cinderella or Oedipus

**Hypothesis 2**: Harmon and Kishōtenketsu are *non-isomorphic* structures.

Despite both having cyclic/four-part aspects, they differ fundamentally:

| Aspect | Harmon Circle | Kishōtenketsu |
|--------|---------------|---------------|
| Conflict | Central (steps 3-6 are struggle) | Optional/absent |
| Transformation | Via ordeal | Via reframing |
| Resolution | Return to order | Reconciliation of perspectives |
| Emotional shape | Valley (descent-ascent) | Plateau with spike |
| Protagonist | Active agent | Often passive observer |

**Hypothesis 3**: There exists a *higher* structure from which both derive.

Both Harmon and kishōtenketsu may be projections of a more abstract "transformation schema"—a universal pattern of *state change* that different cultures instantiate differently. Finding this schema is a goal of our research.

---

## 2. A Category-Theoretic Framework

### 2.1 The Category **Narr**

Define a category **Narr** where:

- **Objects** are *narrative states*: configurations of (characters, world-state, reader knowledge, tension level, thematic position)
  
- **Morphisms** are *narrative beats* or *scenes*: functions that transform one state into another
  
- **Composition** is temporal sequencing: if `f: A → B` and `g: B → C`, then `g ∘ f: A → C` is the combined narrative movement

- **Identity** `id_A: A → A` represents "nothing happens"—a static scene

A complete story is a morphism `S: Initial → Final` in **Narr**.

### 2.2 The Harmon Circle as a Diagram Category

We can formalize the Harmon Circle as a **diagram** H: **8** → **Narr**, where **8** is the cyclic category with 8 objects and morphisms forming a cycle.

Define the indexing category **Circle₈**:

```
Objects: {1, 2, 3, 4, 5, 6, 7, 8}

Morphisms: 
  step₁: 1 → 2  (Comfort → Need)
  step₂: 2 → 3  (Need → Go)
  step₃: 3 → 4  (Go → Search)
  step₄: 4 → 5  (Search → Find)
  step₅: 5 → 6  (Find → Take)
  step₆: 6 → 7  (Take → Return)
  step₇: 7 → 8  (Return → Change)
  step₈: 8 → 1  (Change → Comfort')  [optional: for sequels/series]
  
Plus all compositions and identities.
```

A **Harmon story** is a functor H: **Circle₈** → **Narr** that:
- Maps each object i to a narrative state H(i)
- Maps each step morphism to a scene/beat transforming states
- Respects composition: H(stepⱼ ∘ stepᵢ) = H(stepⱼ) ∘ H(stepᵢ)

The **descent/ascent structure** is captured by a functor D: **Circle₈** → **2**, where **2** = {Order, Chaos}:

```
D(1) = D(2) = D(7) = D(8) = Order
D(3) = D(4) = D(5) = D(6) = Chaos
```

The **threshold crossings** (steps 2→3 and 6→7) are the only morphisms where D changes value.

### 2.3 Kishōtenketsu as a Diagram Category

Similarly, kishōtenketsu defines a diagram K: **4** → **Narr**:

```
Objects: {Ki, Shō, Ten, Ketsu}

Morphisms:
  intro: Ki → Shō    (Introduction → Development)
  twist: Shō → Ten   (Development → Twist)
  resolve: Ten → Ketsu (Twist → Reconciliation)
  
Plus compositions and identities.
```

But unlike **Circle₈**, the kishōtenketsu diagram has a distinguished morphism: `twist` is marked as the **epistemic peak**. We can formalize this with a functor E: **4** → **ℝ** measuring epistemic intensity:

```
E(Ki) = 0     (baseline)
E(Shō) = 0.3  (building understanding)
E(Ten) = 1.0  (maximum reframing)
E(Ketsu) = 0.5 (settled new understanding)
```

The shape is *not* a valley but a **ramp with spike**:

```
E(t)
  1.0 │         ╱╲
      │        ╱  ╲
  0.5 │       ╱    ╲___
  0.3 │    __╱
    0 │___╱
      └─────────────────
        Ki  Shō  Ten  Ketsu
```

### 2.4 Relating Harmon and Kishōtenketsu via Natural Transformation

Can we relate Harmon and kishōtenketsu structurally? 

Define a **coarsening functor** C: **Circle₈** → **4** that collapses the eight steps into four acts:

```
C(1) = C(2) = Ki       (setup)
C(3) = C(4) = Shō      (development)
C(5) = C(6) = Ten      (crisis/twist)
C(7) = C(8) = Ketsu    (resolution)
```

This gives us a commutative triangle:

```
              Circle₈
               / |
              /  |
           H /   | C
            /    |
           ↓     ↓
         Narr ←── 4
              K
```

The question: Does H factor through K? That is, is there a natural transformation η: H ⇒ K ∘ C?

**Answer**: Not in general. The Harmon Circle encodes *more* structure than kishōtenketsu:

1. **Conflict requirement**: H(3→4→5→6) must involve struggle; K has no such requirement
2. **Agent requirement**: H requires protagonist *action*; K allows passive observation
3. **Return structure**: H(7→8) is about *returning changed*; K(Ketsu) is about *reconciling perspectives*

However, we can define a **forgetful functor** U: **Harmon-Stories** → **Kish-Stories** that:
- Forgets conflict structure
- Forgets agent/patient distinctions  
- Maps return→change to reconciliation

This functor has a **left adjoint** (free Harmon structure on a kishōtenketsu skeleton), which explains why kishōtenketsu stories can be "adapted" to Western structure by adding conflict—but something is lost in translation.

### 2.5 Observation Functors

The key insight: *We cannot observe* **Narr** *directly*. We only have access to its image under various **observation functors**.

Define a family of functors F_i: **Narr** → **Trajectory**, where **Trajectory** is a category of time-indexed paths in some measurement space.

#### F_sentiment: Emotional Valence Functor

Maps narrative states to positions on a happiness-sadness axis. This is the Reagan et al. functor.

```
F_sentiment(state) = Σ (word ∈ state.text) sentiment_score(word) / |state.text|
```

**Image**: The six shapes are Im(F_sentiment) restricted to the Gutenberg corpus.

#### F_arousal: Activation/Tension Functor

Maps to positions on a calm-excited axis. Captures "action" independent of valence.

```
F_arousal(state) = Σ (word ∈ state.text) arousal_score(word) / |state.text|
```

#### F_epistemic: Information/Surprise Functor  

Maps to positions on a certainty-uncertainty axis. Captures revelation and mystery.

```
F_epistemic(state) = -log P(state | previous_states)  // Surprisal
```

This can be computed using language model perplexity over sliding windows.

#### F_social: Relationship Network Functor

Maps to graph structures. Captures social dynamics.

```
F_social(state) = Graph(characters, interactions_in_state)
```

#### F_thematic: Conceptual Distance Functor

Maps to positions in a semantic embedding space.

```
F_thematic(state) = centroid(embeddings(key_concepts_in_state))
```

### 2.6 The Multi-Functor Hypothesis

**Claim**: Different narrative traditions optimize different functors.

| Tradition | Primary Functor | Secondary | Structure |
|-----------|-----------------|-----------|-----------|
| Western (conflict-driven) | F_sentiment | F_arousal | Three-act |
| Harmon Circle (TV/film) | F_sentiment | F_social | Eight-step cycle |
| Kishōtenketsu | F_epistemic | F_thematic | Four-act |
| Telenovela | F_social | F_sentiment | Serial |
| Slice-of-life | F_thematic | — | Episodic |

The "six shapes" are the archetypal forms in Im(F_sentiment). But Im(F_epistemic) may have *different* archetypal forms—perhaps corresponding to kishōtenketsu's structure.

### 2.7 Natural Transformations Between Functors

Given two functors F, G: **Narr** → **Trajectory**, a natural transformation η: F ⇒ G consists of morphisms η_S: F(S) → G(S) for each story S, such that the appropriate diagrams commute.

**Example**: Tolstoy's *The Death of Ivan Ilyich*

- F_sentiment(Ivan Ilyich) = Tragedy (steady fall—physical deterioration)
- F_epistemic(Ivan Ilyich) = Rags-to-Riches (steady rise—spiritual awakening)

These are *anti-correlated*. The natural transformation between them captures the novella's central irony: dying is the path to truly living.

### 2.8 Stories as Cones and Limits

For complex narratives with multiple threads (e.g., War and Peace), model the story as a **diagram** D: J → **Narr** where J is an indexing category (characters, plotlines).

The novel as experienced is the **limit** lim D—the universal object that maps consistently to all threads.

Observation functors then give us **cones** over the observed trajectories:

```
        lim F∘D
         /|\
        / | \
    F(thread₁) F(thread₂) F(thread₃)
```

Different readers may construct different limits depending on which threads they weight—explaining divergent interpretations.

---

## 3. Research Design

### 3.1 Corpus Selection as Fiber Product

We construct our study corpus as a **fiber product** over three base categories:

#### Temporal Fiber
- **Classical**: Pre-1900 (overlap with Gutenberg)
- **Modern**: 1900-1980
- **Contemporary**: 1980-2010  
- **Web-native**: 2010-present

#### Cultural Fiber
- **Western-published**: Traditional English/European publishing
- **Japanese**: Light novels, web novels (Narou/Syosetu), published fiction
- **Korean**: Web novels (Munpia, Kakao Page), webtoons
- **Chinese**: Web fiction (Qidian, Jinjiang), published fiction
- **Fan-produced**: AO3, Wattpad (mixed cultural origins)

#### Editorial Fiber
- **Gatekept**: Traditional publishing
- **Self-published**: Amazon KDP, web platforms
- **Community-curated**: Fan fiction with kudos/comments
- **Raw**: Minimally filtered web uploads

The fiber product gives us a principled sampling space:

```
Corpus ⊆ Temporal ×_{Base} Cultural ×_{Base} Editorial
```

### 3.2 Concrete Data Sources

| Source | Est. Size | Languages | Accessibility |
|--------|-----------|-----------|---------------|
| Project Gutenberg | ~60K | English+ | Public domain |
| AO3 | 12M+ works | Multi | TOS permits scraping |
| Fanfiction.net | ~10M works | Multi | TOS permits scraping |
| Syosetu (小説家になろう) | 1M+ works | Japanese | TOS permits |
| Qidian (起点中文网) | 10M+ works | Chinese | Restricted |
| Wattpad | 90M+ works | Multi | API available |
| OpenSubtitles | 4M+ films | Multi | Research access |
| IMDB Synopses | 500K+ | English | Scrapable |

### 3.3 Sampling Strategy

For each cell in the fiber product with sufficient data:

1. **Random sample**: 1,000 works (controls for popularity bias)
2. **Popularity-weighted sample**: 1,000 works (captures cultural salience)
3. **Stratified by length**: Short (<10K words), Medium (10-50K), Long (>50K)

Target: ~50,000 works across all fibers.

### 3.4 Multi-Functor Pipeline

For each work in corpus:

```
┌─────────────────────────────────────────────────────────────┐
│  Raw Text                                                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Preprocessing                                               │
│  - Sentence segmentation                                     │
│  - Sliding window (1000 tokens, 500 overlap)                │
│  - Normalization (0-1 on narrative time axis)               │
└───────────────────────────┬─────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┬─────────────────┐
            ▼               ▼               ▼                 ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ F_sentiment  │ │ F_arousal    │ │ F_epistemic  │ │ F_thematic   │
    │              │ │              │ │              │ │              │
    │ VADER/labMT  │ │ NRC-VAD      │ │ GPT-2        │ │ Sentence     │
    │ + BERT       │ │ Lexicon      │ │ Surprisal    │ │ Transformers │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │                │
           ▼                ▼                ▼                ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Time Series per Functor                                      │
    │  [(t₁, v₁), (t₂, v₂), ..., (tₙ, vₙ)]                         │
    └───────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Shape Extraction                                             │
    │  - Smoothing (Savitzky-Golay / Fourier)                      │
    │  - Peak/valley detection                                      │
    │  - DTW distance matrix                                        │
    └───────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Clustering (per functor, per corpus fiber)                   │
    │  - Hierarchical clustering                                    │
    │  - K-means with k selection via silhouette                   │
    │  - t-SNE/UMAP visualization                                  │
    └──────────────────────────────────────────────────────────────┘
```

### 3.5 Analysis Questions

#### Q1: Shape Stability Across Functors

For a given corpus, do the same works cluster together under different functors?

**Method**: Compute adjusted Rand index between clusterings:
```
ARI(Clustering_F_sentiment, Clustering_F_epistemic)
```

**Hypothesis**: Low ARI indicates functors capture orthogonal structure.

#### Q2: Shape Stability Across Cultures

For a given functor, do the same archetypal shapes emerge across cultural fibers?

**Method**: Train classifier on Western corpus, test on Japanese/Korean/Chinese.

**Hypothesis**: Poor cross-cultural transfer indicates culturally-specific shapes.

#### Q3: Functor Optimization by Tradition

Do different traditions show tighter clustering under different functors?

**Method**: Compare average within-cluster variance:
```
Var_Western(F_sentiment) vs Var_Western(F_epistemic)
Var_Japanese(F_sentiment) vs Var_Japanese(F_epistemic)
```

**Hypothesis**: Western stories cluster tighter under F_sentiment; kishōtenketsu-influenced stories cluster tighter under F_epistemic.

#### Q4: Temporal Drift

Have dominant shapes changed over time within a tradition?

**Method**: Train era-specific classifiers, examine shape distribution shifts.

**Hypothesis**: Web-native fiction shows different shape distributions than gatekept fiction.

#### Q5: Editorial Effect

Does editorial gatekeeping select for specific shapes?

**Method**: Compare published vs. fan-produced within same fandom.

**Hypothesis**: Published fiction shows less variance, fewer "non-standard" shapes.

#### Q6: Harmon Circle Detection and Decomposition

Can we detect Harmon Circle structure in narratives, and does it predict shape classification?

**Method**: 
1. Train a classifier to identify 8-step structure in stories
2. Compute "Harmon completeness score" (how many steps present, in order)
3. Correlate with Reagan shape classification

**Hypothesis**: Stories with high Harmon completeness will cluster as "man-in-a-hole" under F_sentiment. Stories with partial Harmon structure will show predictable shape variants (truncated = tragedy, inverted = rags-to-riches).

#### Q7: Cross-Structural Adaptation

When kishōtenketsu stories are adapted for Western audiences, what transformations occur?

**Method**: Compare original Japanese/Korean works with their Western adaptations (remakes, translations, localizations).

**Hypothesis**: Adaptations will show "Harmonification"—insertion of conflict beats, agent transformation, and return structure—visible as changes in functor trajectories.

---

## 4. Technical Implementation

### 4.1 Functor Implementations

#### F_sentiment (Python pseudocode)

```python
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentFunctor:
    def __init__(self, method='ensemble'):
        self.vader = SentimentIntensityAnalyzer()
        self.bert = pipeline('sentiment-analysis', 
                            model='nlptown/bert-base-multilingual-uncased-sentiment')
    
    def __call__(self, windows: List[str]) -> np.ndarray:
        """Map list of text windows to sentiment trajectory."""
        scores = []
        for w in windows:
            vader_score = self.vader.polarity_scores(w)['compound']
            bert_score = self._bert_to_scalar(self.bert(w[:512])[0])
            scores.append((vader_score + bert_score) / 2)
        return np.array(scores)
```

#### F_epistemic (using surprisal)

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class EpistemicFunctor:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        
    def __call__(self, windows: List[str]) -> np.ndarray:
        """Map text windows to surprisal (negative log probability)."""
        surprisals = []
        context = ""
        for w in windows:
            inputs = self.tokenizer(context + w, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
            # Perplexity of window given context
            surprisals.append(outputs.loss.item())
            context = (context + w)[-2048:]  # Sliding context window
        return np.array(surprisals)
```

#### F_thematic (semantic trajectory)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class ThematicFunctor:
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
    
    def __call__(self, windows: List[str]) -> np.ndarray:
        """Map text windows to trajectory in embedding space."""
        embeddings = self.model.encode(windows)
        # Compute distance from initial position
        distances = np.linalg.norm(embeddings - embeddings[0], axis=1)
        return distances
```

### 4.2 Harmon Circle Detection

```python
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class HarmonStep:
    """Represents one step of the Harmon Circle."""
    name: str
    keywords: List[str]
    position: str  # 'order' or 'chaos'
    sentiment_expected: str  # 'high', 'falling', 'low', 'rising'

HARMON_STEPS = [
    HarmonStep("comfort", ["home", "routine", "normal", "everyday"], "order", "high"),
    HarmonStep("need", ["want", "desire", "lack", "need", "problem"], "order", "high"),
    HarmonStep("go", ["leave", "enter", "cross", "journey", "venture"], "chaos", "falling"),
    HarmonStep("search", ["struggle", "try", "fail", "adapt", "learn"], "chaos", "falling"),
    HarmonStep("find", ["discover", "achieve", "obtain", "reach", "find"], "chaos", "low"),
    HarmonStep("take", ["cost", "price", "sacrifice", "lose", "pay"], "chaos", "rising"),
    HarmonStep("return", ["back", "return", "home", "escape", "leave"], "order", "rising"),
    HarmonStep("change", ["different", "changed", "new", "transform", "grow"], "order", "high"),
]

class HarmonDetector:
    def __init__(self, embedding_model):
        self.model = embedding_model
        self.step_embeddings = self._compute_step_embeddings()
    
    def _compute_step_embeddings(self) -> np.ndarray:
        """Create embedding anchors for each Harmon step."""
        step_texts = [" ".join(step.keywords) for step in HARMON_STEPS]
        return self.model.encode(step_texts)
    
    def detect_steps(self, windows: List[str]) -> List[Tuple[int, float]]:
        """
        For each text window, identify closest Harmon step.
        Returns list of (step_index, confidence) tuples.
        """
        window_embeddings = self.model.encode(windows)
        results = []
        for emb in window_embeddings:
            similarities = np.dot(self.step_embeddings, emb)
            best_step = np.argmax(similarities)
            confidence = similarities[best_step]
            results.append((best_step, confidence))
        return results
    
    def compute_harmon_score(self, windows: List[str]) -> dict:
        """
        Compute how well a narrative follows Harmon structure.
        """
        detections = self.detect_steps(windows)
        steps_detected = [d[0] for d in detections]
        
        # Check for sequential ordering
        order_score = self._compute_order_score(steps_detected)
        
        # Check for completeness (all 8 steps present)
        unique_steps = set(steps_detected)
        completeness = len(unique_steps) / 8.0
        
        # Check for proper position (order vs chaos)
        position_score = self._compute_position_score(steps_detected, len(windows))
        
        return {
            'order_score': order_score,
            'completeness': completeness,
            'position_score': position_score,
            'harmon_total': (order_score + completeness + position_score) / 3,
            'step_sequence': steps_detected
        }
    
    def _compute_order_score(self, steps: List[int]) -> float:
        """Measure how sequential the step detections are."""
        if len(steps) < 2:
            return 0.0
        correct_transitions = 0
        for i in range(len(steps) - 1):
            if steps[i+1] == (steps[i] + 1) % 8 or steps[i+1] == steps[i]:
                correct_transitions += 1
        return correct_transitions / (len(steps) - 1)
    
    def _compute_position_score(self, steps: List[int], total_windows: int) -> float:
        """Check if order/chaos steps appear in correct halves."""
        midpoint = total_windows // 2
        correct = 0
        for i, step in enumerate(steps):
            expected_position = HARMON_STEPS[step].position
            actual_position = "order" if (i < midpoint * 0.3 or i > midpoint * 1.7) else "chaos"
            if expected_position == actual_position:
                correct += 1
        return correct / len(steps) if steps else 0.0


class KishotenketsuDetector:
    """Detect four-act kishōtenketsu structure."""
    
    def __init__(self, epistemic_functor):
        self.F_epistemic = epistemic_functor
    
    def compute_score(self, windows: List[str]) -> dict:
        """
        Detect kishōtenketsu structure via epistemic arc analysis.
        """
        # Compute epistemic trajectory (surprisal over narrative time)
        trajectory = self.F_epistemic(windows)
        n = len(trajectory)
        
        # Kishōtenketsu predicts: flat-ish for Ki/Shō, spike at Ten (~75%), settle at Ketsu
        # Divide into four quarters
        q1 = trajectory[:n//4]           # Ki
        q2 = trajectory[n//4:n//2]       # Shō  
        q3 = trajectory[n//2:3*n//4]     # Ten
        q4 = trajectory[3*n//4:]         # Ketsu
        
        # Check for expected pattern
        ki_sho_flat = np.std(np.concatenate([q1, q2])) < np.std(trajectory) * 0.8
        ten_spike = np.max(q3) > np.mean(np.concatenate([q1, q2])) + np.std(trajectory)
        ketsu_settle = np.mean(q4) < np.max(q3) and np.mean(q4) > np.mean(q1)
        
        # Find the exact position of the "ten" (twist)
        ten_position = (np.argmax(trajectory) / n) if len(trajectory) > 0 else 0
        ten_in_expected_range = 0.5 < ten_position < 0.85
        
        # Compute overall score
        scores = [ki_sho_flat, ten_spike, ketsu_settle, ten_in_expected_range]
        kish_total = sum(scores) / len(scores)
        
        return {
            'ki_sho_flat': ki_sho_flat,
            'ten_spike': ten_spike,
            'ten_position': ten_position,
            'ketsu_settle': ketsu_settle,
            'kish_total': kish_total,
            'trajectory': trajectory
        }


class StructuralComparator:
    """Compare Harmon and kishōtenketsu structure in narratives."""
    
    def __init__(self, embedding_model, epistemic_functor):
        self.harmon = HarmonDetector(embedding_model)
        self.kish = KishotenketsuDetector(epistemic_functor)
    
    def analyze(self, windows: List[str]) -> dict:
        """Full structural analysis."""
        h_result = self.harmon.compute_harmon_score(windows)
        k_result = self.kish.compute_score(windows)
        
        # Compute H-K spectrum position
        h_score = h_result['harmon_total']
        k_score = k_result['kish_total']
        
        # Position: -1 (pure Harmon) to +1 (pure kishōtenketsu)
        if h_score + k_score > 0:
            hk_position = (k_score - h_score) / (h_score + k_score)
        else:
            hk_position = 0.0
        
        return {
            'harmon': h_result,
            'kishotenketsu': k_result,
            'hk_spectrum_position': hk_position,
            'structure_type': self._classify_structure(h_score, k_score, hk_position)
        }
    
    def _classify_structure(self, h: float, k: float, pos: float) -> str:
        if h > 0.7 and k < 0.3:
            return "harmon_dominant"
        elif k > 0.7 and h < 0.3:
            return "kishotenketsu_dominant"
        elif h > 0.5 and k > 0.5:
            return "hybrid_both"
        elif pos > 0.3:
            return "kishotenketsu_leaning"
        elif pos < -0.3:
            return "harmon_leaning"
        else:
            return "ambiguous"
```

### 4.3 Shape Clustering

```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from fastdtw import fastdtw

def cluster_trajectories(trajectories: List[np.ndarray], n_clusters: int) -> np.ndarray:
    """Cluster trajectories using DTW distance."""
    # Normalize all trajectories to same length
    normalized = [np.interp(np.linspace(0, 1, 100), 
                           np.linspace(0, 1, len(t)), t) 
                  for t in trajectories]
    
    # Compute pairwise DTW distances
    n = len(normalized)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d, _ = fastdtw(normalized[i], normalized[j])
            distances[i, j] = distances[j, i] = d
    
    # Hierarchical clustering
    condensed = pdist(distances)
    Z = linkage(condensed, method='ward')
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    
    return labels
```

### 4.4 Cross-Cultural Transfer Test

```python
from sklearn.metrics import adjusted_rand_score, classification_report

def cross_cultural_transfer(
    source_corpus: Corpus,
    target_corpus: Corpus,
    functor: Functor,
    n_clusters: int = 6
) -> dict:
    """Test if shape taxonomy transfers across cultures."""
    
    # Extract trajectories
    source_trajs = [functor(work) for work in source_corpus]
    target_trajs = [functor(work) for work in target_corpus]
    
    # Cluster source corpus
    source_labels = cluster_trajectories(source_trajs, n_clusters)
    
    # Train classifier on source
    clf = train_trajectory_classifier(source_trajs, source_labels)
    
    # Predict on target
    target_preds = clf.predict(target_trajs)
    
    # Also cluster target natively
    target_native_labels = cluster_trajectories(target_trajs, n_clusters)
    
    return {
        'source_to_target_predictions': target_preds,
        'target_native_clusters': target_native_labels,
        'transfer_coherence': adjusted_rand_score(target_preds, target_native_labels),
        'cluster_distribution_source': np.bincount(source_labels),
        'cluster_distribution_target_pred': np.bincount(target_preds),
        'cluster_distribution_target_native': np.bincount(target_native_labels)
    }
```

---

## 5. Expected Findings and Implications

### 5.1 Predicted Results

Based on theoretical considerations:

1. **F_sentiment shapes will NOT transfer cleanly to Japanese/Korean corpora**
   - Kishōtenketsu-influenced works will show flatter emotional arcs
   - They will cluster as "anomalies" under Western-trained classifiers

2. **F_epistemic will reveal kishōtenketsu structure**
   - Four-cluster solution will emerge naturally
   - The "ten" (twist) will appear as a surprisal spike at ~75% position

3. **Fan fiction will show hybrid shapes**
   - Anime/manga fandoms will show kishōtenketsu influence even in English
   - Western-source fandoms will show modified Reagan shapes (more "comfort" endings)

4. **Web-native fiction will diverge from gatekept**
   - Serial structures will show oscillating patterns (cliffhangers)
   - Less pressure toward "resolution" = more open endings

5. **Multi-functor analysis will reveal hidden structure in Tolstoy**
   - War and Peace: Multiple overlapping arcs that cohere only at the limit
   - Anna Karenina: Contravariant sentiment/epistemic trajectories

### 5.2 Category-Theoretic Implications

If different traditions optimize different functors, this suggests:

**Narrative universals exist at the level of** ***Narr*** **itself, not at the level of any particular functor's image.**

The "six shapes" and kishōtenketsu are both *shadows* of a richer underlying structure. Different cultures have developed different *projection operators* that emphasize different aspects of narrative transformation.

This is analogous to how:
- A sphere projects to a circle from one angle, an ellipse from another
- Both are valid "shapes of a sphere" but neither is *the* shape
- The sphere itself is the universal object

**Conjecture**: There exists a "narrative sphere"—a universal structure of transformation—and cultural traditions are projection functors from this sphere to human experience.

### 5.3 Practical Applications

1. **Cross-cultural translation/adaptation**
   - Identify which functor the source optimizes
   - Restructure to optimize appropriate target-culture functor
   - Explains why some adaptations "feel wrong"

2. **AI story generation**
   - Current models trained on Western corpora optimize F_sentiment
   - To generate kishōtenketsu stories, train/fine-tune on F_epistemic objective

3. **Recommendation systems**
   - Match readers to stories not just by content but by shape preference
   - Users who like kishōtenketsu anime may prefer literary fiction with twist structures

4. **Writing pedagogy**
   - Teach multiple functor awareness
   - "Your story works emotionally but lacks epistemic arc" = actionable feedback

---

## 6. Limitations and Future Work

### 6.1 Methodological Limitations

- **Lexicon-based sentiment** may not capture irony, subtext, or cultural idiom
- **LLM surprisal** reflects model training distribution, not human expectation
- **Translation effects** may distort cross-linguistic comparison
- **Length normalization** may obscure pacing differences

### 6.2 Extensions

1. **Video/Film functor**: Extend to audiovisual using multimodal models
2. **Interactive narrative**: Games, choose-your-own-adventure → branching morphisms
3. **Oral tradition**: Record and analyze performed storytelling
4. **Historical depth**: Ancient/medieval texts across cultures
5. **Reader response**: Correlate functor trajectories with empirical reading experience data

### 6.3 The Deeper Question

Ultimately, this research asks: **Is there a universal grammar of narrative?**

Chomsky proposed universal grammar for language. Propp proposed universal functions for folktales. Campbell proposed universal hero's journey.

Category theory offers a rigorous framework to ask: What is the *minimal* structure that all narratives share? What are the *functors* that different cultures have evolved to extract meaning from that structure?

The six shapes and kishōtenketsu may both be *local optima* in a vast space of possible narrative projections—solutions that different cultures found to the problem of making transformations meaningful.

---

## 7. References

Reagan, A. J., Mitchell, L., Kiley, D., Danforth, C. M., & Dodds, P. S. (2016). The emotional arcs of stories are dominated by six basic shapes. *EPJ Data Science*, 5(1), 31.

Vonnegut, K. (1981). *Palm Sunday: An Autobiographical Collage*. Delacorte Press.

Propp, V. (1968). *Morphology of the Folktale*. University of Texas Press.

Campbell, J. (1949). *The Hero with a Thousand Faces*. Pantheon Books.

Harmon, D. (2009). Story Structure 101: Super Basic Shit. Channel 101 Wiki.

Harmon, D. (2011). Story Structure 104: The Juicy Details. Channel 101 Wiki.

Vogler, C. (2007). *The Writer's Journey: Mythic Structure for Writers* (3rd ed.). Michael Wiese Productions.

Jockers, M. L. (2015). Syuzhet: Extract Sentiment and Plot Arcs from Text. R package.

Kim, Y. M. (2017). Worldwide Story Structures. Blog post.

Hayashida, K. (2011). Mario Level Design via Kishōtenketsu. GDC Presentation.

Mac Lane, S. (1978). *Categories for the Working Mathematician*. Springer.

Fong, B., & Spivak, D. I. (2019). *An Invitation to Applied Category Theory: Seven Sketches in Compositionality*. Cambridge University Press.

Bakhtin, M. M. (1981). *The Dialogic Imagination*. University of Texas Press.

Moretti, F. (2005). *Graphs, Maps, Trees: Abstract Models for a Literary History*. Verso.

---

## Appendix A: Tolstoy Case Studies

### A.1 Anna Karenina: Fiber Analysis

Anna Karenina presents two primary narrative threads:

**Thread A (Anna)**: 
- F_sentiment: Icarus (rise-fall)
- F_social: Network collapse (connection → isolation)
- F_thematic: Circular return (Moscow train bookends)

**Thread L (Levin)**:
- F_sentiment: Man-in-a-hole (fall-rise), but noisy
- F_epistemic: Steady rise (accumulating insight)
- F_thematic: Linear journey (doubt → faith)

The novel's structure is a **pullback**:

```
            Anna Karenina
               /    \
              /      \
           π_A      π_L
            /          \
     Thread A        Thread L
            \          /
             \        /
              \      /
            Oblonsky
          (shared base)
```

Oblonsky functions as the base object—his domestic crisis opens the novel, and his presence marks the intersection points. The novel is coherent because both threads project consistently to scenes involving Oblonsky.

### A.2 War and Peace: Colimit Structure

War and Peace is better modeled as a **colimit** over a diagram of threads:

- Pierre's arc
- Andrei's arc
- Natasha's arc
- The Rostovs (collective)
- The war (impersonal)
- Historical philosophy (authorial)

The novel provides **cocone** morphisms from each thread to the whole, but unlike Anna Karenina, there's no single "Oblonsky" base object. Instead, coherence comes from:

1. **Temporal grounding**: All threads indexed to historical dates
2. **Spatial proximity**: Characters move through same physical/social spaces
3. **Thematic resonance**: Each thread reflects on fate/agency/history

Tolstoy's philosophical essays function as **naturality conditions**—they make explicit the transformations that allow different threads to cohere.

### A.3 The Death of Ivan Ilyich: Contravariant Duality

This novella is the clearest case of **contravariant functors**:

```
                   Narr
                  /    \
         F_physical    F_spiritual
              |              |
              ▼              ▼
         Trajectory      Trajectory
              |              |
           Fall           Rise
```

The functors are contravariant: as F_physical decreases (health → death), F_spiritual increases (death → life). The novella's power comes from this precise anti-correlation.

At the terminal object (Ivan's death), the functors converge:

```
lim(F_physical) = lim(F_spiritual) = point of transcendence
```

This is a **categorical reconciliation of opposites**—the novel constructs a narrative space where physical death and spiritual birth are the *same morphism* viewed through different projections.

### A.4 Tolstoy on the H-K Spectrum

Interestingly, Tolstoy's works resist clean classification on either the Harmon or kishōtenketsu model:

**Anna Karenina: Dual Structure**

| Thread | Harmon Score (predicted) | Kishōtenketsu Score | Notes |
|--------|--------------------------|---------------------|-------|
| Anna | 0.6 (partial) | 0.3 | Steps 1-6 complete, 7-8 inverted (no return/change, only destruction) |
| Levin | 0.4 (weak) | 0.7 | Minimal conflict; epistemic arc dominates; "ten" = peasant's words |

The novel as a whole is **structurally hybrid**: Anna's thread provides Harmon-style conflict and descent, while Levin's thread provides kishōtenketsu-style epistemic transformation. The reader experiences both simultaneously.

**War and Peace: Beyond Both Models**

War and Peace exceeds both structures because:

1. **Multiple simultaneous circles**: Pierre, Andrei, Natasha each traverse partial Harmon circles at different phases
2. **Historical "ten"**: The 1812 invasion functions as kishōtenketsu's twist—it reframes everything
3. **Philosophical ketsu**: Tolstoy's epilogue essays are explicit reconciliation/synthesis

Predicted structural scores:
- Harmon (any single thread): ~0.5 (incomplete circles)
- Kishōtenketsu (any single thread): ~0.4 (twist present but resolution unclear)
- **Combined/colimit structure**: Neither model captures adequately

This suggests Tolstoy intuitively discovered a **higher structure** that subsumes both Western and Eastern models—a polyphonic narrative architecture that our categorical framework can begin to formalize as a diagram in **Narr** rather than a single morphism.

**The Death of Ivan Ilyich: Anti-Harmon**

| Harmon Step | Present? | Notes |
|-------------|----------|-------|
| 1. Comfort | ✓ | Ivan's successful bourgeois life |
| 2. Need | ✗ | Ivan has no want—he's satisfied |
| 3. Go | ✗ | Ivan doesn't choose to enter chaos; it invades (illness) |
| 4. Search | ✓ (inverted) | Ivan searches for cure, meaning—but passively |
| 5. Find | ✓ | Ivan finds truth only at death |
| 6. Take | ✗ | No sacrifice—death is imposed |
| 7. Return | ✗ | No return possible |
| 8. Change | ✓ | Transformation occurs, but too late for worldly return |

Harmon score: ~0.35 (structure violated, not completed)
Kishōtenketsu score: ~0.6 (epistemic spike at death, but no ketsu/reconciliation for characters)

The novella is **anti-Harmon**: it systematically violates the expected structure to create its devastating effect. The reader expects return and change; Ivan gets only death. This makes it a critical test case for structural detection—a high-quality work that deliberately resists the dominant pattern.

---

## Appendix B: Harmon Circle Case Studies

### B.1 The Circle in Television: Community and Rick and Morty

Dan Harmon developed his Story Circle explicitly for television writing. Each episode of *Community* was designed to traverse the full circle, making it an ideal test corpus.

**Example: "Remedial Chaos Theory" (Community S3E04)**

This episode creates seven alternate timelines from one moment (rolling a die to determine who gets pizza). Each timeline is a *partial* Harmon traversal:

| Timeline | Steps Completed | Outcome |
|----------|-----------------|---------|
| Timeline 1 (Abed) | 1-2-3 only | Truncated (Abed removed) |
| Timeline 2 (Shirley) | 1-8 compressed | Full but rushed |
| Timeline 3 (Pierce) | 1-6 (death) | Tragedy variant |
| Timeline 4 (Britta) | 1-5-skip-8 | Missing sacrifice |
| Timeline 5 (Troy) | 1-8 full | "Prime" timeline |
| Timeline 6 (Annie) | 1-6 repeated | Loop/trap |
| Timeline 7 ("Darkest") | 1-4 only | Chaos without return |

The episode is a **diagram of circles**—a functor from the timeline category to **Circle₈**. The "prime" timeline is distinguished by being the unique complete traversal.

**Rick and Morty: Subverting the Circle**

The show often *inverts* or *corrupts* the Harmon structure for nihilistic effect:

- Rick frequently completes steps 1-6 but *refuses* step 7 (return) or 8 (change)
- Episodes may start at step 5 (already in chaos) with no setup
- "Change" is often ironic or undone by next episode (serial reset)

This makes Rick and Morty a test case for **anti-Harmon** detection: Can our system identify deliberate structural violations?

### B.2 Harmon Structure in Film: Comparative Analysis

We can compare Harmon-adherent films with non-Harmon structures:

**High Harmon Adherence** (predicted):
- *The Matrix* (1999): Neo's journey maps cleanly to all 8 steps
- *Finding Nemo* (2003): Marlin's transformation is textbook circle
- *Get Out* (2017): Chris's descent and return with change

**Low Harmon Adherence** (predicted):
- *Spirited Away* (2001): Kishōtenketsu structure, minimal conflict drive
- *Lost in Translation* (2003): No clear "take" or "return" beats
- *Jeanne Dielman* (1975): Anti-narrative, steps deliberately absent

**Hybrid/Complex**:
- *Parasite* (2019): Kishōtenketsu "ten" (twist) mapped onto Western thriller beats
- *Arrival* (2016): Circle is present but *non-linear* in presentation
- *Memento* (2000): Circle traversed backwards

### B.3 The Harmon-Kishōtenketsu Spectrum

Rather than a binary, we can define a continuous **H-K spectrum**:

```
H ←————————————————————————————→ K

Conflict-driven          Twist-driven
Agent transformation     Perspective shift
Return to order          Reconciliation of views
Emotional valley         Epistemic spike
```

Every narrative has a position on this spectrum. Our detection system can compute:

```python
def compute_hk_position(story) -> float:
    """
    Returns value in [-1, 1].
    -1 = pure Harmon, +1 = pure kishōtenketsu, 0 = hybrid
    """
    harmon_score = harmon_detector.compute_harmon_score(story)['harmon_total']
    kish_score = kishōtenketsu_detector.compute_score(story)['kish_total']
    
    # Normalize and compute position
    h_norm = 2 * harmon_score - 1  # Map [0,1] to [-1,1]
    k_norm = 2 * kish_score - 1
    
    return (k_norm - h_norm) / 2  # Average positions
```

This allows us to:
1. Map entire corpora on the H-K spectrum
2. Track cultural drift over time
3. Identify hybrid works that draw from both traditions
4. Predict adaptation success (works near center may translate better)

### B.4 Fan Fiction and Structural Mixing

Fan fiction is particularly interesting because it often:
- Takes source material from one tradition
- Written by authors steeped in another tradition
- Published without editorial gatekeeping

**Hypothesis**: Anime/manga fanfic written by Western authors will show *structural code-switching*—alternating between Harmon and kishōtenketsu patterns, or attempting hybrid forms.

**Test**: Compare:
- Harry Potter fic (Western source, mixed authors)
- Naruto fic (Japanese source, mixed authors)  
- MCU fic (Western source, Western-dominated authors)
- BTS fic (Korean source, global authors)

We predict Naruto and BTS fic will show higher kishōtenketsu influence, but this may depend on author cultural background (detectable via username patterns, author notes, publication platform).

---

## Appendix C: Implementation Roadmap

### Phase 1: Infrastructure (Months 1-2)
- Set up corpus ingestion pipelines
- Implement functor classes with test coverage
- Build visualization dashboard for trajectories

### Phase 2: Western Baseline (Months 3-4)
- Replicate Reagan et al. on Gutenberg
- Extend to AO3 Western-source fandoms
- Validate six-shape taxonomy

### Phase 3: Cross-Cultural Collection (Months 4-6)
- Partner with Japanese/Korean/Chinese literature scholars
- Collect and preprocess non-Western corpora
- Handle translation/multilingual challenges

### Phase 4: Multi-Functor Analysis (Months 6-9)
- Run all functors on all corpora
- Clustering and shape extraction
- Cross-cultural transfer experiments

### Phase 5: Synthesis and Writing (Months 9-12)
- Statistical analysis
- Theoretical interpretation
- Paper drafting and revision

---

*Document prepared for the Categorical Solutions Architect blog series.*
*Author: [Ibrahim Cesar] | Date: [December 2025]*
