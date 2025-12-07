# Functorial Narrative Analysis: A Cross-Cultural Study of Story Shapes

---

## Abstract

Reagan et al. (2016) identified six emotional arc shapes dominating English-language fiction in the Project Gutenberg corpus. We argue this finding is potentially an artifact of temporal, cultural, and editorial selection biases—and more radically, may reflect the specific constraints of print culture itself (the "Gutenberg Parenthesis"). We propose a category-theoretic framework for narrative analysis that (1) formalizes story structure as morphisms in a category **Narr**, (2) defines multiple observation functors capturing different aspects of narrative transformation, (3) enables principled cross-cultural comparison by treating different narrative traditions as different subcategories or as optimizing different functorial objectives, and (4) situates the "six shapes" within a deeper archaeology of narrative spanning oral epic (Gilgamesh, Homer), print culture, and algorithmic/streaming-era content. We address questions including: Are narrative arcs universal or artifacts of print culture? Are contemporary stories being optimized for dopamine response patterns? What structures appear in humanity's oldest narratives? We outline a concrete research design using corpora ranging from ancient Mesopotamian epic to modern fan fiction and streaming television.

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

#### F_entropy: Shannon Information Functor

Maps to information-theoretic measures of predictability and surprise.

```
F_entropy(state) = H(P(next_state | current_state))  // Entropy over predictions
```

This functor family deserves special attention because it connects directly to Shannon's mathematical theory of communication and provides rigorous foundations for concepts like "suspense," "surprise," and "plot twist."

**Key Information-Theoretic Measures for Narrative:**

| Measure | Formula | Narrative Interpretation |
|---------|---------|-------------------------|
| **Entropy** | H(X) = -Σ p(x) log p(x) | Uncertainty about what happens next |
| **Surprisal** | I(x) = -log p(x) | How unexpected an event is |
| **Mutual Information** | I(X;Y) = H(X) - H(X\|Y) | How much knowing history helps predict future |
| **Jensen-Shannon Divergence** | JSD(P\|\|Q) | Magnitude of state change (plot pivot) |
| **Cross-Entropy** | H(P,Q) = -Σ p(x) log q(x) | Reader prediction error |

**Theoretical Foundation:**

The Uniform Information Density (UID) hypothesis proposes that effective communication distributes information evenly across an utterance. Applied to narrative, this suggests:

1. **Good pacing** = relatively uniform surprisal across the narrative
2. **Suspense** = high entropy (uncertainty about future states)
3. **Surprise/Twist** = high surprisal (low probability event occurs)
4. **Cliffhanger** = entropy spike at structural boundary
5. **Resolution** = entropy collapse (uncertainty eliminated)

Recent work (Wilmot & Keller, 2020) formalized suspense as **uncertainty reduction**—a forward-looking measure of how unexpected the continuation is. This outperformed backward-looking surprise measures in predicting human suspense judgments.

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
│  Raw Text                                                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Preprocessing                                              │
│  - Sentence segmentation                                    │
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
    │  Time Series per Functor                                     │
    │  [(t₁, v₁), (t₂, v₂), ..., (tₙ, vₙ)]                          │
    └───────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Shape Extraction                                            │
    │  - Smoothing (Savitzky-Golay / Fourier)                      │
    │  - Peak/valley detection                                     │
    │  - DTW distance matrix                                       │
    └───────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Clustering (per functor, per corpus fiber)                  │
    │  - Hierarchical clustering                                   │
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

#### Q8: The Gutenberg Parenthesis Hypothesis

Are the "six shapes" an artifact of print culture rather than universal narrative structure?

**Background**: The Gutenberg Parenthesis thesis (Pettitt, Sauerberg) argues that print culture (c. 1450-2000) represents a historical anomaly—a "parenthesis" between oral culture and digital/post-print culture. Print imposed:
- Linear, fixed sequence (no audience interaction)
- Individual, silent reading (no communal performance)
- Closure and completion (physical book has an ending)
- Author as singular authority (no collective retelling)

**Method**: 
1. Compare oral tradition narratives (folk tales, myths, oral epics) with print-era novels
2. Compare print-era novels with born-digital fiction (web serials, interactive fiction, fan fiction)
3. Analyze whether the "six shapes" appear consistently across all three eras

**Hypothesis**: The six shapes may be optimized for *print reading*—specifically, for the experience of a solitary reader who cannot influence the story and expects closure. Oral narratives may show more episodic/cyclical structures; digital narratives may show more open/branching structures.

**Test cases**:
- Pre-print: Gilgamesh, Odyssey, Mahabharata, Beowulf, 1001 Nights
- Print-canonical: Gutenberg corpus (Reagan's data)
- Post-print: Web serials (Worm, Mother of Learning), interactive fiction, serialized web novels

#### Q9: Dopamine Optimization and Neurochemical Arc Engineering

Are contemporary stories being optimized for neurochemical response patterns?

**Background**: The "attention economy" and streaming-era content creation may be selecting for narrative structures that maximize engagement metrics. This could mean stories are being engineered to trigger dopamine release at specific intervals—creating a kind of "slot machine" narrative structure.

**Observable patterns**:
- **Cliffhangers** at episode/chapter boundaries (dopamine anticipation)
- **Micro-peaks** every 5-10 minutes (TikTok-era attention spans)
- **Variable reward schedules** (unpredictable positive beats)
- **Loss aversion beats** (threatening beloved characters)

**Method**:
1. Compare arc structures across eras:
   - Pre-streaming television (1960-1995)
   - Early streaming (2007-2015)
   - Algorithmic streaming (2015-present)
2. Measure "peak density" (emotional spikes per unit narrative time)
3. Correlate with engagement metrics where available

**Hypothesis**: Contemporary streaming content shows *higher peak density* and *more variable reward scheduling* than earlier forms. The "six shapes" may be fragmenting into "micro-arc cascades"—rapid oscillations optimized for binge-watching neurochemistry.

**Neurochemical model**:
```
Traditional arc:     ____/\____/\____/\_______/\
                    (slow build to major peaks)

Streaming-optimized: /\/\/\/\/\/\/\/\/\/\/\/\/\
                    (constant micro-stimulation)
```

**Implication**: If true, this suggests the Reagan shapes describe a *historical* optimum that is being superseded by attention-economy pressures. Future dominant shapes may look very different.

#### Q10: The Gilgamesh Test—Pre-Literary Narrative Structure

Does the oldest substantial narrative in human history conform to the "six shapes"?

**Background**: The Epic of Gilgamesh (c. 2100 BCE) predates the Gutenberg corpus by ~3,500 years and originates in a completely different cultural context (Sumerian/Akkadian Mesopotamia). If the "six shapes" are truly universal, Gilgamesh should exhibit them. If Gilgamesh shows different patterns, this supports cultural specificity.

**Method**:
1. Apply all four functors (F_sentiment, F_arousal, F_epistemic, F_thematic) to Gilgamesh
2. Compare with Homeric epics (Iliad, Odyssey)—intermediate age, oral-formulaic tradition
3. Compare with Sanskrit epics (Mahabharata, Ramayana)—different cultural origin
4. Compare with Norse sagas, Chinese classics (Journey to the West, Romance of Three Kingdoms)

**Structural analysis of Gilgamesh**:

| Section | Content | Predicted F_sentiment | Predicted F_epistemic |
|---------|---------|----------------------|----------------------|
| Tablets I-II | Gilgamesh as tyrant; Enkidu's creation and civilizing | Rising (friendship) | Rising (world-building) |
| Tablets III-V | Journey to Cedar Forest; defeating Humbaba | Rising then falling (victory + guilt) | Peak (confronting unknown) |
| Tablet VI | Ishtar's proposal; Bull of Heaven | Falling (hubris) | Stable |
| Tablet VII | Enkidu's death | Major fall | Epistemic crisis |
| Tablets VIII-IX | Gilgamesh's grief; journey to Utnapishtim | Continued fall / search | Rising (seeking answers) |
| Tablets X-XI | Flood story; plant of immortality lost | Rise then fall | Major peak (ten?) |
| Tablet XII | Coda—Enkidu's shade | Ambiguous / reconciliation | Reconciliation |

**Initial hypothesis**: Gilgamesh may show **Oedipus structure** (fall-rise-fall) under F_sentiment, but with a **kishōtenketsu-like epistemic structure**—the flood narrative (Tablet XI) functions as a "ten" that reframes everything.

**Deeper question**: Gilgamesh is about *failing to achieve immortality* but *achieving wisdom*. This is a contravariant structure like Ivan Ilyich: the physical/goal trajectory falls while the epistemic/spiritual trajectory rises. Ancient narratives may privilege this dual-functor structure over the single-functor shapes Reagan identified.

**Cross-cultural comparison matrix**:

| Epic | Origin | Era | Predicted Sentiment Shape | Predicted Epistemic Shape | H-K Position |
|------|--------|-----|---------------------------|---------------------------|--------------|
| Gilgamesh | Mesopotamia | ~2100 BCE | Oedipus | Rags-to-riches | +0.3 (K-leaning) |
| Iliad | Greece | ~750 BCE | Tragedy | Flat (no growth) | -0.5 (H-leaning) |
| Odyssey | Greece | ~750 BCE | Man-in-hole | Cinderella | -0.3 (H-leaning) |
| Mahabharata | India | ~400 BCE | Complex/composite | Multiple peaks | 0.0 (hybrid) |
| Beowulf | Anglo-Saxon | ~700 CE | Rise-fall-rise-fall | Stable→peak at death | -0.2 (H-leaning) |
| Journey to West | China | ~1590 CE | Episodic/flat | Episodic peaks | +0.4 (K-leaning) |

**If confirmed**: The presence of kishōtenketsu-like structures in pre-Homeric epic would suggest the Western conflict-driven model is a *regional specialization*, not a universal default.

#### Q11: Shannon Entropy and Narrative Structure

Can information theory provide a rigorous foundation for narrative analysis?

**Background**: Claude Shannon's 1948 "A Mathematical Theory of Communication" established that information is the reduction of uncertainty. Recent work has applied these concepts to narrative, formalizing intuitions about suspense, surprise, and pacing.

**Key Research Findings**:

1. **Narrative Information Theory (2024)**: Recent work introduces an information-theoretic framework for capturing narrative dynamics using:
   - State entropy (uncertainty about current emotional configuration)
   - Predictive entropy (uncertainty about what happens next)
   - Jensen-Shannon Divergence between states (magnitude of plot pivots)
   
2. **Suspense as Uncertainty Reduction (Wilmot & Keller, 2020)**: Suspense correlates with forward-looking uncertainty—how unpredictable the continuation is. This outperforms backward-looking surprise in predicting human suspense judgments.

3. **Uniform Information Density (UID) Hypothesis**: Effective communication maintains relatively stable information flow. Applied to narrative, this predicts:
   - Well-paced stories have relatively uniform surprisal
   - Cliffhangers create entropy spikes at boundaries
   - Resolutions collapse entropy toward zero
   
4. **Genre-Specific Entropy Patterns**: Reality TV shows higher entropy (broader emotional mix) and higher JSD (more frequent pivots) than dramas/thrillers, which maintain lower entropy focused on specific emotional tones.

**Method**: Define an entropy functor F_H: **Narr** → **ℝ-Trajectory**:

```
F_H(window) = H(P(next_state | window)) = -Σ p(s) log p(s)
```

Where P is estimated using a language model's predictive distribution.

**Derived Measures**:

| Metric | Definition | Narrative Meaning |
|--------|------------|-------------------|
| **Mean Entropy** | avg(F_H) over narrative | Overall unpredictability |
| **Entropy Variance** | var(F_H) over narrative | Pacing consistency |
| **Entropy Slope** | trend(F_H) over narrative | Building vs. resolving tension |
| **Peak Entropy Position** | argmax(F_H) | Location of maximum uncertainty |
| **Entropy at Boundaries** | F_H at act/chapter breaks | Cliffhanger intensity |

**Hypotheses**:

1. **Kishōtenketsu will show entropy spike at "ten" position (~75%)**
   - The twist is informationally surprising
   - Followed by entropy reduction in ketsu (reconciliation)

2. **Harmon Circle will show entropy valley at midpoint**
   - Steps 5-6 (Find/Take) are predictable given setup
   - Entropy rises again in return phase

3. **Reagan shapes map to entropy signatures**:
   | Shape | Predicted Entropy Pattern |
   |-------|---------------------------|
   | Rags-to-riches | Decreasing (world becomes more predictable) |
   | Tragedy | Increasing then collapse (doom becomes certain) |
   | Man-in-hole | Valley (descent predictable, ascent reduces uncertainty) |
   | Icarus | Peak then collapse (fall becomes inevitable) |
   | Cinderella | Valley-peak-valley (multiple uncertainty cycles) |
   | Oedipus | Peak-valley-peak (reveals create new uncertainties) |

4. **Cross-cultural entropy patterns will differ**:
   - Western conflict narratives: entropy driven by outcome uncertainty
   - Kishōtenketsu: entropy driven by interpretation uncertainty (what does this *mean*?)
   - The *type* of uncertainty differs even if magnitude is similar

5. **Streaming-era content will show higher entropy variance**
   - More frequent spikes (micro-cliffhangers)
   - Less sustained low-entropy periods (no "boring" stretches)
   - Optimized for preventing viewer departure

**Connection to Dopamine Hypothesis (Q9)**:

Information theory provides a mechanistic bridge to neurochemistry:
- Surprisal (unexpected events) triggers dopamine release
- Uncertainty (high entropy) creates anticipation
- Resolution (entropy collapse) provides satisfaction

The "dopamine optimization" hypothesis can be reformulated information-theoretically: streaming content maximizes **integrated surprisal** (total dopamine) while maintaining **entropy variance** (sustained engagement).

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

### 4.4 Shannon Entropy Functor Implementation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.special import rel_entr

class NarrativeEntropyFunctor:
    """
    Compute Shannon entropy measures over narrative trajectory.
    Based on Narrative Information Theory framework.
    """
    
    def __init__(self, model_name='gpt2-medium'):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model.eval()
    
    def compute_entropy(self, text: str) -> float:
        """Compute entropy of next-token distribution (uncertainty about next state)."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last position
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            h = scipy_entropy(probs, base=2)
        return h
    
    def compute_surprisal(self, context: str, continuation: str) -> float:
        """Compute surprisal of continuation given context."""
        full_text = context + continuation
        inputs = self.tokenizer(full_text, return_tensors='pt', truncation=True)
        context_len = len(self.tokenizer(context)['input_ids'])
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            # Get per-token losses for continuation only
            logits = outputs.logits[0, context_len-1:-1, :]
            targets = inputs['input_ids'][0, context_len:]
            
            log_probs = torch.log_softmax(logits, dim=-1)
            token_surprisals = -log_probs.gather(1, targets.unsqueeze(1)).squeeze()
            
        return token_surprisals.mean().item()
    
    def __call__(self, windows: list) -> np.ndarray:
        """Compute entropy trajectory over narrative windows."""
        context = ""
        entropies = []
        for w in windows:
            context += " " + w
            # Use sliding context window
            h = self.compute_entropy(context[-4000:])
            entropies.append(h)
        return np.array(entropies)
    
    def compute_jsd_trajectory(self, windows: list) -> np.ndarray:
        """
        Compute Jensen-Shannon Divergence between consecutive states.
        High JSD = major plot pivot / emotional shift.
        """
        def get_distribution(text):
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                return torch.softmax(logits, dim=-1).cpu().numpy()
        
        jsds = [0.0]  # First window has no predecessor
        prev_dist = get_distribution(windows[0])
        
        for w in windows[1:]:
            curr_dist = get_distribution(w)
            # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
            m = 0.5 * (prev_dist + curr_dist)
            jsd = 0.5 * np.sum(rel_entr(prev_dist, m)) + 0.5 * np.sum(rel_entr(curr_dist, m))
            jsds.append(jsd)
            prev_dist = curr_dist
            
        return np.array(jsds)


def narrative_entropy_metrics(trajectory: np.ndarray, jsd_trajectory: np.ndarray = None) -> dict:
    """Extract meaningful metrics from entropy trajectory."""
    n = len(trajectory)
    
    metrics = {
        # Basic statistics
        'mean_entropy': np.mean(trajectory),
        'entropy_variance': np.var(trajectory),
        'entropy_std': np.std(trajectory),
        'max_entropy': np.max(trajectory),
        'min_entropy': np.min(trajectory),
        'entropy_range': np.max(trajectory) - np.min(trajectory),
        
        # Structural positions
        'max_entropy_position': np.argmax(trajectory) / n,  # Normalized 0-1
        'min_entropy_position': np.argmin(trajectory) / n,
        
        # Trend analysis
        'entropy_slope': np.polyfit(range(n), trajectory, 1)[0],
        'final_entropy': trajectory[-1],
        'initial_entropy': trajectory[0],
        'entropy_change': trajectory[-1] - trajectory[0],
        
        # Pacing metrics (UID-related)
        'pacing_uniformity': 1.0 / (1.0 + np.var(trajectory)),  # Higher = more uniform
        
        # Boundary analysis
        'final_tenth_mean': np.mean(trajectory[-n//10:]) if n >= 10 else trajectory[-1],
        'cliffhanger_score': trajectory[-1] - np.mean(trajectory),
        
        # Quarter analysis (for kishōtenketsu detection)
        'q1_entropy': np.mean(trajectory[:n//4]),
        'q2_entropy': np.mean(trajectory[n//4:n//2]),
        'q3_entropy': np.mean(trajectory[n//2:3*n//4]),
        'q4_entropy': np.mean(trajectory[3*n//4:]),
    }
    
    # Kishōtenketsu signature: spike in Q3
    metrics['ten_spike_score'] = metrics['q3_entropy'] - (metrics['q1_entropy'] + metrics['q2_entropy']) / 2
    
    # Resolution score: how much entropy drops at end
    metrics['resolution_score'] = metrics['max_entropy'] - metrics['final_entropy']
    
    if jsd_trajectory is not None:
        metrics['mean_jsd'] = np.mean(jsd_trajectory)
        metrics['max_jsd'] = np.max(jsd_trajectory)
        metrics['max_jsd_position'] = np.argmax(jsd_trajectory) / len(jsd_trajectory)
        metrics['pivot_count'] = np.sum(jsd_trajectory > np.mean(jsd_trajectory) + np.std(jsd_trajectory))
    
    return metrics


def detect_entropy_structure(metrics: dict) -> str:
    """Classify narrative structure based on entropy signature."""
    
    # Kishōtenketsu: spike in third quarter, resolution in fourth
    if metrics['ten_spike_score'] > 0.5 and metrics['q4_entropy'] < metrics['q3_entropy']:
        return 'kishotenketsu'
    
    # Mystery: high throughout, sharp drop at end
    if metrics['mean_entropy'] > 8.0 and metrics['resolution_score'] > 2.0:
        return 'mystery'
    
    # Tragedy: rising entropy (doom becomes certain)
    if metrics['entropy_slope'] > 0.01 and metrics['final_entropy'] > metrics['initial_entropy']:
        return 'tragedy'
    
    # Man-in-hole: valley structure
    if metrics['min_entropy_position'] > 0.3 and metrics['min_entropy_position'] < 0.7:
        if metrics['final_entropy'] > metrics['min_entropy']:
            return 'man_in_hole'
    
    # Uniform/slice-of-life: low variance
    if metrics['pacing_uniformity'] > 0.8:
        return 'slice_of_life'
    
    return 'unclassified'
```

### 4.5 Cross-Cultural Transfer Test

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

6. **The Gutenberg Parenthesis will be partially confirmed**
   - Pre-print narratives (Gilgamesh, Homeric epic) will show episodic or dual-arc structures
   - Print-era narratives will cluster into the six Reagan shapes
   - Post-print/streaming content will show fragmented, high-frequency oscillation

7. **Dopamine optimization will be measurable**
   - Peak density (emotional spikes per unit time) will increase from print → streaming era
   - Modern serialized content will show "cliffhanger signatures" at structural boundaries
   - Algorithmic-era content (2015+) will show highest peak density

8. **Gilgamesh will show a "staggered dual-arc" structure**
   - Sentiment trajectory: Oedipus-like (fall-rise-fall)
   - Epistemic trajectory: Delayed rise (wisdom comes after suffering)
   - This phase-shifted structure may represent an *older* narrative paradigm

9. **The "six shapes" will prove to be historically contingent**
   - Universal across print-era Western corpora
   - Partially applicable to non-Western print literature
   - Poorly applicable to pre-print epic and post-print digital native fiction
   - The shapes are *print culture artifacts*, not human universals

### 5.2 Category-Theoretic Implications

If different traditions optimize different functors, this suggests:

**Narrative universals exist at the level of** ***Narr*** **itself, not at the level of any particular functor's image.**

The "six shapes" and kishōtenketsu are both *shadows* of a richer underlying structure. Different cultures have developed different *projection operators* that emphasize different aspects of narrative transformation.

This is analogous to how:
- A sphere projects to a circle from one angle, an ellipse from another
- Both are valid "shapes of a sphere" but neither is *the* shape
- The sphere itself is the universal object

**Conjecture**: There exists a "narrative sphere"—a universal structure of transformation—and cultural traditions are projection functors from this sphere to human experience.

### 5.3 Media-Archaeological Implications

If the Gutenberg Parenthesis hypothesis is confirmed, the implications are profound:

**The "six shapes" may be a ~500-year anomaly**, an optimization for a specific technological configuration (print + individual reading + closure expectation). Before and after this parenthesis, different shapes dominate.

This suggests a **media-archaeological periodization**:

| Era | Dominant Shapes | Optimization Target |
|-----|-----------------|---------------------|
| Pre-print (oral) | Episodic, cyclical, modular | Memorability, ritual repetition |
| Print (Gutenberg) | Six Reagan shapes | Individual closure, emotional catharsis |
| Digital/Streaming | Micro-arc cascades, open structures | Attention capture, dopamine optimization |

**The deeper question**: Is there a shape-invariant structure beneath these variations? Or are shapes *entirely* media-determined, with no universal substrate?

A category-theoretic answer: The **category Narr** may be universal (transformations exist everywhere), but the **functors** we use to observe it are technologically constructed. What changes across eras is not narrative itself but our *instrumentation* for detecting it.

This parallels physics: the universe doesn't change when we build better telescopes, but what we can *see* changes dramatically. Similarly, oral epic, print novel, and streaming series may all contain the same underlying transformations—but the affordances of each medium select for different observable projections.

### 5.4 Practical Applications

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

Pettitt, T. (2007). Before the Gutenberg Parenthesis: Elizabethan-American Compatibilities. *Media in Transition 5*.

Sauerberg, L. O. (2009). The Encyclopedia and the Gutenberg Parenthesis. *MIT Communications Forum*.

George, A. R. (2003). *The Babylonian Gilgamesh Epic: Introduction, Critical Edition and Cuneiform Texts*. Oxford University Press.

Lord, A. B. (1960). *The Singer of Tales*. Harvard University Press.

Ong, W. J. (1982). *Orality and Literacy: The Technologizing of the Word*. Methuen.

Alter, A. (2017). *Irresistible: The Rise of Addictive Technology and the Business of Keeping Us Hooked*. Penguin.

Schultz, W. (2015). Neuronal Reward and Decision Signals: From Theories to Data. *Physiological Reviews*, 95(3), 853-951.

Zak, P. J. (2015). Why Inspiring Stories Make Us React: The Neuroscience of Narrative. *Cerebrum*, 2015, 2.

Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379-423.

Wilmot, D., & Keller, F. (2020). Modelling Suspense in Short Stories as Uncertainty Reduction over Neural Representation. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 1763-1788.

Ely, J., Frankel, A., & Kamenica, E. (2015). Suspense and Surprise. *Journal of Political Economy*, 123(1), 215-260.

Genzel, D., & Charniak, E. (2002). Entropy Rate Constancy in Text. *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics*, 199-206.

Hale, J. (2001). A Probabilistic Earley Parser as a Psycholinguistic Model. *Proceedings of the Second Meeting of the North American Chapter of the Association for Computational Linguistics*, 1-8.

Levy, R. (2008). Expectation-Based Syntactic Comprehension. *Cognition*, 106(3), 1126-1177.

Futrell, R., Levy, R., & Gibson, E. (2020). Dependency Locality as an Explanatory Principle for Word Order. *Language*, 96(2), 371-412.

Meister, C., Pimentel, T., Haller, P., Jäger, L., Cotterell, R., & Levy, R. (2021). Revisiting the Uniform Information Density Hypothesis. *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, 963-980.

Crocker, M. W., & Demberg, V. (2015). Information Density and Linguistic Encoding (IDeaL). *KI - Künstliche Intelligenz*, 29, 275-281.

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

## Appendix D: The Gutenberg Parenthesis and Narrative Archaeology

### D.1 The Parenthesis Thesis

Thomas Pettitt and Lars Ole Sauerberg argue that print culture (c. 1450-2000) represents a historical anomaly—a "parenthesis" between two eras of more fluid, participatory communication:

```
ORAL CULTURE ──── [ PRINT PARENTHESIS ] ──── DIGITAL CULTURE
(participatory)    (fixed, authoritative)     (participatory?)
 ~50,000 years        ~550 years               ~50 years
```

Print imposed specific constraints on narrative:

| Constraint | Effect on Story Structure |
|------------|---------------------------|
| **Fixity** | Stories must have one "correct" version |
| **Linearity** | Reader proceeds start→finish, no branching |
| **Closure** | Books end; stories must resolve |
| **Individuality** | Silent reading, private interpretation |
| **Authority** | Author as singular origin |
| **Commodity** | Stories as products, optimized for sale |

**Hypothesis**: The "six shapes" may be *print-optimized*—structures that work well for solitary readers who expect closure and cannot interact with the text.

### D.2 Pre-Print Narrative Structures

Oral and manuscript traditions show different structural patterns:

**Episodic/Cyclic Structure**
- 1001 Nights: Frame narrative with embedded, interruptible stories
- Mahabharata: Massive embedding, digressions, multiple narrative levels
- Icelandic sagas: Episodic, genealogical, no single "arc"

**Formulaic/Modular Structure**
- Homeric epics: Oral-formulaic composition using pre-fab units
- Epic of Gilgamesh: Tablet-based structure, possibly assembled from independent poems
- Beowulf: Three major episodes, weakly integrated

**Participatory/Variable Structure**
- Folk tales: Different versions in different tellings
- Ballads: Singer adapts to audience
- Religious narratives: Ritual context shapes meaning

These structures prioritize **memorability**, **adaptability**, and **communal experience** over the **closure** and **individual arc** that print culture emphasizes.

### D.3 The Gilgamesh Deep Dive

The Epic of Gilgamesh is our oldest substantial narrative—and a crucial test case.

**Textual History**:
- Old Babylonian version (~1800 BCE): Shorter, possibly episodic
- Standard Babylonian version (~1200 BCE): Integrated 11-tablet epic
- Tablet XII: Later addition, doesn't fit narratively

This textual history suggests Gilgamesh was **assembled from modules**, not composed as a unified arc. The "shape" we see may be an artifact of editorial integration, not original composition.

**Structural Analysis**:

```
F_sentiment trajectory (predicted):
     │      ╱╲ Humbaba    Enkidu's
     │     ╱  ╲ victory   death
     │    ╱    ╲         ╱╲
     │   ╱      ╲       ╱  ╲     Plant
     │  ╱        ╲     ╱    ╲   lost
     │ ╱          ╲   ╱      ╲  ╱╲
     │╱            ╲_╱        ╲╱  ╲___
     └────────────────────────────────
       I  II III IV  V  VI VII VIII IX  X  XI XII
       
F_epistemic trajectory (predicted):
     │                            Flood
     │                           narrative
     │                             ╱╲
     │                            ╱  ╲
     │                    Ut-    ╱    ╲
     │               napishtim  ╱      ╲___
     │    Enkidu's    ╱╲      ╱
     │    wisdom     ╱  ╲    ╱
     │      ╱╲      ╱    ╲  ╱
     │     ╱  ╲____╱      ╲╱
     │____╱
     └────────────────────────────────
       I  II III IV  V  VI VII VIII IX  X  XI XII
```

**Key observation**: The two trajectories are **phase-shifted**. The epistemic peak (Flood narrative, Tablet XI) comes *after* the emotional nadir (Enkidu's death, Tablet VII). This is neither man-in-a-hole nor kishōtenketsu—it's a **staggered dual-arc structure**.

**Possible interpretation**: Ancient narrative may have understood that wisdom comes *through* suffering, not as resolution *of* suffering. The structure embeds a temporal theory: emotional loss precedes (and enables) epistemic gain.

### D.4 Dopamine Archaeology: From Ritual to Algorithm

A speculative history of narrative-neurochemistry coupling:

| Era | Context | Dopamine Pattern | Structural Consequence |
|-----|---------|------------------|------------------------|
| **Oral/Ritual** | Communal, performative | Social bonding + anticipation | Episodic, call-and-response, familiar stories |
| **Manuscript** | Elite, devotional | Slow, contemplative | Allegorical, digressive, re-readable |
| **Early Print** | Individual, leisured | Sustained attention | Rising action → climax → resolution |
| **Serial Print** | Mass market, periodical | Cliffhanger anticipation | Episode structure, multiple arcs |
| **Cinema** | Collective dark room | Immersive, 2-hour sustained | Three-act, spectacle peaks |
| **Television** | Domestic, interruptible | Commercial-break rhythm | Act breaks, mini-cliffs |
| **Streaming** | Bingeable, algorithmic | Continuous micro-dosing | Constant stimulation, no closure |
| **Short-form** | Scrolling, competitive | Instant hook, rapid payoff | No build-up, pure spike |

**The compression hypothesis**: Each technological shift *compresses* the reward cycle:

```
Time to first dopamine peak:

Gilgamesh:     ~2 hours (performed over days)
Novel:         ~30-60 minutes
Film:          ~15-20 minutes  
TV episode:    ~8-12 minutes
Streaming:     ~3-5 minutes
TikTok:        ~3-5 seconds
```

If this pattern holds, we should see *different* dominant shapes at each technological stratum—not because human psychology changes, but because the *optimization target* changes.

### D.5 Implications for the Research Design

These questions require expanding our corpus and methods:

**Corpus additions**:
- Ancient epics in translation (Gilgamesh, Iliad, Odyssey, Mahabharata, Beowulf)
- Oral transcriptions (folk tale collections, ethnographic recordings)
- Web serials (Worm, Practical Guide to Evil, web novel platforms)
- Short-form narrative (TikTok story accounts, Twitter fiction threads)

**Method additions**:
- **Peak density metric**: emotional peaks per unit narrative time
- **Closure index**: how definitively does the ending resolve tensions?
- **Episodic coherence**: how self-contained are sub-units?
- **Anticipation markers**: linguistic signals of "what happens next" pull

**New analysis questions**:

1. **Peak density by era**: Does modern content show higher peak density?
2. **Closure by medium**: Do print novels show more closure than web serials?
3. **Shape stability across millennia**: Do the six shapes appear in Gilgamesh?
4. **Oral vs. print structure**: Do folk tale collections show different shapes than novels?

**Expected findings**:

If the Gutenberg Parenthesis hypothesis is correct:
- Pre-print narratives will show **episodic, cyclical, or modular** structures
- Print narratives will show **the six Reagan shapes** (closure-oriented)
- Post-print narratives will show **fragmented, open, or cascading** structures

The "six shapes" may be a ~500-year anomaly, not a human universal.

---

## Appendix F: Information Theory and Narrative—A Theoretical Synthesis

### F.1 Shannon's Framework Applied to Story

Claude Shannon's 1948 paper established that information is the reduction of uncertainty. A message is informative to the degree that it narrows down possibilities. Applied to narrative:

**The reader is a decoder.** They receive a sequence of narrative states (sentences, scenes, chapters) and must reconstruct the "message"—the story's meaning, trajectory, and resolution.

**The author is an encoder.** They must compress a complex world, characters, and events into a linear sequence of words, optimizing for both comprehensibility and engagement.

**The channel has capacity limits.** Human cognitive processing has bandwidth constraints—too much information per unit time causes overload; too little causes boredom.

### F.2 Core Information-Theoretic Measures

**Entropy**: H(X) = -Σ p(x) log₂ p(x)

For narrative: entropy over possible next states. High entropy = high suspense (many possible continuations). Low entropy = predictability or resolution.

**Surprisal**: I(x) = -log₂ p(x)

For narrative: how unexpected an event is. A character's sudden death in chapter 3 has high surprisal. A hero defeating the villain in the climax has lower surprisal.

**Mutual Information**: I(X; Y) = H(X) - H(X|Y)

For narrative: how much knowing one part helps predict another. High mutual information between setup and payoff indicates tight plotting. Low mutual information indicates loose, episodic structure.

**Kullback-Leibler Divergence**: KL(P||Q) = Σ p(x) log(p(x)/q(x))

For narrative: the "distance" between what the reader expected and what happened. A plot twist maximizes KL divergence between predicted and actual continuation.

### F.3 The Uniform Information Density Hypothesis

The UID hypothesis proposes that effective communication distributes information uniformly across the signal. Evidence includes:

1. Speakers slow down for unpredictable words
2. Syntactic reduction (contractions, ellipsis) occurs in predictable contexts
3. High-frequency words are shorter (Zipf's law optimization)

**Applied to narrative**, UID predicts:

- Good pacing = relatively uniform surprisal across the text
- "Purple prose" violations = surprisal too high (overloaded)
- "Boring" passages = surprisal too low (uninformative)
- Skilled authors modulate surprisal around a comfortable mean

**However**, recent work suggests UID is incomplete:

> "Uniform Information Density Isn't the Whole Story"

Narrative *intentionally* violates UID for effect:
- Cliffhangers create entropy spikes
- Twists create surprisal spikes
- Resolutions create entropy collapse

The question becomes: **What is the optimal surprisal *contour* for narrative?**

### F.4 Suspense vs. Surprise: A Crucial Distinction

Wilmot & Keller (2020) formalized two competing models:

**Surprise** (backward-looking): How unexpected is the current state given history?
```
Surprise(t) = KL(P(s_t | s_{<t}) || P(s_t))
```

**Suspense** (forward-looking): How uncertain are we about what comes next?
```
Suspense(t) = H(P(s_{t+1} | s_{≤t}))
```

Empirically, **suspense (uncertainty reduction) better predicts human suspense judgments** than surprise.

This has structural implications:

| Structure | Surprise Pattern | Suspense Pattern |
|-----------|------------------|------------------|
| Mystery | Low until reveal | High throughout |
| Thriller | Periodic spikes | Sustained high |
| Tragedy | Rising (doom inevitable) | Falling (outcome certain) |
| Comedy | Periodic spikes | Oscillating |

### F.5 Entropy Signatures of Narrative Structures

**Harmon Circle**:
```
Entropy
   │    ╱╲                     ╱╲
   │   ╱  ╲                   ╱  ╲
   │  ╱    ╲                 ╱    ╲
   │ ╱      ╲_______________╱      ╲
   │╱        (chaos=certainty)      ╲___
   └────────────────────────────────────
     1  2  3  4  5  6  7  8
     Order → Chaos → Order
```

The "bottom" of the Circle (steps 4-6) has *lower* entropy because the character is committed to a path. Uncertainty is highest at thresholds (steps 2-3, 6-7).

**Kishōtenketsu**:
```
Entropy
   │              ╱╲
   │             ╱  ╲
   │            ╱    ╲
   │___________╱      ╲_____
   │                        
   └────────────────────────
     Ki    Shō    Ten   Ketsu
```

Entropy spike at "ten" (twist) because the reframing opens new interpretive possibilities. Ketsu resolves by showing how to integrate old and new understanding.

**Reagan's Six Shapes** (entropy interpretation):

| Shape | Entropy Trajectory |
|-------|-------------------|
| Rags-to-riches | Decreasing (world becomes ordered) |
| Tragedy | Increasing then plateau (chaos wins) |
| Man-in-hole | V-shaped (certainty → uncertainty → resolution) |
| Icarus | Inverted-V (uncertainty → certainty of fall) |
| Cinderella | W-shaped (multiple uncertainty cycles) |
| Oedipus | M-shaped (revelations create new uncertainties) |

### F.6 Genre as Entropy Contract

Different genres establish different **entropy contracts** with readers:

| Genre | Expected Entropy Range | Tolerance for Spikes | Resolution Expected? |
|-------|------------------------|----------------------|---------------------|
| Literary fiction | Medium-high | High | No |
| Mystery | High until end | One major drop | Yes |
| Romance | Medium, oscillating | Medium | Yes (HEA) |
| Horror | High, rising | High | Variable |
| Slice-of-life | Low, uniform | Low | No |
| Thriller | High throughout | Very high | Yes |

**Genre violations** = breaking the entropy contract. A mystery that resolves at 50% feels wrong. A slice-of-life with a major twist may feel like a betrayal.

### F.7 Cross-Cultural Entropy Optimization

Different cultures may optimize *different* information-theoretic objectives:

**Western (conflict-driven)**:
- Optimize: Total surprisal (maximize emotional impact)
- Constraint: Maintain comprehensibility
- Result: High-entropy middles, low-entropy endings

**Kishōtenketsu (reframing-driven)**:
- Optimize: KL divergence at twist (maximize perspective shift)
- Constraint: Maintain coherence across reframe
- Result: Low entropy until twist, spike, then resolution

**Oral tradition (memory-driven)**:
- Optimize: Minimize entropy (maximize memorability)
- Constraint: Maintain interest
- Result: Formulaic, repetitive, low entropy throughout

**Streaming-era (engagement-driven)**:
- Optimize: Entropy variance (prevent departure)
- Constraint: None (no closure required)
- Result: Constant micro-spikes, never resolving

### F.8 The Category-Theoretic Connection

Information theory and category theory connect via:

**Entropy as a functor**: H: Prob → ℝ⁺

Where **Prob** is the category of probability distributions with stochastic maps as morphisms.

**Narrative states as distributions**: Each narrative window induces a distribution over possible continuations.

**Story as morphism in Prob**: A story is a sequence of morphisms (updates) between distributions.

The **information-theoretic observation functor**:
```
F_H: Narr → Prob → ℝ⁺
```

Composes the "narrative to distribution" functor with the entropy functor.

Different narrative traditions correspond to different **optimization functionals** over trajectories in ℝ⁺:

| Tradition | Functional |
|-----------|------------|
| Western | max ∫ surprisal dt |
| Kishōtenketsu | max KL(P_{pre-twist} || P_{post-twist}) |
| Slice-of-life | min Var(entropy) |
| Streaming | max Var(entropy) subject to E[entropy] > threshold |

### F.9 Empirical Predictions

1. **Reading time should correlate with surprisal** (established in psycholinguistics)

2. **Engagement metrics should correlate with entropy variance** (testable with streaming data)

3. **Cross-cultural differences should appear in entropy signatures** (testable with multilingual corpora)

4. **Historical shift toward higher entropy variance** (testable across eras)

5. **Kishōtenketsu should show distinctive KL-divergence spike** (testable with Japanese/Korean corpora)

---

## Appendix G: Implementation Roadmap

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
