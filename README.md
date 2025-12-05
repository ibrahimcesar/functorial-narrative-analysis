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

---

## 2. A Category-Theoretic Framework

### 2.1 The Category **Narr**

Define a category **Narr** where:

- **Objects** are *narrative states*: configurations of (characters, world-state, reader knowledge, tension level, thematic position)
  
- **Morphisms** are *narrative beats* or *scenes*: functions that transform one state into another
  
- **Composition** is temporal sequencing: if `f: A → B` and `g: B → C`, then `g ∘ f: A → C` is the combined narrative movement

- **Identity** `id_A: A → A` represents "nothing happens"—a static scene

A complete story is a morphism `S: Initial → Final` in **Narr**.

### 2.2 Observation Functors

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

### 2.3 The Multi-Functor Hypothesis

**Claim**: Different narrative traditions optimize different functors.

| Tradition | Primary Functor | Secondary | Structure |
|-----------|-----------------|-----------|-----------|
| Western (conflict-driven) | F_sentiment | F_arousal | Three-act |
| Kishōtenketsu | F_epistemic | F_thematic | Four-act |
| Telenovela | F_social | F_sentiment | Serial |
| Slice-of-life | F_thematic | — | Episodic |

The "six shapes" are the archetypal forms in Im(F_sentiment). But Im(F_epistemic) may have *different* archetypal forms—perhaps corresponding to kishōtenketsu's structure.

### 2.4 Natural Transformations Between Functors

Given two functors F, G: **Narr** → **Trajectory**, a natural transformation η: F ⇒ G consists of morphisms η_S: F(S) → G(S) for each story S, such that the appropriate diagrams commute.

**Example**: Tolstoy's *The Death of Ivan Ilyich*

- F_sentiment(Ivan Ilyich) = Tragedy (steady fall—physical deterioration)
- F_epistemic(Ivan Ilyich) = Rags-to-Riches (steady rise—spiritual awakening)

These are *anti-correlated*. The natural transformation between them captures the novella's central irony: dying is the path to truly living.

### 2.5 Stories as Cones and Limits

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

### 4.2 Shape Clustering

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

### 4.3 Cross-Cultural Transfer Test

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

Jockers, M. L. (2015). Syuzhet: Extract Sentiment and Plot Arcs from Text. R package.

Kim, Y. M. (2017). Worldwide Story Structures. Blog post.

Hayashida, K. (2011). Mario Level Design via Kishōtenketsu. GDC Presentation.

Mac Lane, S. (1978). *Categories for the Working Mathematician*. Springer.

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

---

## Appendix B: Implementation Roadmap

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

*Author: [Ibrahim Cesar] | Date: [December 2025]*
