# Category Narr: A Practical Implementation

## Overview

Category **Narr** is the source category in our functorial narrative analysis framework. It was previously theoretical - we observed it only through functor images. Now it's **practical and implemented**.

```
Narr (concrete!)     F      Trajectory
   ↓                →          ↓
Objects: States              ℝⁿ
Morphisms: Beats             Differences
```

## The Category Structure

### Objects: NarrativeState

An object in Narr is a **narrative state** - a specific moment in the story characterized by:

```python
@dataclass
class NarrativeState:
    content: str              # The text at this moment
    position: float           # Where in narrative [0, 1]
    state_id: str             # Unique identifier
    features: Dict[str, float]  # Extracted properties
    knowledge_state: Dict[str, bool]  # What's known
    characters_present: List[str]
```

**Key insight**: Traditional narrative "moments" (exposition, rising action, climax) become concrete objects with measurable features.

### Morphisms: NarrativeMorphism

A morphism in Narr is a **narrative transition** - a beat that takes the story from one state to another:

```python
@dataclass
class NarrativeMorphism:
    source: NarrativeState    # Where we start
    target: NarrativeState    # Where we end
    morphism_type: MorphismType  # What kind of beat
    feature_deltas: Dict[str, float]  # How features change
    intensity: float          # How dramatic
```

**Morphism types** (narrative beats):
- `IDENTITY` - No change (id morphism)
- `REVELATION` - Information revealed
- `REVERSAL` - Fortune inverted
- `COMPLICATION` - New obstacle
- `RESOLUTION` - Conflict resolved
- `ESCALATION` - Stakes raised
- `REFLECTION` - Character contemplation
- `CLIMAX` - Peak intensity
- `DENOUEMENT` - Falling action

### Composition

Morphisms compose: if `f: A → B` and `g: B → C`, then `f @ g: A → C`

```python
# Python supports @ operator for composition
path = beginning_to_middle @ middle_to_end
```

This is the core operation making Narr a **category**.

## Category Laws (Verified!)

The implementation verifies all category laws:

```
✓ identity_left:  id_A ∘ f = f
✓ identity_right: f ∘ id_B = f
✓ associativity:  (f ∘ g) ∘ h = f ∘ (g ∘ h)
```

Run `category.verify_category_laws()` to check.

## Practical Usage

### Creating a Category from Text

```python
from src.categories import CategoryNarr

category = CategoryNarr.from_text(
    text="Your narrative text here...",
    narrative_id="my_story",
    title="My Story",
    n_states=20,  # Number of states to create
    feature_extractors={
        "tension": my_tension_function,
        "sentiment": my_sentiment_function,
    }
)

# Verify it's a valid category
laws = category.verify_category_laws()
print(laws)  # {'identity_left': True, 'identity_right': True, 'associativity': True}
```

### Analyzing the Narrative Arc

```python
# Get the sequence of morphisms
arc = category.narrative_arc()

for morphism in arc:
    print(f"{morphism.morphism_type.value}: intensity={morphism.intensity:.2f}")

# Output:
# exposition: intensity=0.10
# complication: intensity=0.45
# escalation: intensity=0.72
# climax: intensity=0.95
# resolution: intensity=0.30
# denouement: intensity=0.15
```

### Composing Morphisms

```python
# Get the full journey from start to end
first_state = category.objects[0]
last_state = category.objects[-1]

full_journey = category.path(first_state, last_state)
print(f"Total change: {full_journey.feature_deltas}")
```

## Natural Transformations

A natural transformation `η: F ⇒ G` relates two observation functors.

```python
from src.categories import NaturalTransformationBuilder, FunctorComparator

# Compare sentiment and entropy functors
comparator = FunctorComparator(category)
comparator.add_functor(SentimentFunctor())
comparator.add_functor(EntropyFunctor())

# Build transformation
transform = comparator.compare("sentiment", "entropy")
print(f"Is natural: {transform.is_natural}")
print(f"Mean error: {transform.mean_error:.4f}")

# Get correlation matrix
names, corr = comparator.correlation_matrix()
```

**Naturality** means: how you transform at one state consistently relates to transformations at other states. A natural transformation is "coherent" across the whole narrative.

## Integration with ICC

```python
from src.categories.integration import NarrativeFunctorialAnalyzer

analyzer = NarrativeFunctorialAnalyzer()
analyzer.add_functor(SentimentFunctor())
analyzer.add_functor(EntropyFunctor())

# Full analysis
analysis = analyzer.analyze_with_icc(
    text="...",
    narrative_id="story_1",
    title="My Story"
)

print(f"ICC Class: {analysis.icc_result.icc_class}")
print(f"Cultural prediction: {analysis.icc_result.cultural_prediction}")
```

## What This Enables

### 1. **Rigorous Morphism Analysis**
Instead of vague "narrative beats", we have typed morphisms with measurable properties.

### 2. **Composition = Narrative Summarization**
Composing morphisms gives a "summary" of a narrative segment.

### 3. **Natural Transformations = Functor Comparison**
How does sentiment relate to entropy? The transformation tells us.

### 4. **Category Laws = Consistency Check**
If laws fail, something is wrong with our construction.

### 5. **Cultural Analysis at Category Level**
Different cultures might have different morphism patterns:
- Western: many REVERSAL and ESCALATION morphisms
- Japanese: more REFLECTION and TRANSITION morphisms

## Theoretical Implications

1. **Category Narr is no longer abstract** - it's computable
2. **Morphism composition has meaning** - narrative segments can be "added"
3. **Natural transformations are testable** - functor relationships are measurable
4. **The framework is falsifiable** - category laws can be checked

## Files

- `src/categories/narr.py` - Core Category Narr implementation
- `src/categories/natural_transformations.py` - Natural transformations
- `src/categories/integration.py` - Integration with existing functors

## Example Output

```
======================================================================
FUNCTORIAL NARRATIVE ANALYSIS - FULL INTEGRATION DEMO
======================================================================

1. CATEGORY NARR CONSTRUCTION
   Title: The Prince's Trial
   Objects (states): 8
   Morphisms: 7

   Category laws:
     identity_left: ✓
     identity_right: ✓
     associativity: ✓

2. MORPHISM PATTERN ANALYSIS
   Arc length: 7
   Peak intensity: 0.96 at position 0.43
   Morphism types:
     transition: 5
     climax: 1
     denouement: 1

3. NARRATIVE ARC (sequence of morphisms)
   1. [transition] intensity=0.43
   2. [transition] intensity=0.00
   3. [transition] intensity=0.00
   4. [transition] intensity=0.96
   5. [climax] intensity=0.53
   6. [transition] intensity=0.27
   7. [denouement] intensity=0.47

4. MORPHISM COMPOSITION
   f: transition
   g: transition
   h: transition
   f @ g: spans state_000 → state_002
   f @ g @ h: spans state_000 → state_003

5. FULL NARRATIVE PATH (composed morphism)
   Start → End composition exists
   Peak intensity along path: 0.96

======================================================================
Category Narr is now PRACTICAL - not just theoretical!
======================================================================
```

## Future Directions

1. **Richer morphism classification** - Use ML to classify narrative beats
2. **Character-specific subcategories** - Track individual character arcs
3. **Inter-narrative functors** - Map between different stories
4. **Limit/colimit constructions** - Narrative "quotients" and "products"
5. **2-categorical structure** - Morphisms between morphisms (revisions)
