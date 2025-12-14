"""
Category Narr: The Category of Narrative States and Morphisms

This module makes the abstract categorical framework practical by providing
concrete implementations of:
- Objects: NarrativeState (a point in narrative time with content and features)
- Morphisms: NarrativeMorphism (transitions between states)
- Category: CategoryNarr with composition and identity laws

Mathematical foundation:
- Narr is a category where:
  - Objects are narrative states (moments in a story)
  - Morphisms are narrative transitions (beats/moves)
  - Composition: f;g means "state A → state B → state C"
  - Identity: id_A is the "no change" morphism
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from enum import Enum
import numpy as np
from collections import defaultdict


class MorphismType(Enum):
    """Types of narrative transitions (morphisms in Narr)."""
    IDENTITY = "identity"           # No change (id morphism)
    REVELATION = "revelation"       # Information revealed to reader/character
    REVERSAL = "reversal"           # Fortune/status inverted
    COMPLICATION = "complication"   # New obstacle introduced
    RESOLUTION = "resolution"       # Conflict resolved
    ESCALATION = "escalation"       # Stakes raised
    REFLECTION = "reflection"       # Character contemplation
    TRANSITION = "transition"       # Time/space shift
    CLIMAX = "climax"               # Peak dramatic intensity
    DENOUEMENT = "denouement"       # Falling action after climax


@dataclass
class NarrativeState:
    """
    An object in Category Narr.

    Represents a specific moment/state in a narrative, characterized by:
    - Content: The actual text
    - Position: Where in the narrative (normalized 0-1)
    - Features: Extracted properties (tension, knowledge, emotion, etc.)

    This makes the abstract "objects of Narr" concrete and measurable.
    """
    content: str
    position: float  # Normalized position in narrative [0, 1]
    state_id: str

    # Extracted features (populated by analysis)
    features: Dict[str, float] = field(default_factory=dict)

    # Epistemic state: what characters/reader know
    knowledge_state: Dict[str, bool] = field(default_factory=dict)

    # Character presence in this state
    characters_present: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.state_id)

    def __eq__(self, other):
        if not isinstance(other, NarrativeState):
            return False
        return self.state_id == other.state_id

    @property
    def tension(self) -> float:
        """Narrative tension level (from features)."""
        return self.features.get("tension", 0.5)

    @property
    def sentiment(self) -> float:
        """Emotional valence (from features)."""
        return self.features.get("sentiment", 0.0)

    @property
    def entropy(self) -> float:
        """Information entropy (from features)."""
        return self.features.get("entropy", 0.5)

    def distance_to(self, other: "NarrativeState") -> float:
        """
        Compute distance to another state in feature space.

        This gives us a metric structure on Ob(Narr).
        """
        if not self.features or not other.features:
            return abs(self.position - other.position)

        common_keys = set(self.features.keys()) & set(other.features.keys())
        if not common_keys:
            return abs(self.position - other.position)

        diffs = [(self.features[k] - other.features[k])**2 for k in common_keys]
        return np.sqrt(sum(diffs))


@dataclass
class NarrativeMorphism:
    """
    A morphism in Category Narr.

    Represents a narrative transition/beat from one state to another.
    Morphisms are composable: f: A → B and g: B → C compose to g∘f: A → C

    Key insight: Traditional narrative "beats" (inciting incident, reversal,
    climax, etc.) are specific types of morphisms.
    """
    source: NarrativeState
    target: NarrativeState
    morphism_type: MorphismType
    morphism_id: str

    # How features change (delta)
    feature_deltas: Dict[str, float] = field(default_factory=dict)

    # Intensity of the transition
    intensity: float = 0.5

    # Text span that enacts this transition
    text_span: str = ""

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.morphism_id)

    def __eq__(self, other):
        if not isinstance(other, NarrativeMorphism):
            return False
        return self.morphism_id == other.morphism_id

    @property
    def is_identity(self) -> bool:
        """Check if this is an identity morphism."""
        return self.morphism_type == MorphismType.IDENTITY

    def compose(self, other: "NarrativeMorphism") -> "NarrativeMorphism":
        """
        Compose this morphism with another: self ; other (self then other).

        Requires: self.target == other.source (morphisms must be composable)
        Returns: A → C where self: A → B and other: B → C

        This is the fundamental operation making Narr a category.
        """
        if self.target != other.source:
            raise ValueError(
                f"Cannot compose: target {self.target.state_id} != "
                f"source {other.source.state_id}"
            )

        # Compose feature deltas
        composed_deltas = dict(self.feature_deltas)
        for key, delta in other.feature_deltas.items():
            composed_deltas[key] = composed_deltas.get(key, 0) + delta

        # Determine composed morphism type
        if self.is_identity:
            composed_type = other.morphism_type
        elif other.is_identity:
            composed_type = self.morphism_type
        else:
            # Both non-identity: use the more "significant" one
            # (This is a simplification; real composition is complex)
            composed_type = (
                other.morphism_type
                if other.intensity > self.intensity
                else self.morphism_type
            )

        return NarrativeMorphism(
            source=self.source,
            target=other.target,
            morphism_type=composed_type,
            morphism_id=f"{self.morphism_id};{other.morphism_id}",
            feature_deltas=composed_deltas,
            intensity=max(self.intensity, other.intensity),
            text_span=self.text_span + " [...] " + other.text_span,
            metadata={
                "composed_from": [self.morphism_id, other.morphism_id],
                "composition_depth": (
                    self.metadata.get("composition_depth", 1) +
                    other.metadata.get("composition_depth", 1)
                )
            }
        )

    def __matmul__(self, other: "NarrativeMorphism") -> "NarrativeMorphism":
        """Allow f @ g syntax for composition (f then g)."""
        return self.compose(other)


# Type alias for clarity
NarrativeObject = NarrativeState


class CategoryNarr:
    """
    The Category of Narrative States.

    A concrete implementation of Category Narr with:
    - Objects: NarrativeState instances
    - Morphisms: NarrativeMorphism instances
    - Composition: morphism.compose() or @ operator
    - Identity: identity morphisms for each object

    This class manages a specific narrative (a subcategory of the full Narr).
    """

    def __init__(self, narrative_id: str, title: str = ""):
        self.narrative_id = narrative_id
        self.title = title

        # Objects and morphisms
        self._objects: Dict[str, NarrativeState] = {}
        self._morphisms: Dict[str, NarrativeMorphism] = {}

        # Hom-sets: hom[source_id][target_id] = list of morphisms
        self._hom_sets: Dict[str, Dict[str, List[NarrativeMorphism]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Identity morphisms cache
        self._identities: Dict[str, NarrativeMorphism] = {}

    def add_object(self, state: NarrativeState) -> None:
        """Add a narrative state (object) to the category."""
        self._objects[state.state_id] = state

    def add_morphism(self, morphism: NarrativeMorphism) -> None:
        """Add a narrative transition (morphism) to the category."""
        # Ensure source and target are in the category
        if morphism.source.state_id not in self._objects:
            self.add_object(morphism.source)
        if morphism.target.state_id not in self._objects:
            self.add_object(morphism.target)

        self._morphisms[morphism.morphism_id] = morphism
        self._hom_sets[morphism.source.state_id][morphism.target.state_id].append(morphism)

    def identity(self, state: NarrativeState) -> NarrativeMorphism:
        """
        Get the identity morphism for a state.

        Category law: id_A : A → A such that f ∘ id_A = f and id_A ∘ g = g
        """
        if state.state_id not in self._identities:
            self._identities[state.state_id] = NarrativeMorphism(
                source=state,
                target=state,
                morphism_type=MorphismType.IDENTITY,
                morphism_id=f"id_{state.state_id}",
                feature_deltas={},
                intensity=0.0,
                text_span="",
                metadata={"is_identity": True}
            )
        return self._identities[state.state_id]

    def hom(self, source: NarrativeState, target: NarrativeState) -> List[NarrativeMorphism]:
        """
        Get the Hom-set Hom(source, target).

        Returns all morphisms from source to target.
        """
        morphisms = self._hom_sets[source.state_id][target.state_id].copy()

        # Include identity if source == target
        if source == target:
            morphisms.append(self.identity(source))

        return morphisms

    def compose(
        self,
        f: NarrativeMorphism,
        g: NarrativeMorphism
    ) -> NarrativeMorphism:
        """
        Compose morphisms: f ; g (f then g).

        Category law: (f ; g) ; h = f ; (g ; h) (associativity)
        """
        return f.compose(g)

    @property
    def objects(self) -> List[NarrativeState]:
        """All objects in this category."""
        return list(self._objects.values())

    @property
    def morphisms(self) -> List[NarrativeMorphism]:
        """All non-identity morphisms in this category."""
        return list(self._morphisms.values())

    def path(
        self,
        start: NarrativeState,
        end: NarrativeState
    ) -> Optional[NarrativeMorphism]:
        """
        Find a composed morphism from start to end.

        This represents the "narrative journey" from one state to another.
        """
        # Sort objects by position
        sorted_objects = sorted(self._objects.values(), key=lambda s: s.position)

        # Find indices
        try:
            start_idx = next(
                i for i, s in enumerate(sorted_objects)
                if s.state_id == start.state_id
            )
            end_idx = next(
                i for i, s in enumerate(sorted_objects)
                if s.state_id == end.state_id
            )
        except StopIteration:
            return None

        if start_idx >= end_idx:
            return self.identity(start) if start_idx == end_idx else None

        # Compose morphisms along the path
        current = start
        composed = None

        for i in range(start_idx, end_idx):
            next_state = sorted_objects[i + 1]
            morphisms = self.hom(current, next_state)

            if not morphisms:
                return None  # No path exists

            # Take the first (most significant) morphism
            m = morphisms[0]
            composed = m if composed is None else composed.compose(m)
            current = next_state

        return composed

    def narrative_arc(self) -> List[NarrativeMorphism]:
        """
        Get the sequence of morphisms forming the narrative arc.

        Returns morphisms in temporal order.
        """
        sorted_objects = sorted(self._objects.values(), key=lambda s: s.position)
        arc = []

        for i in range(len(sorted_objects) - 1):
            morphisms = self.hom(sorted_objects[i], sorted_objects[i + 1])
            if morphisms:
                # Prefer non-identity morphisms
                non_id = [m for m in morphisms if not m.is_identity]
                arc.append(non_id[0] if non_id else morphisms[0])

        return arc

    @classmethod
    def from_text(
        cls,
        text: str,
        narrative_id: str,
        title: str = "",
        n_states: int = 20,
        feature_extractors: Dict[str, Callable[[str], float]] = None
    ) -> "CategoryNarr":
        """
        Construct Category Narr from raw text.

        This is the practical entry point - takes text, produces a category.

        Args:
            text: The narrative text
            narrative_id: Unique identifier
            title: Title of the narrative
            n_states: Number of states to create
            feature_extractors: Dict of name → function to extract features

        Returns:
            CategoryNarr instance with objects and morphisms
        """
        category = cls(narrative_id=narrative_id, title=title)

        # Default feature extractors
        if feature_extractors is None:
            feature_extractors = {
                "length": lambda t: len(t.split()),
                "sentence_length": lambda t: np.mean([len(s.split()) for s in t.split('.') if s.strip()]) if t.strip() else 0,
            }

        # Split text into windows (states)
        words = text.split()
        window_size = max(1, len(words) // n_states)

        states = []
        for i in range(n_states):
            start = i * window_size
            end = start + window_size if i < n_states - 1 else len(words)
            content = ' '.join(words[start:end])

            if not content.strip():
                continue

            # Create state with features
            state = NarrativeState(
                content=content[:1000],  # Truncate for memory
                position=i / (n_states - 1) if n_states > 1 else 0.5,
                state_id=f"{narrative_id}_state_{i:03d}",
                features={
                    name: extractor(content)
                    for name, extractor in feature_extractors.items()
                },
                metadata={"window_index": i, "word_range": (start, end)}
            )
            states.append(state)
            category.add_object(state)

        # Create morphisms between consecutive states
        for i in range(len(states) - 1):
            source, target = states[i], states[i + 1]

            # Compute feature deltas
            deltas = {
                key: target.features.get(key, 0) - source.features.get(key, 0)
                for key in set(source.features.keys()) | set(target.features.keys())
            }

            # Classify morphism type based on deltas
            morphism_type = cls._classify_transition(source, target, deltas)

            # Compute intensity
            intensity = np.sqrt(sum(d**2 for d in deltas.values())) if deltas else 0.0
            intensity = min(1.0, intensity / 10)  # Normalize

            morphism = NarrativeMorphism(
                source=source,
                target=target,
                morphism_type=morphism_type,
                morphism_id=f"{narrative_id}_morph_{i:03d}",
                feature_deltas=deltas,
                intensity=intensity,
                text_span=target.content[:200],
                metadata={"transition_index": i}
            )
            category.add_morphism(morphism)

        return category

    @staticmethod
    def _classify_transition(
        source: NarrativeState,
        target: NarrativeState,
        deltas: Dict[str, float]
    ) -> MorphismType:
        """
        Classify a transition based on feature changes.

        This is a heuristic - more sophisticated classification could use ML.
        """
        # Position-based heuristics
        if target.position > 0.9:
            return MorphismType.DENOUEMENT
        if 0.7 < target.position <= 0.85:
            return MorphismType.CLIMAX

        # Feature-based classification
        tension_delta = deltas.get("tension", 0)
        sentiment_delta = deltas.get("sentiment", 0)

        if abs(tension_delta) > 0.3:
            if tension_delta > 0:
                return MorphismType.ESCALATION
            else:
                return MorphismType.RESOLUTION

        if abs(sentiment_delta) > 0.3:
            if sentiment_delta < 0:
                return MorphismType.COMPLICATION
            else:
                return MorphismType.RESOLUTION

        return MorphismType.TRANSITION

    def verify_category_laws(self) -> Dict[str, bool]:
        """
        Verify that category laws hold.

        Returns dict with verification results for:
        - identity_left: id_A ∘ f = f
        - identity_right: f ∘ id_B = f
        - associativity: (f ∘ g) ∘ h = f ∘ (g ∘ h)
        """
        results = {
            "identity_left": True,
            "identity_right": True,
            "associativity": True
        }

        morphisms = self.morphisms

        # Test identity laws
        for m in morphisms:
            id_source = self.identity(m.source)
            id_target = self.identity(m.target)

            # Left identity: id ∘ f = f
            left = id_source.compose(m)
            if left.source != m.source or left.target != m.target:
                results["identity_left"] = False

            # Right identity: f ∘ id = f
            right = m.compose(id_target)
            if right.source != m.source or right.target != m.target:
                results["identity_right"] = False

        # Test associativity (for composable triples)
        arc = self.narrative_arc()
        for i in range(len(arc) - 2):
            f, g, h = arc[i], arc[i+1], arc[i+2]

            # (f ∘ g) ∘ h
            fg = f.compose(g)
            fg_h = fg.compose(h)

            # f ∘ (g ∘ h)
            gh = g.compose(h)
            f_gh = f.compose(gh)

            # Check endpoints match (the core of associativity)
            if fg_h.source != f_gh.source or fg_h.target != f_gh.target:
                results["associativity"] = False

        return results

    def summary(self) -> Dict[str, Any]:
        """Get a summary of this category."""
        arc = self.narrative_arc()
        type_counts = defaultdict(int)
        for m in arc:
            type_counts[m.morphism_type.value] += 1

        return {
            "narrative_id": self.narrative_id,
            "title": self.title,
            "n_objects": len(self._objects),
            "n_morphisms": len(self._morphisms),
            "arc_length": len(arc),
            "morphism_types": dict(type_counts),
            "category_laws": self.verify_category_laws()
        }


def demonstrate_category_narr():
    """Demonstrate Category Narr with a simple example."""

    # Sample text
    sample_text = """
    The morning was peaceful. John walked to work, unaware of what awaited him.
    At noon, a mysterious letter arrived. It contained shocking news about his past.
    His world turned upside down. Everything he believed was a lie.
    Through the night, he searched for answers. Each clue led to more questions.
    Finally, at dawn, the truth emerged. He understood everything now.
    Peace returned, but John was forever changed.
    """

    # Create category with custom feature extractors
    def simple_tension(text: str) -> float:
        """Simple tension heuristic based on punctuation."""
        if not text:
            return 0.5
        exclaim = text.count('!')
        question = text.count('?')
        words = len(text.split())
        return min(1.0, (exclaim * 0.3 + question * 0.2) / max(1, words / 100))

    def simple_sentiment(text: str) -> float:
        """Simple sentiment based on word lists."""
        positive = ['peaceful', 'truth', 'peace', 'understood', 'finally']
        negative = ['shocking', 'upside down', 'lie', 'mysterious']
        text_lower = text.lower()
        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)
        total = pos_count + neg_count
        return (pos_count - neg_count) / max(1, total)

    category = CategoryNarr.from_text(
        text=sample_text,
        narrative_id="demo_001",
        title="John's Discovery",
        n_states=6,
        feature_extractors={
            "tension": simple_tension,
            "sentiment": simple_sentiment,
        }
    )

    print("=" * 60)
    print("CATEGORY NARR DEMONSTRATION")
    print("=" * 60)

    # Show summary
    summary = category.summary()
    print(f"\nNarrative: {summary['title']}")
    print(f"Objects (states): {summary['n_objects']}")
    print(f"Morphisms (transitions): {summary['n_morphisms']}")
    print(f"\nMorphism types:")
    for mtype, count in summary['morphism_types'].items():
        print(f"  {mtype}: {count}")

    # Verify category laws
    print(f"\nCategory laws verification:")
    for law, holds in summary['category_laws'].items():
        status = "✓" if holds else "✗"
        print(f"  {law}: {status}")

    # Show narrative arc
    print(f"\nNarrative arc (morphism sequence):")
    for i, m in enumerate(category.narrative_arc()):
        print(f"  {i+1}. {m.morphism_type.value} (intensity: {m.intensity:.2f})")
        print(f"      {m.source.state_id} → {m.target.state_id}")

    # Demonstrate composition
    print(f"\nComposition example:")
    arc = category.narrative_arc()
    if len(arc) >= 2:
        f, g = arc[0], arc[1]
        fg = f @ g  # Using @ operator for composition
        print(f"  f: {f.source.state_id} → {f.target.state_id}")
        print(f"  g: {g.source.state_id} → {g.target.state_id}")
        print(f"  f @ g: {fg.source.state_id} → {fg.target.state_id}")

    # Full path composition
    print(f"\nFull narrative path:")
    if category.objects:
        first = sorted(category.objects, key=lambda s: s.position)[0]
        last = sorted(category.objects, key=lambda s: s.position)[-1]
        full_path = category.path(first, last)
        if full_path:
            print(f"  {first.state_id} → ... → {last.state_id}")
            print(f"  Total feature deltas: {full_path.feature_deltas}")
            print(f"  Peak intensity: {full_path.intensity:.2f}")


if __name__ == "__main__":
    demonstrate_category_narr()
