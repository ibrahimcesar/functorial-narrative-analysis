"""
Natural Transformations between Observation Functors

A natural transformation η: F ⇒ G between functors F, G: Narr → Trajectory
provides a systematic way to transform one observation into another while
preserving categorical structure.

Mathematical definition:
For each object A in Narr, we have a morphism η_A: F(A) → G(A) in Trajectory
such that for any morphism f: A → B in Narr:
    G(f) ∘ η_A = η_B ∘ F(f)  (naturality square commutes)

Practical use:
- Compare how different functors observe the same narrative
- Identify where different measurements agree/disagree
- Combine observations into richer views
"""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Tuple, Any, Optional
import numpy as np

from .narr import CategoryNarr, NarrativeState, NarrativeMorphism
from ..functors.base import Trajectory, BaseFunctor


@dataclass
class TransformationComponent:
    """
    A component η_A of a natural transformation at object A.

    Maps F(A) to G(A) in the target category (Trajectory values).
    """
    state_id: str
    source_value: float  # F(A)
    target_value: float  # G(A)
    transform_value: float  # η_A (the "correction" from F to G)

    @property
    def ratio(self) -> float:
        """Ratio G(A) / F(A) if F(A) != 0."""
        if abs(self.source_value) < 1e-8:
            return 0.0
        return self.target_value / self.source_value


@dataclass
class NaturalTransformation:
    """
    A natural transformation η: F ⇒ G between observation functors.

    Properties we verify:
    1. Components exist for all objects (states)
    2. Naturality squares commute (up to numerical tolerance)

    Use cases:
    - η: F_sentiment ⇒ F_entropy: How sentiment relates to entropy
    - η: F_sentiment ⇒ F_arousal: Sentiment → Arousal mapping
    """
    source_functor: str  # F
    target_functor: str  # G
    components: Dict[str, TransformationComponent] = field(default_factory=dict)
    naturality_errors: List[Tuple[str, str, float]] = field(default_factory=list)

    @property
    def is_natural(self) -> bool:
        """Check if transformation is natural (all squares commute)."""
        if not self.naturality_errors:
            return True
        # Allow small numerical errors
        max_error = max(e[2] for e in self.naturality_errors)
        return max_error < 0.1

    @property
    def mean_error(self) -> float:
        """Mean naturality error (0 = perfectly natural)."""
        if not self.naturality_errors:
            return 0.0
        return np.mean([e[2] for e in self.naturality_errors])

    def component_at(self, state_id: str) -> Optional[TransformationComponent]:
        """Get the component η_A at state A."""
        return self.components.get(state_id)

    def summary(self) -> Dict[str, Any]:
        """Summarize the natural transformation."""
        if not self.components:
            return {"error": "No components computed"}

        transform_values = [c.transform_value for c in self.components.values()]
        ratios = [c.ratio for c in self.components.values() if abs(c.ratio) < 100]

        return {
            "source": self.source_functor,
            "target": self.target_functor,
            "n_components": len(self.components),
            "is_natural": self.is_natural,
            "mean_naturality_error": self.mean_error,
            "transform_stats": {
                "mean": np.mean(transform_values),
                "std": np.std(transform_values),
                "min": np.min(transform_values),
                "max": np.max(transform_values),
            },
            "ratio_stats": {
                "mean": np.mean(ratios) if ratios else 0,
                "std": np.std(ratios) if ratios else 0,
            },
        }


class NaturalTransformationBuilder:
    """
    Builds natural transformations between observation functors.

    Given a narrative (CategoryNarr) and two functors F, G,
    constructs η: F ⇒ G and verifies naturality.
    """

    def __init__(self, category: CategoryNarr):
        self.category = category
        self._functor_values: Dict[str, Dict[str, float]] = {}

    def apply_functor(
        self,
        functor: BaseFunctor,
        functor_name: str = None
    ) -> Dict[str, float]:
        """
        Apply a functor to all states in the category.

        Returns dict mapping state_id → F(state) value.
        """
        name = functor_name or functor.name

        if name in self._functor_values:
            return self._functor_values[name]

        values = {}
        for state in self.category.objects:
            # Apply functor to state's content
            try:
                trajectory = functor([state.content])
                value = trajectory.values[0] if len(trajectory.values) > 0 else 0.0
            except Exception:
                value = 0.0
            values[state.state_id] = value

        self._functor_values[name] = values
        return values

    def apply_from_trajectories(
        self,
        trajectory_f: Trajectory,
        trajectory_g: Trajectory,
        functor_f_name: str,
        functor_g_name: str
    ) -> None:
        """
        Set functor values from pre-computed trajectories.

        Useful when you already have functor outputs.
        """
        states = sorted(self.category.objects, key=lambda s: s.position)

        # Resample trajectories to match state count
        n_states = len(states)
        f_values = np.interp(
            np.linspace(0, 1, n_states),
            trajectory_f.time_points,
            trajectory_f.values
        )
        g_values = np.interp(
            np.linspace(0, 1, n_states),
            trajectory_g.time_points,
            trajectory_g.values
        )

        self._functor_values[functor_f_name] = {
            states[i].state_id: f_values[i] for i in range(n_states)
        }
        self._functor_values[functor_g_name] = {
            states[i].state_id: g_values[i] for i in range(n_states)
        }

    def build_transformation(
        self,
        source_functor: str,
        target_functor: str,
        transform_fn: Callable[[float, float], float] = None
    ) -> NaturalTransformation:
        """
        Build a natural transformation η: F ⇒ G.

        Args:
            source_functor: Name of F
            target_functor: Name of G
            transform_fn: Optional function (f_val, g_val) → transform_val
                         Default is simple difference: g - f

        Returns:
            NaturalTransformation with components and naturality verification
        """
        if source_functor not in self._functor_values:
            raise ValueError(f"Functor {source_functor} not applied. Call apply_functor first.")
        if target_functor not in self._functor_values:
            raise ValueError(f"Functor {target_functor} not applied. Call apply_functor first.")

        f_values = self._functor_values[source_functor]
        g_values = self._functor_values[target_functor]

        if transform_fn is None:
            transform_fn = lambda f, g: g - f  # Default: difference

        # Build components
        components = {}
        for state_id in f_values.keys():
            if state_id not in g_values:
                continue

            f_val = f_values[state_id]
            g_val = g_values[state_id]
            transform_val = transform_fn(f_val, g_val)

            components[state_id] = TransformationComponent(
                state_id=state_id,
                source_value=f_val,
                target_value=g_val,
                transform_value=transform_val
            )

        transformation = NaturalTransformation(
            source_functor=source_functor,
            target_functor=target_functor,
            components=components
        )

        # Verify naturality
        self._verify_naturality(transformation)

        return transformation

    def _verify_naturality(self, transformation: NaturalTransformation) -> None:
        """
        Verify naturality: G(f) ∘ η_A = η_B ∘ F(f) for all morphisms f: A → B.

        In our setting (1D trajectories), this becomes:
        G(B) - G(A) + η_A ≈ η_B + F(B) - F(A)

        Which simplifies to checking that the transform is "coherent" across morphisms.
        """
        f_values = self._functor_values[transformation.source_functor]
        g_values = self._functor_values[transformation.target_functor]

        errors = []
        for morphism in self.category.morphisms:
            src_id = morphism.source.state_id
            tgt_id = morphism.target.state_id

            if src_id not in f_values or tgt_id not in f_values:
                continue
            if src_id not in g_values or tgt_id not in g_values:
                continue

            # F(f) ≈ F(B) - F(A) (change under F)
            f_change = f_values[tgt_id] - f_values[src_id]

            # G(f) ≈ G(B) - G(A) (change under G)
            g_change = g_values[tgt_id] - g_values[src_id]

            # η_A and η_B
            eta_a = transformation.components[src_id].transform_value
            eta_b = transformation.components[tgt_id].transform_value

            # Naturality check: G(f) + η_A ≈ η_B + F(f)
            # Rearranged: G(f) - F(f) ≈ η_B - η_A
            lhs = g_change - f_change
            rhs = eta_b - eta_a

            error = abs(lhs - rhs)
            if error > 0.01:  # Non-trivial error
                errors.append((src_id, tgt_id, error))

        transformation.naturality_errors = errors


class FunctorComparator:
    """
    Compare multiple functors on a narrative using natural transformations.

    Provides insights into how different observation methods relate.
    """

    def __init__(self, category: CategoryNarr):
        self.category = category
        self.builder = NaturalTransformationBuilder(category)
        self.functors: Dict[str, BaseFunctor] = {}
        self.transformations: Dict[Tuple[str, str], NaturalTransformation] = {}

    def add_functor(self, functor: BaseFunctor, name: str = None) -> None:
        """Add a functor to compare."""
        fname = name or functor.name
        self.functors[fname] = functor
        self.builder.apply_functor(functor, fname)

    def add_trajectory(self, trajectory: Trajectory, name: str = None) -> None:
        """Add pre-computed trajectory as a functor output."""
        fname = name or trajectory.functor_name
        states = sorted(self.category.objects, key=lambda s: s.position)
        n_states = len(states)

        # Resample to match states
        values = np.interp(
            np.linspace(0, 1, n_states),
            trajectory.time_points,
            trajectory.values
        )

        self.builder._functor_values[fname] = {
            states[i].state_id: values[i] for i in range(n_states)
        }

    def compare(self, f_name: str, g_name: str) -> NaturalTransformation:
        """Build natural transformation from F to G."""
        key = (f_name, g_name)
        if key not in self.transformations:
            self.transformations[key] = self.builder.build_transformation(f_name, g_name)
        return self.transformations[key]

    def compare_all(self) -> Dict[Tuple[str, str], NaturalTransformation]:
        """Build all pairwise natural transformations."""
        names = list(self.builder._functor_values.keys())
        for i, f_name in enumerate(names):
            for g_name in names[i+1:]:
                self.compare(f_name, g_name)
                self.compare(g_name, f_name)
        return self.transformations

    def correlation_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Compute correlation matrix between all functor outputs.

        Returns (functor_names, correlation_matrix).
        """
        names = list(self.builder._functor_values.keys())
        n = len(names)

        # Get all values in consistent order
        state_ids = sorted(self.builder._functor_values[names[0]].keys())
        matrix = np.zeros((n, len(state_ids)))

        for i, name in enumerate(names):
            for j, sid in enumerate(state_ids):
                matrix[i, j] = self.builder._functor_values[name].get(sid, 0)

        # Compute correlation
        corr = np.corrcoef(matrix)
        return names, corr

    def report(self) -> str:
        """Generate a report comparing all functors."""
        lines = ["=" * 60, "FUNCTOR COMPARISON REPORT", "=" * 60, ""]

        # Correlation matrix
        names, corr = self.correlation_matrix()
        lines.append("Correlation Matrix:")
        header = "            " + "  ".join(f"{n[:8]:>8}" for n in names)
        lines.append(header)
        for i, name in enumerate(names):
            row = f"{name[:10]:>10}  " + "  ".join(f"{corr[i,j]:8.3f}" for j in range(len(names)))
            lines.append(row)

        lines.append("")
        lines.append("Natural Transformations:")

        # Compare all pairs
        self.compare_all()
        for (f, g), transform in self.transformations.items():
            summary = transform.summary()
            natural_str = "✓ natural" if summary["is_natural"] else "✗ not natural"
            lines.append(f"  {f} ⇒ {g}: {natural_str} (error: {summary['mean_naturality_error']:.4f})")

        return "\n".join(lines)


def demonstrate_natural_transformations():
    """Demonstrate natural transformations with a simple example."""
    from .narr import CategoryNarr

    # Create a simple category
    sample_text = """
    The morning was peaceful. John walked to work.
    A mysterious letter arrived. Shocking news.
    His world turned upside down. Everything was a lie.
    He searched for answers. Clues led to questions.
    The truth emerged. He understood everything.
    Peace returned. John was changed.
    """

    category = CategoryNarr.from_text(
        text=sample_text,
        narrative_id="demo",
        title="Demo",
        n_states=6
    )

    print("=" * 60)
    print("NATURAL TRANSFORMATIONS DEMONSTRATION")
    print("=" * 60)

    # Create mock functor outputs
    states = sorted(category.objects, key=lambda s: s.position)
    n = len(states)

    # Mock "sentiment" trajectory
    sentiment_values = np.array([0.5, 0.3, -0.2, -0.1, 0.4, 0.6])[:n]

    # Mock "arousal" trajectory
    arousal_values = np.array([0.3, 0.5, 0.8, 0.7, 0.4, 0.2])[:n]

    builder = NaturalTransformationBuilder(category)

    # Set values manually
    builder._functor_values["sentiment"] = {
        states[i].state_id: sentiment_values[i] for i in range(n)
    }
    builder._functor_values["arousal"] = {
        states[i].state_id: arousal_values[i] for i in range(n)
    }

    # Build transformation
    transform = builder.build_transformation("sentiment", "arousal")
    summary = transform.summary()

    print(f"\nTransformation: {summary['source']} ⇒ {summary['target']}")
    print(f"Is natural: {summary['is_natural']}")
    print(f"Mean error: {summary['mean_naturality_error']:.4f}")
    print(f"\nComponents (η_A for each state):")
    for comp in transform.components.values():
        print(f"  {comp.state_id}: F={comp.source_value:.3f} → G={comp.target_value:.3f} (η={comp.transform_value:.3f})")


if __name__ == "__main__":
    demonstrate_natural_transformations()
