"""
Integration module connecting Category Narr with existing observation functors.

This bridges the abstract categorical framework with practical analysis,
allowing narratives to be:
1. Parsed into Category Narr (objects = states, morphisms = transitions)
2. Observed through multiple functors (F_sentiment, F_entropy, etc.)
3. Compared via natural transformations
4. Classified with ICC model

The full pipeline:
    Text → CategoryNarr → [F_sentiment, F_entropy, ...] → Trajectories → ICC/Patterns
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .narr import CategoryNarr, NarrativeState, NarrativeMorphism, MorphismType
from .natural_transformations import (
    NaturalTransformation,
    NaturalTransformationBuilder,
    FunctorComparator
)
from ..functors.base import Trajectory, BaseFunctor


@dataclass
class FunctorialAnalysis:
    """
    Complete functorial analysis of a narrative.

    Contains:
    - The source category (Narr)
    - Functor outputs (Trajectories)
    - Natural transformations between functors
    - ICC classification
    """
    category: CategoryNarr
    trajectories: Dict[str, Trajectory]
    transformations: Dict[Tuple[str, str], NaturalTransformation]
    functor_correlations: Optional[np.ndarray] = None
    icc_result: Optional[Any] = None

    def summary(self) -> Dict[str, Any]:
        """Summarize the analysis."""
        cat_summary = self.category.summary()

        # Transform summaries
        transform_info = {}
        for (f, g), t in self.transformations.items():
            s = t.summary()
            transform_info[f"{f}⇒{g}"] = {
                "is_natural": s["is_natural"],
                "mean_error": s["mean_naturality_error"]
            }

        return {
            "narrative_id": cat_summary["narrative_id"],
            "title": cat_summary["title"],
            "n_states": cat_summary["n_objects"],
            "n_morphisms": cat_summary["n_morphisms"],
            "category_laws_verified": all(cat_summary["category_laws"].values()),
            "functors_applied": list(self.trajectories.keys()),
            "natural_transformations": transform_info,
            "icc_class": self.icc_result.icc_class if self.icc_result else None
        }


class NarrativeFunctorialAnalyzer:
    """
    Main analyzer that combines Category Narr with observation functors.

    This is the practical entry point for functorial narrative analysis.
    """

    def __init__(self, functors: List[BaseFunctor] = None):
        """
        Initialize analyzer with a set of observation functors.

        Args:
            functors: List of functors to apply. If None, uses defaults.
        """
        self.functors = functors or []
        self._functor_names = [f.name for f in self.functors]

    def add_functor(self, functor: BaseFunctor) -> None:
        """Add an observation functor."""
        self.functors.append(functor)
        self._functor_names.append(functor.name)

    def analyze(
        self,
        text: str,
        narrative_id: str,
        title: str = "",
        n_states: int = 20
    ) -> FunctorialAnalysis:
        """
        Perform complete functorial analysis on a text.

        This is the main entry point that:
        1. Constructs Category Narr from text
        2. Applies all functors to produce trajectories
        3. Computes natural transformations between functors
        4. Optionally classifies with ICC

        Args:
            text: The narrative text
            narrative_id: Unique identifier
            title: Title of the narrative
            n_states: Number of states to create

        Returns:
            FunctorialAnalysis with all results
        """
        # 1. Construct Category Narr
        category = CategoryNarr.from_text(
            text=text,
            narrative_id=narrative_id,
            title=title,
            n_states=n_states,
            feature_extractors=self._get_feature_extractors()
        )

        # 2. Apply functors
        trajectories = {}
        for functor in self.functors:
            try:
                traj = functor.process_text(text)
                trajectories[functor.name] = traj
            except Exception as e:
                print(f"Warning: Functor {functor.name} failed: {e}")

        # 3. Build natural transformations
        comparator = FunctorComparator(category)
        for name, traj in trajectories.items():
            comparator.add_trajectory(traj, name)

        transformations = {}
        if len(trajectories) >= 2:
            transformations = comparator.compare_all()

        # 4. Compute correlations
        correlations = None
        if len(trajectories) >= 2:
            _, correlations = comparator.correlation_matrix()

        # 5. Update category states with functor values
        self._enrich_category(category, trajectories)

        return FunctorialAnalysis(
            category=category,
            trajectories=trajectories,
            transformations=transformations,
            functor_correlations=correlations
        )

    def _get_feature_extractors(self) -> Dict[str, Any]:
        """Get feature extractors for category construction."""
        return {
            "length": lambda t: len(t.split()),
            "sentence_length": lambda t: (
                np.mean([len(s.split()) for s in t.split('.') if s.strip()])
                if t.strip() else 0
            ),
        }

    def _enrich_category(
        self,
        category: CategoryNarr,
        trajectories: Dict[str, Trajectory]
    ) -> None:
        """
        Enrich category states with functor values.

        Updates state.features with values from each functor.
        """
        states = sorted(category.objects, key=lambda s: s.position)
        n_states = len(states)

        for functor_name, traj in trajectories.items():
            # Resample trajectory to match states
            values = np.interp(
                np.linspace(0, 1, n_states),
                traj.time_points,
                traj.values
            )

            for i, state in enumerate(states):
                state.features[functor_name] = float(values[i])

        # Update morphism deltas
        for morphism in category.morphisms:
            for functor_name in trajectories.keys():
                src_val = morphism.source.features.get(functor_name, 0)
                tgt_val = morphism.target.features.get(functor_name, 0)
                morphism.feature_deltas[functor_name] = tgt_val - src_val

    def analyze_with_icc(
        self,
        text: str,
        narrative_id: str,
        title: str = "",
        n_states: int = 20,
        primary_functor: str = None
    ) -> FunctorialAnalysis:
        """
        Analyze and classify with ICC model.

        Args:
            text: The narrative text
            narrative_id: Unique identifier
            title: Title of the narrative
            n_states: Number of states
            primary_functor: Which functor's trajectory to use for ICC
                            (default: first available)

        Returns:
            FunctorialAnalysis with ICC classification
        """
        analysis = self.analyze(text, narrative_id, title, n_states)

        # Import ICC detector
        try:
            from ..detectors.icc import ICCDetector
            detector = ICCDetector()

            # Choose trajectory for ICC
            if not analysis.trajectories:
                return analysis

            if primary_functor and primary_functor in analysis.trajectories:
                traj = analysis.trajectories[primary_functor]
            else:
                traj = list(analysis.trajectories.values())[0]

            # Normalize trajectory for ICC
            normalized = traj.normalize()
            analysis.icc_result = detector.detect(
                normalized.values,
                trajectory_id=narrative_id,
                title=title
            )

        except Exception as e:
            print(f"Warning: ICC classification failed: {e}")

        return analysis


def analyze_morphism_patterns(category: CategoryNarr) -> Dict[str, Any]:
    """
    Analyze the pattern of morphism types in a narrative.

    Returns statistics about the narrative's morphism sequence.
    """
    arc = category.narrative_arc()
    if not arc:
        return {"error": "No morphisms in arc"}

    # Count types
    type_counts = {}
    for m in arc:
        t = m.morphism_type.value
        type_counts[t] = type_counts.get(t, 0) + 1

    # Find climax position
    climax_positions = [
        i / len(arc) for i, m in enumerate(arc)
        if m.morphism_type == MorphismType.CLIMAX
    ]

    # Intensity profile
    intensities = [m.intensity for m in arc]

    # Peak intensity position
    if intensities:
        peak_idx = np.argmax(intensities)
        peak_position = peak_idx / len(arc)
    else:
        peak_position = 0.5

    return {
        "arc_length": len(arc),
        "morphism_types": type_counts,
        "climax_positions": climax_positions,
        "peak_intensity": max(intensities) if intensities else 0,
        "peak_position": peak_position,
        "mean_intensity": np.mean(intensities) if intensities else 0,
        "intensity_variance": np.var(intensities) if intensities else 0
    }


def demonstrate_integration():
    """Demonstrate the full integration."""

    # Sample narrative
    sample_text = """
    The kingdom had known peace for a generation. Young Prince Aldric spent his days
    in scholarly pursuits, unaware of the darkness gathering at the borders.

    One autumn evening, a wounded messenger arrived at the castle gates. He carried
    news of invasion - the Shadow Host had crossed the mountains. Everything Aldric
    knew was about to change forever.

    The king called his council. Plans were made in haste. Aldric was sent east with
    a small company to seek allies among the mountain clans. It was his first true
    test as a prince.

    The journey was perilous. Bandits attacked at the river crossing. Three soldiers
    fell. Aldric himself was wounded, but he pressed on. The mountain paths were
    treacherous, and winter came early.

    At last they reached the clan holds. But the chieftains were suspicious - why
    should they aid the lowland king who had ignored them for decades? Aldric had
    to prove himself in their ritual combat.

    He fought the champion Rothgar for three hours under the watching eyes of the
    clans. When he finally won, bloodied but standing, the chieftains pledged their
    spears to his cause.

    The combined forces descended from the mountains. At the Battle of Thornfield,
    Aldric led the mountain warriors in a charge that broke the Shadow Host's flank.
    The enemy general fell to his sword.

    Victory came at great cost. The kingdom was saved, but Aldric's father had fallen
    defending the capital. The young prince returned home as king, forever changed by
    war. Peace returned, but it was a different peace - hard-won and precious.
    """

    print("=" * 70)
    print("FUNCTORIAL NARRATIVE ANALYSIS - FULL INTEGRATION DEMO")
    print("=" * 70)

    # Create category
    category = CategoryNarr.from_text(
        text=sample_text,
        narrative_id="prince_aldric",
        title="The Prince's Trial",
        n_states=8
    )

    print(f"\n1. CATEGORY NARR CONSTRUCTION")
    print(f"   Title: {category.title}")
    print(f"   Objects (states): {len(category.objects)}")
    print(f"   Morphisms: {len(category.morphisms)}")

    # Verify category laws
    laws = category.verify_category_laws()
    print(f"\n   Category laws:")
    for law, holds in laws.items():
        status = "✓" if holds else "✗"
        print(f"     {law}: {status}")

    # Morphism analysis
    print(f"\n2. MORPHISM PATTERN ANALYSIS")
    patterns = analyze_morphism_patterns(category)
    print(f"   Arc length: {patterns['arc_length']}")
    print(f"   Peak intensity: {patterns['peak_intensity']:.2f} at position {patterns['peak_position']:.2f}")
    print(f"   Morphism types:")
    for mtype, count in patterns['morphism_types'].items():
        print(f"     {mtype}: {count}")

    # Show narrative arc
    print(f"\n3. NARRATIVE ARC (sequence of morphisms)")
    arc = category.narrative_arc()
    for i, m in enumerate(arc):
        print(f"   {i+1}. [{m.morphism_type.value}] intensity={m.intensity:.2f}")

    # Demonstrate composition
    print(f"\n4. MORPHISM COMPOSITION")
    if len(arc) >= 3:
        f, g, h = arc[0], arc[1], arc[2]
        print(f"   f: {f.morphism_type.value}")
        print(f"   g: {g.morphism_type.value}")
        print(f"   h: {h.morphism_type.value}")

        fg = f @ g
        print(f"   f @ g: spans {fg.source.state_id} → {fg.target.state_id}")

        fgh = f @ g @ h
        print(f"   f @ g @ h: spans {fgh.source.state_id} → {fgh.target.state_id}")

    # Full path
    print(f"\n5. FULL NARRATIVE PATH (composed morphism)")
    first = sorted(category.objects, key=lambda s: s.position)[0]
    last = sorted(category.objects, key=lambda s: s.position)[-1]
    full_path = category.path(first, last)
    if full_path:
        print(f"   Start → End composition exists")
        print(f"   Peak intensity along path: {full_path.intensity:.2f}")

    print("\n" + "=" * 70)
    print("Category Narr is now PRACTICAL - not just theoretical!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_integration()
