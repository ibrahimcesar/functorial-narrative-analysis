"""
Falsifiability Analysis for Narrative Structure Models

Measures the epistemological validity of narrative models by testing:
1. SPECIFICITY: What % of random trajectories pass? (Lower = more meaningful)
2. DISCRIMINATION: Can models distinguish genuinely different structures?
3. OVERLAP: How much do models agree? (High overlap = redundant models)
4. NULL HYPOTHESIS: Would shuffled texts still "fit"?

A good model should:
- Reject most random/noise trajectories
- Distinguish between qualitatively different narratives
- Not overlap completely with other models
- Fail on shuffled/destroyed text structure

Models that pass everything are measuring nothing.

"The criterion of the scientific status of a theory is its falsifiability."
    - Karl Popper
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter

from src.detectors.kishotenketsu import KishotenketsuDetector
from src.detectors.aristotle import AristotleDetector
from src.detectors.freytag import FreytagDetector
from src.detectors.harmon_circle import HarmonCircleDetector
from src.detectors.campbell import CampbellDetector
from src.detectors.reagan_shapes import ReaganClassifier


@dataclass
class FalsifiabilityMetrics:
    """Metrics measuring how falsifiable a model is."""
    model_name: str

    # Specificity: % of random trajectories rejected
    random_rejection_rate: float  # Higher = more specific
    noise_rejection_rate: float   # Rejection of pure noise

    # Null hypothesis tests
    shuffled_pass_rate: float     # % of shuffled texts that still pass
    reversed_pass_rate: float     # % of reversed texts that pass

    # Discrimination power
    real_vs_random_auc: float     # AUC for distinguishing real from random
    cross_cultural_discrimination: float  # Can distinguish JP from Western?

    # Overlap with other models
    model_correlations: Dict[str, float] = field(default_factory=dict)

    # Derived scores
    falsifiability_score: float = 0.0  # 0-1, higher = more falsifiable
    scientific_validity: str = ""      # Interpretation

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "random_rejection_rate": float(self.random_rejection_rate),
            "noise_rejection_rate": float(self.noise_rejection_rate),
            "shuffled_pass_rate": float(self.shuffled_pass_rate),
            "reversed_pass_rate": float(self.reversed_pass_rate),
            "real_vs_random_auc": float(self.real_vs_random_auc),
            "cross_cultural_discrimination": float(self.cross_cultural_discrimination),
            "model_correlations": {k: float(v) for k, v in self.model_correlations.items()},
            "falsifiability_score": float(self.falsifiability_score),
            "scientific_validity": self.scientific_validity,
        }


@dataclass
class EpistemologicalReport:
    """Complete epistemological analysis of narrative models."""
    model_metrics: Dict[str, FalsifiabilityMetrics]
    model_ranking: List[str]  # Best to worst falsifiability
    redundancy_clusters: List[List[str]]  # Groups of highly correlated models
    recommendations: List[str]
    summary: str

    def to_dict(self) -> dict:
        return {
            "model_metrics": {k: v.to_dict() for k, v in self.model_metrics.items()},
            "model_ranking": self.model_ranking,
            "redundancy_clusters": self.redundancy_clusters,
            "recommendations": self.recommendations,
            "summary": self.summary,
        }


class FalsifiabilityAnalyzer:
    """
    Analyzes the epistemological validity of narrative structure models.

    Tests whether models can be falsified or if they're so loose
    they can fit anything (and thus explain nothing).
    """

    def __init__(self, n_random_samples: int = 500, n_bootstrap: int = 100):
        self.n_random_samples = n_random_samples
        self.n_bootstrap = n_bootstrap

        # Initialize all detectors
        self.detectors = {
            "kishotenketsu": KishotenketsuDetector(),
            "aristotle": AristotleDetector(),
            "freytag": FreytagDetector(),
            "harmon_circle": HarmonCircleDetector(),
            "campbell": CampbellDetector(),
        }
        self.reagan_classifier = ReaganClassifier()

    def _generate_random_trajectory(self, length: int = 100) -> np.ndarray:
        """Generate purely random trajectory (uniform noise)."""
        return np.random.uniform(0, 1, length)

    def _generate_brownian_trajectory(self, length: int = 100) -> np.ndarray:
        """Generate random walk (more realistic random narrative)."""
        steps = np.random.normal(0, 0.1, length)
        trajectory = np.cumsum(steps)
        # Normalize to [0, 1]
        trajectory = (trajectory - trajectory.min()) / (trajectory.max() - trajectory.min() + 1e-8)
        return trajectory

    def _generate_smooth_random(self, length: int = 100) -> np.ndarray:
        """Generate smooth random trajectory (random but narratively plausible)."""
        # Random trajectory
        raw = np.random.uniform(0, 1, length)
        # Smooth it heavily
        if length > 10:
            smoothed = savgol_filter(raw, min(51, length // 2 * 2 + 1), 3)
        else:
            smoothed = raw
        # Normalize
        smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)
        return smoothed

    def _shuffle_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Shuffle trajectory segments (destroys structure, preserves distribution)."""
        # Split into chunks and shuffle
        n_chunks = 10
        chunk_size = len(trajectory) // n_chunks
        chunks = [trajectory[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]
        np.random.shuffle(chunks)
        return np.concatenate(chunks)

    def _reverse_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Reverse trajectory (tests directionality sensitivity)."""
        return trajectory[::-1].copy()

    def _get_detector_confidence(self, detector_name: str, trajectory: np.ndarray) -> float:
        """Get confidence score from a detector."""
        try:
            if detector_name == "reagan_shapes":
                result = self.reagan_classifier.classify(trajectory)
                return result.confidence
            else:
                detector = self.detectors[detector_name]
                result = detector.detect(trajectory)

                if hasattr(result, 'conformance_score'):
                    return result.conformance_score
                elif hasattr(result, 'confidence'):
                    return result.confidence
                elif hasattr(result, 'score'):
                    return result.score
                else:
                    return 0.5
        except Exception as e:
            return 0.0

    def _compute_auc(self, real_scores: List[float], random_scores: List[float]) -> float:
        """Compute AUC for distinguishing real from random trajectories."""
        from sklearn.metrics import roc_auc_score

        labels = [1] * len(real_scores) + [0] * len(random_scores)
        scores = list(real_scores) + list(random_scores)

        if len(set(labels)) < 2 or len(scores) < 2:
            return 0.5

        try:
            return roc_auc_score(labels, scores)
        except:
            return 0.5

    def analyze_model(
        self,
        model_name: str,
        real_trajectories: List[np.ndarray],
        detection_threshold: float = 0.5
    ) -> FalsifiabilityMetrics:
        """
        Analyze falsifiability of a single model.

        Args:
            model_name: Name of the detector
            real_trajectories: List of real narrative trajectories
            detection_threshold: Threshold for "detection"
        """
        print(f"    Analyzing {model_name}...")

        # Get scores on real trajectories
        real_scores = []
        for traj in real_trajectories:
            score = self._get_detector_confidence(model_name, traj)
            real_scores.append(score)

        # Test 1: Random trajectory rejection
        random_scores = []
        for _ in range(self.n_random_samples):
            traj = self._generate_smooth_random(100)
            score = self._get_detector_confidence(model_name, traj)
            random_scores.append(score)

        random_pass_rate = sum(1 for s in random_scores if s > detection_threshold) / len(random_scores)
        random_rejection_rate = 1 - random_pass_rate

        # Test 2: Pure noise rejection
        noise_scores = []
        for _ in range(self.n_random_samples // 2):
            traj = self._generate_random_trajectory(100)
            score = self._get_detector_confidence(model_name, traj)
            noise_scores.append(score)

        noise_pass_rate = sum(1 for s in noise_scores if s > detection_threshold) / len(noise_scores)
        noise_rejection_rate = 1 - noise_pass_rate

        # Test 3: Shuffled text pass rate (null hypothesis)
        shuffled_scores = []
        for traj in real_trajectories[:min(50, len(real_trajectories))]:
            shuffled = self._shuffle_trajectory(traj)
            score = self._get_detector_confidence(model_name, shuffled)
            shuffled_scores.append(score)

        shuffled_pass_rate = sum(1 for s in shuffled_scores if s > detection_threshold) / len(shuffled_scores) if shuffled_scores else 0

        # Test 4: Reversed trajectory pass rate
        reversed_scores = []
        for traj in real_trajectories[:min(50, len(real_trajectories))]:
            reversed_traj = self._reverse_trajectory(traj)
            score = self._get_detector_confidence(model_name, reversed_traj)
            reversed_scores.append(score)

        reversed_pass_rate = sum(1 for s in reversed_scores if s > detection_threshold) / len(reversed_scores) if reversed_scores else 0

        # Test 5: AUC for real vs random
        auc = self._compute_auc(real_scores, random_scores)

        # Compute falsifiability score
        # Higher is better: rejects random, rejects shuffled, high AUC
        falsifiability_score = (
            random_rejection_rate * 0.3 +
            noise_rejection_rate * 0.2 +
            (1 - shuffled_pass_rate) * 0.25 +
            (1 - reversed_pass_rate) * 0.1 +
            (auc - 0.5) * 2 * 0.15  # Scale AUC contribution
        )
        falsifiability_score = max(0, min(1, falsifiability_score))

        # Interpretation
        if falsifiability_score > 0.7:
            validity = "GOOD: Model is meaningfully selective"
        elif falsifiability_score > 0.5:
            validity = "MODERATE: Model shows some discrimination"
        elif falsifiability_score > 0.3:
            validity = "WEAK: Model accepts too much noise"
        else:
            validity = "POOR: Model is essentially unfalsifiable"

        return FalsifiabilityMetrics(
            model_name=model_name,
            random_rejection_rate=random_rejection_rate,
            noise_rejection_rate=noise_rejection_rate,
            shuffled_pass_rate=shuffled_pass_rate,
            reversed_pass_rate=reversed_pass_rate,
            real_vs_random_auc=auc,
            cross_cultural_discrimination=0.0,  # Filled in later
            falsifiability_score=falsifiability_score,
            scientific_validity=validity,
        )

    def compute_model_correlations(
        self,
        trajectories: List[np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute correlations between model scores.

        High correlation = models are redundant (measuring same thing)
        """
        model_names = list(self.detectors.keys()) + ["reagan_shapes"]

        # Get all scores
        all_scores = {name: [] for name in model_names}

        for traj in trajectories[:min(100, len(trajectories))]:
            for name in model_names:
                score = self._get_detector_confidence(name, traj)
                all_scores[name].append(score)

        # Compute pairwise correlations
        correlations = {}
        for name1 in model_names:
            correlations[name1] = {}
            for name2 in model_names:
                if name1 == name2:
                    correlations[name1][name2] = 1.0
                else:
                    try:
                        corr, _ = stats.pearsonr(all_scores[name1], all_scores[name2])
                        correlations[name1][name2] = float(corr) if not np.isnan(corr) else 0.0
                    except:
                        correlations[name1][name2] = 0.0

        return correlations

    def find_redundancy_clusters(
        self,
        correlations: Dict[str, Dict[str, float]],
        threshold: float = 0.7
    ) -> List[List[str]]:
        """Find clusters of highly correlated (redundant) models."""
        model_names = list(correlations.keys())
        clusters = []
        used = set()

        for name1 in model_names:
            if name1 in used:
                continue

            cluster = [name1]
            for name2 in model_names:
                if name2 != name1 and name2 not in used:
                    if abs(correlations[name1].get(name2, 0)) > threshold:
                        cluster.append(name2)
                        used.add(name2)

            if len(cluster) > 1:
                clusters.append(cluster)
            used.add(name1)

        return clusters

    def analyze_cross_cultural_discrimination(
        self,
        japanese_trajectories: List[np.ndarray],
        western_trajectories: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Test if models can distinguish Japanese from Western narratives.

        Higher discrimination = model captures cultural differences.
        """
        model_names = list(self.detectors.keys()) + ["reagan_shapes"]
        discrimination = {}

        for name in model_names:
            jp_scores = [self._get_detector_confidence(name, t) for t in japanese_trajectories[:50]]
            we_scores = [self._get_detector_confidence(name, t) for t in western_trajectories[:50]]

            if len(jp_scores) > 3 and len(we_scores) > 3:
                try:
                    # Mann-Whitney U test
                    stat, p_value = stats.mannwhitneyu(jp_scores, we_scores, alternative='two-sided')
                    # Effect size (rank-biserial correlation)
                    n1, n2 = len(jp_scores), len(we_scores)
                    effect = 1 - (2 * stat) / (n1 * n2)
                    discrimination[name] = abs(effect)
                except:
                    discrimination[name] = 0.0
            else:
                discrimination[name] = 0.0

        return discrimination

    def generate_report(
        self,
        real_trajectories: List[np.ndarray],
        japanese_trajectories: Optional[List[np.ndarray]] = None,
        western_trajectories: Optional[List[np.ndarray]] = None,
    ) -> EpistemologicalReport:
        """
        Generate complete epistemological analysis.
        """
        print("\n" + "=" * 60)
        print("EPISTEMOLOGICAL FALSIFIABILITY ANALYSIS")
        print("Testing scientific validity of narrative models")
        print("=" * 60)

        model_names = list(self.detectors.keys()) + ["reagan_shapes"]

        # Analyze each model
        print("\n[1/4] Analyzing model falsifiability...")
        model_metrics = {}
        for name in model_names:
            metrics = self.analyze_model(name, real_trajectories)
            model_metrics[name] = metrics

        # Compute cross-cultural discrimination
        if japanese_trajectories and western_trajectories:
            print("\n[2/4] Testing cross-cultural discrimination...")
            discrimination = self.analyze_cross_cultural_discrimination(
                japanese_trajectories, western_trajectories
            )
            for name, disc in discrimination.items():
                if name in model_metrics:
                    model_metrics[name].cross_cultural_discrimination = disc

        # Compute model correlations
        print("\n[3/4] Computing model correlations...")
        correlations = self.compute_model_correlations(real_trajectories)
        for name in model_names:
            if name in model_metrics:
                model_metrics[name].model_correlations = correlations.get(name, {})

        # Find redundancy clusters
        redundancy_clusters = self.find_redundancy_clusters(correlations)

        # Rank models by falsifiability
        model_ranking = sorted(
            model_names,
            key=lambda n: model_metrics[n].falsifiability_score,
            reverse=True
        )

        # Generate recommendations
        print("\n[4/4] Generating recommendations...")
        recommendations = self._generate_recommendations(model_metrics, redundancy_clusters)

        # Summary
        summary = self._generate_summary(model_metrics, model_ranking, redundancy_clusters)

        return EpistemologicalReport(
            model_metrics=model_metrics,
            model_ranking=model_ranking,
            redundancy_clusters=redundancy_clusters,
            recommendations=recommendations,
            summary=summary,
        )

    def _generate_recommendations(
        self,
        metrics: Dict[str, FalsifiabilityMetrics],
        clusters: List[List[str]]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recs = []

        # Check for unfalsifiable models
        unfalsifiable = [n for n, m in metrics.items() if m.falsifiability_score < 0.3]
        if unfalsifiable:
            recs.append(
                f"EPISTEMOLOGICAL CONCERN: {', '.join(unfalsifiable)} show low falsifiability. "
                "These models may be too loose to make meaningful claims about narrative structure."
            )

        # Check for redundancy
        if clusters:
            for cluster in clusters:
                recs.append(
                    f"REDUNDANCY: {', '.join(cluster)} are highly correlated. "
                    "These models may be measuring the same underlying property."
                )

        # Check shuffled pass rates
        high_shuffle = [n for n, m in metrics.items() if m.shuffled_pass_rate > 0.5]
        if high_shuffle:
            recs.append(
                f"NULL HYPOTHESIS FAILURE: {', '.join(high_shuffle)} pass on shuffled texts. "
                "These models don't require temporal coherence - any distribution works."
            )

        # Positive recommendations
        good_models = [n for n, m in metrics.items() if m.falsifiability_score > 0.6]
        if good_models:
            recs.append(
                f"RECOMMENDED: {', '.join(good_models)} show meaningful selectivity "
                "and could support valid scientific claims."
            )

        # Cultural discrimination
        discriminating = [n for n, m in metrics.items() if m.cross_cultural_discrimination > 0.3]
        if discriminating:
            recs.append(
                f"CULTURAL SENSITIVITY: {', '.join(discriminating)} can distinguish "
                "Japanese from Western narratives, supporting culture-specific analysis."
            )

        return recs

    def _generate_summary(
        self,
        metrics: Dict[str, FalsifiabilityMetrics],
        ranking: List[str],
        clusters: List[List[str]]
    ) -> str:
        """Generate human-readable summary."""
        lines = [
            "FALSIFIABILITY ANALYSIS SUMMARY",
            "=" * 40,
            "",
            "Model Rankings (Most to Least Falsifiable):",
        ]

        for i, name in enumerate(ranking, 1):
            m = metrics[name]
            lines.append(
                f"  {i}. {name}: {m.falsifiability_score:.3f} - {m.scientific_validity}"
            )

        lines.extend([
            "",
            "Key Metrics:",
        ])

        for name in ranking:
            m = metrics[name]
            lines.append(f"\n  {name}:")
            lines.append(f"    Random rejection: {m.random_rejection_rate:.1%}")
            lines.append(f"    Shuffled pass rate: {m.shuffled_pass_rate:.1%} (lower = better)")
            lines.append(f"    Real vs Random AUC: {m.real_vs_random_auc:.3f}")
            lines.append(f"    Cultural discrimination: {m.cross_cultural_discrimination:.3f}")

        if clusters:
            lines.extend([
                "",
                "Redundant Model Clusters:",
            ])
            for cluster in clusters:
                lines.append(f"  - {', '.join(cluster)}")

        # Overall assessment
        avg_falsifiability = np.mean([m.falsifiability_score for m in metrics.values()])
        lines.extend([
            "",
            f"Overall Assessment:",
            f"  Average Falsifiability: {avg_falsifiability:.3f}",
        ])

        if avg_falsifiability < 0.4:
            lines.append(
                "  CONCERN: Most models show weak falsifiability. "
                "Traditional narrative models may be epistemologically problematic."
            )
        elif avg_falsifiability < 0.6:
            lines.append(
                "  MODERATE: Models show mixed validity. "
                "Use with caution and prefer higher-ranked models."
            )
        else:
            lines.append(
                "  GOOD: Models generally show meaningful selectivity."
            )

        return "\n".join(lines)


def run_falsifiability_analysis():
    """Run complete falsifiability analysis on corpus."""
    from src.geometry.surprisal import SurprisalExtractor

    print("=" * 60)
    print("FALSIFIABILITY ANALYSIS")
    print("Testing if narrative models can be scientifically falsified")
    print("=" * 60)

    # Load trajectories from corpus
    extractor = SurprisalExtractor(method="entropy")

    japanese_trajectories = []
    western_trajectories = []

    # Load Japanese texts
    print("\n[1/3] Loading Japanese trajectories...")
    jp_locations = [
        Path("data/raw/aozora_extended/texts"),
        Path("data/raw/aozora"),
    ]

    for jp_dir in jp_locations:
        if not jp_dir.exists():
            continue
        for f in list(jp_dir.glob("*.json"))[:30]:
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                content = data.get("content", data.get("text", ""))
                if len(content) > 5000:
                    traj = extractor.extract(content)
                    japanese_trajectories.append(np.array(traj.values))
            except:
                pass

    print(f"  Loaded {len(japanese_trajectories)} Japanese trajectories")

    # Load Western texts
    print("\n[2/3] Loading Western trajectories...")
    we_locations = [
        Path("data/raw/gutenberg"),
        Path("data/raw/english"),
    ]

    for we_dir in we_locations:
        if not we_dir.exists():
            continue
        for f in list(we_dir.glob("*.json"))[:30]:
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                content = data.get("content", data.get("text", ""))
                if len(content) > 5000:
                    traj = extractor.extract(content)
                    western_trajectories.append(np.array(traj.values))
            except:
                pass

    print(f"  Loaded {len(western_trajectories)} Western trajectories")

    # Combine for general analysis
    all_trajectories = japanese_trajectories + western_trajectories

    if len(all_trajectories) < 10:
        print("\nERROR: Not enough trajectories for analysis")
        return None

    # Run analysis
    print("\n[3/3] Running falsifiability analysis...")
    analyzer = FalsifiabilityAnalyzer(n_random_samples=200)

    report = analyzer.generate_report(
        real_trajectories=all_trajectories,
        japanese_trajectories=japanese_trajectories,
        western_trajectories=western_trajectories,
    )

    # Print summary
    print("\n" + "=" * 60)
    print(report.summary)
    print("=" * 60)

    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"  • {rec}")

    # Save results
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "falsifiability_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    return report


if __name__ == "__main__":
    run_falsifiability_analysis()
