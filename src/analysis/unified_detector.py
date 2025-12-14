"""
Unified Detector Analysis

Runs ALL narrative structure detectors on each text to provide
comprehensive cross-cultural structural analysis.

Detectors:
    1. KishotenketsuDetector - Japanese 4-act structure (起承転結)
    2. AristotleDetector - Classical 3-act dramatic structure
    3. FreytagDetector - 5-act Freytag's Pyramid
    4. HarmonCircleDetector - Dan Harmon's 8-beat Story Circle
    5. CampbellDetector - Hero's Journey (17 stages)
    6. ReaganClassifier - 6 emotional arc shapes

This allows us to answer: Which structural models fit which cultures?
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import numpy as np

from src.detectors.kishotenketsu import KishotenketsuDetector
from src.detectors.aristotle import AristotleDetector
from src.detectors.freytag import FreytagDetector
from src.detectors.harmon_circle import HarmonCircleDetector
from src.detectors.campbell import CampbellDetector
from src.detectors.reagan_shapes import ReaganClassifier
from src.geometry.surprisal import SurprisalExtractor


@dataclass
class DetectorResult:
    """Result from a single detector."""
    detector_name: str
    detected: bool
    confidence: float
    pattern_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "detector_name": self.detector_name,
            "detected": bool(self.detected),
            "confidence": float(self.confidence),
            "pattern_name": self.pattern_name,
            "details": self.details
        }


@dataclass
class UnifiedAnalysis:
    """Complete analysis of a text using all detectors."""
    work_id: str
    title: str
    author: str
    culture: str  # 'japanese', 'western', etc.
    detector_results: Dict[str, DetectorResult]
    best_fit_model: str
    best_fit_confidence: float
    model_ranking: List[str]
    cultural_alignment: Dict[str, float]  # Expected vs actual fit

    def to_dict(self) -> dict:
        return {
            "work_id": self.work_id,
            "title": self.title,
            "author": self.author,
            "culture": self.culture,
            "detector_results": {k: v.to_dict() for k, v in self.detector_results.items()},
            "best_fit_model": self.best_fit_model,
            "best_fit_confidence": float(self.best_fit_confidence),
            "model_ranking": self.model_ranking,
            "cultural_alignment": {k: float(v) for k, v in self.cultural_alignment.items()}
        }


class UnifiedDetector:
    """
    Runs all narrative structure detectors on texts.

    Provides comprehensive analysis to test cross-cultural
    structural universality hypotheses.
    """

    # Cultural origin of each model
    MODEL_ORIGINS = {
        "kishotenketsu": "japanese",
        "aristotle": "western",
        "freytag": "western",
        "harmon_circle": "western",
        "campbell": "western",
        "reagan_shapes": "western"
    }

    def __init__(self, method: str = "entropy"):
        """Initialize all detectors."""
        self.surprisal_extractor = SurprisalExtractor(method=method)

        # Initialize detectors
        self.detectors = {
            "kishotenketsu": KishotenketsuDetector(),
            "aristotle": AristotleDetector(),
            "freytag": FreytagDetector(),
            "harmon_circle": HarmonCircleDetector(),
            "campbell": CampbellDetector(),
        }

        self.reagan_classifier = ReaganClassifier()

    def analyze_text(
        self,
        text: str,
        work_id: str,
        title: str,
        author: str,
        culture: str
    ) -> UnifiedAnalysis:
        """
        Run all detectors on a single text.

        Args:
            text: Full text content
            work_id: Unique identifier
            title: Work title
            author: Author name
            culture: Cultural origin ('japanese', 'western')

        Returns:
            UnifiedAnalysis with all detector results
        """
        # Extract information-theoretic trajectory
        trajectory = self.surprisal_extractor.extract(text)
        values = np.array(trajectory.values)

        detector_results = {}

        # Run each structural detector
        for name, detector in self.detectors.items():
            try:
                result = detector.detect(values, trajectory_id=work_id, title=title)

                # Extract confidence/conformance score
                if hasattr(result, 'conformance_score'):
                    confidence = result.conformance_score
                elif hasattr(result, 'confidence'):
                    confidence = result.confidence
                elif hasattr(result, 'score'):
                    confidence = result.score
                else:
                    confidence = 0.5

                # Determine if pattern detected (threshold varies by detector)
                if name == "kishotenketsu":
                    detected = getattr(result, 'is_kishotenketsu', confidence > 0.3)
                    pattern_name = "kishōtenketsu" if detected else None
                elif name == "campbell":
                    detected = getattr(result, 'is_heros_journey', confidence > 0.5)
                    pattern_name = "hero's journey" if detected else None
                elif name == "harmon_circle":
                    detected = confidence > 0.5
                    pattern_name = "story circle" if detected else None
                elif name == "freytag":
                    detected = confidence > 0.5
                    pattern_name = "freytag pyramid" if detected else None
                elif name == "aristotle":
                    detected = confidence > 0.5
                    pattern_name = "three-act structure" if detected else None
                else:
                    detected = confidence > 0.5
                    pattern_name = name if detected else None

                # Extract additional details
                details = {}
                if hasattr(result, 'to_dict'):
                    raw_details = result.to_dict()
                    # Convert numpy types to Python types
                    for k, v in raw_details.items():
                        if isinstance(v, (np.bool_, np.integer)):
                            details[k] = int(v) if isinstance(v, np.integer) else bool(v)
                        elif isinstance(v, np.floating):
                            details[k] = float(v)
                        elif isinstance(v, np.ndarray):
                            details[k] = v.tolist()[:10]
                        elif isinstance(v, (int, float, str, bool, list, dict, type(None))):
                            details[k] = v
                elif hasattr(result, '__dict__'):
                    for k, v in result.__dict__.items():
                        if isinstance(v, (np.bool_, np.integer)):
                            details[k] = int(v) if isinstance(v, np.integer) else bool(v)
                        elif isinstance(v, np.floating):
                            details[k] = float(v)
                        elif isinstance(v, (int, float, str, bool)):
                            details[k] = v
                        elif isinstance(v, np.ndarray):
                            details[k] = v.tolist()[:10]  # First 10 values

                detector_results[name] = DetectorResult(
                    detector_name=name,
                    detected=detected,
                    confidence=confidence,
                    pattern_name=pattern_name,
                    details=details
                )

            except Exception as e:
                detector_results[name] = DetectorResult(
                    detector_name=name,
                    detected=False,
                    confidence=0.0,
                    pattern_name=None,
                    details={"error": str(e)}
                )

        # Run Reagan shape classifier
        try:
            shape_result = self.reagan_classifier.classify(
                values,
                trajectory_id=work_id,
                title=title
            )

            detector_results["reagan_shapes"] = DetectorResult(
                detector_name="reagan_shapes",
                detected=shape_result.confidence > 0.5,
                confidence=shape_result.confidence,
                pattern_name=shape_result.best_shape_name,
                details={
                    "best_shape": shape_result.best_shape,
                    "shape_scores": shape_result.shape_scores,
                    "notes": shape_result.notes
                }
            )
        except Exception as e:
            detector_results["reagan_shapes"] = DetectorResult(
                detector_name="reagan_shapes",
                detected=False,
                confidence=0.0,
                pattern_name=None,
                details={"error": str(e)}
            )

        # Rank models by confidence
        model_ranking = sorted(
            detector_results.keys(),
            key=lambda k: detector_results[k].confidence,
            reverse=True
        )

        best_fit = model_ranking[0]
        best_confidence = detector_results[best_fit].confidence

        # Calculate cultural alignment
        # Expected: Japanese texts should fit Japanese models better
        #           Western texts should fit Western models better
        cultural_alignment = {}

        for model_name, result in detector_results.items():
            model_origin = self.MODEL_ORIGINS.get(model_name, "unknown")

            if culture == model_origin:
                # Same culture - alignment is confidence
                cultural_alignment[model_name] = result.confidence
            else:
                # Cross-cultural - inverted (high cross-cultural = interesting)
                cultural_alignment[f"{model_name}_cross"] = result.confidence

        return UnifiedAnalysis(
            work_id=work_id,
            title=title,
            author=author,
            culture=culture,
            detector_results=detector_results,
            best_fit_model=best_fit,
            best_fit_confidence=best_confidence,
            model_ranking=model_ranking,
            cultural_alignment=cultural_alignment
        )

    def analyze_corpus(
        self,
        texts: List[Dict],
        culture: str
    ) -> List[UnifiedAnalysis]:
        """
        Analyze all texts in a corpus.

        Args:
            texts: List of dicts with 'content', 'title', 'author', 'id'
            culture: Cultural origin of corpus

        Returns:
            List of UnifiedAnalysis results
        """
        results = []

        for i, text_data in enumerate(texts):
            content = text_data.get("content", "")
            if not content or len(content) < 1000:
                continue

            print(f"  [{i+1}/{len(texts)}] Analyzing: {text_data.get('title', 'Unknown')}")

            try:
                result = self.analyze_text(
                    text=content,
                    work_id=text_data.get("id", f"text_{i}"),
                    title=text_data.get("title", "Unknown"),
                    author=text_data.get("author", "Unknown"),
                    culture=culture
                )
                results.append(result)
            except Exception as e:
                print(f"    Error: {e}")

        return results


def compute_cross_cultural_statistics(
    japanese_results: List[UnifiedAnalysis],
    western_results: List[UnifiedAnalysis]
) -> Dict:
    """
    Compute statistics comparing model fit across cultures.

    Tests hypotheses:
    - H1: Japanese texts fit kishōtenketsu better than Western texts
    - H2: Western texts fit Western models (Aristotle, Freytag, etc.) better
    - H3: Some models are truly universal (similar fit across cultures)
    """
    from scipy.stats import mannwhitneyu

    stats = {
        "n_japanese": len(japanese_results),
        "n_western": len(western_results),
        "model_comparisons": {},
        "best_fit_distribution": {
            "japanese": {},
            "western": {}
        },
        "universality_scores": {}
    }

    # Count best-fit models by culture
    for r in japanese_results:
        model = r.best_fit_model
        stats["best_fit_distribution"]["japanese"][model] = \
            stats["best_fit_distribution"]["japanese"].get(model, 0) + 1

    for r in western_results:
        model = r.best_fit_model
        stats["best_fit_distribution"]["western"][model] = \
            stats["best_fit_distribution"]["western"].get(model, 0) + 1

    # Compare each model across cultures
    model_names = ["kishotenketsu", "aristotle", "freytag",
                   "harmon_circle", "campbell", "reagan_shapes"]

    for model in model_names:
        jp_scores = [r.detector_results[model].confidence
                     for r in japanese_results
                     if model in r.detector_results]

        we_scores = [r.detector_results[model].confidence
                     for r in western_results
                     if model in r.detector_results]

        if len(jp_scores) >= 3 and len(we_scores) >= 3:
            # Mann-Whitney U test
            stat, p_value = mannwhitneyu(jp_scores, we_scores, alternative='two-sided')

            jp_mean = np.mean(jp_scores)
            we_mean = np.mean(we_scores)

            # Effect size (rank-biserial correlation)
            n1, n2 = len(jp_scores), len(we_scores)
            effect_size = 1 - (2 * stat) / (n1 * n2)  # Rank-biserial

            # Determine if universal (no significant difference)
            is_universal = p_value > 0.05

            stats["model_comparisons"][model] = {
                "japanese_mean": float(jp_mean),
                "western_mean": float(we_mean),
                "difference": float(jp_mean - we_mean),
                "mann_whitney_u": float(stat),
                "p_value": float(p_value),
                "effect_size": float(effect_size),
                "significant": p_value < 0.05,
                "higher_in": "japanese" if jp_mean > we_mean else "western"
            }

            # Universality score: lower difference = more universal
            stats["universality_scores"][model] = float(1 - abs(jp_mean - we_mean))

    # Rank models by universality
    stats["universality_ranking"] = sorted(
        stats["universality_scores"].keys(),
        key=lambda k: stats["universality_scores"][k],
        reverse=True
    )

    return stats


def run_unified_analysis():
    """Run unified detector analysis on full corpus."""
    print("=" * 60)
    print("UNIFIED DETECTOR ANALYSIS")
    print("Running ALL 6 detectors on ALL texts")
    print("=" * 60)

    # Initialize
    detector = UnifiedDetector(method="entropy")

    # Load Japanese corpus
    print("\n[1/4] Loading Japanese corpus...")
    japanese_texts = []

    # Check multiple Japanese text locations
    jp_locations = [
        Path("data/raw/aozora_extended/texts"),
        Path("data/raw/aozora"),
        Path("data/raw/aozora_expanded"),
        Path("data/raw/corpus_expanded/japanese"),
    ]

    for jp_dir in jp_locations:
        if not jp_dir.exists():
            continue

        # Try JSON files
        for f in jp_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                content = data.get("content", data.get("text", ""))
                if len(content) > 1000:
                    japanese_texts.append({
                        "id": data.get("id", f.stem),
                        "title": data.get("title", f.stem),
                        "author": data.get("author", "Unknown"),
                        "content": content
                    })
            except:
                pass

        # Try TXT files
        for f in jp_dir.glob("*.txt"):
            try:
                content = f.read_text(encoding='utf-8', errors='ignore')
                if len(content) > 1000:
                    japanese_texts.append({
                        "id": f.stem,
                        "title": f.stem.replace("_", " "),
                        "author": "Unknown",
                        "content": content
                    })
            except:
                pass

    print(f"  Found {len(japanese_texts)} Japanese texts")

    # Load Western corpus
    print("\n[2/4] Loading Western corpus...")
    western_texts = []

    # Check multiple Western text locations
    we_locations = [
        Path("data/raw/gutenberg"),
        Path("data/raw/gutenberg_extended"),
        Path("data/raw/english"),
        Path("data/raw/french"),
        Path("data/raw/german"),
        Path("data/raw/spanish"),
        Path("data/raw/american"),
        Path("data/raw/dostoevsky"),
        Path("data/raw/tolstoy"),
        Path("data/raw/corpus_expanded/western"),
    ]

    for we_dir in we_locations:
        if not we_dir.exists():
            continue

        # Try JSON files
        for f in we_dir.glob("*.json"):
            if f.name == "manifest.json":
                continue
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                content = data.get("content", data.get("text", ""))
                if len(content) > 1000:
                    western_texts.append({
                        "id": data.get("id", f.stem),
                        "title": data.get("title", f.stem),
                        "author": data.get("author", "Unknown"),
                        "content": content
                    })
            except:
                pass

        # Try TXT files
        for f in we_dir.glob("*.txt"):
            try:
                content = f.read_text(encoding='utf-8', errors='ignore')
                if len(content) > 1000:
                    western_texts.append({
                        "id": f.stem,
                        "title": f.stem.replace("_", " "),
                        "author": "Unknown",
                        "content": content
                    })
            except:
                pass

    print(f"  Found {len(western_texts)} Western texts")

    # Analyze Japanese corpus
    print("\n[3/4] Analyzing Japanese corpus with all detectors...")
    japanese_results = detector.analyze_corpus(japanese_texts, culture="japanese")

    # Analyze Western corpus
    print("\n[4/4] Analyzing Western corpus with all detectors...")
    western_results = detector.analyze_corpus(western_texts, culture="western")

    # Compute cross-cultural statistics
    print("\n" + "=" * 60)
    print("COMPUTING CROSS-CULTURAL STATISTICS")
    print("=" * 60)

    stats = compute_cross_cultural_statistics(japanese_results, western_results)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nCorpus sizes: Japanese={stats['n_japanese']}, Western={stats['n_western']}")

    print("\nModel Comparisons (Japanese vs Western mean confidence):")
    print("-" * 50)

    for model, comp in stats["model_comparisons"].items():
        sig = "*" if comp["significant"] else ""
        print(f"  {model:20} JP={comp['japanese_mean']:.3f}  WE={comp['western_mean']:.3f}  "
              f"diff={comp['difference']:+.3f}  p={comp['p_value']:.4f}{sig}")

    print("\nUniversality Ranking (most to least universal):")
    for i, model in enumerate(stats["universality_ranking"], 1):
        score = stats["universality_scores"][model]
        print(f"  {i}. {model}: {score:.3f}")

    print("\nBest-Fit Model Distribution:")
    print("  Japanese texts:")
    for model, count in sorted(stats["best_fit_distribution"]["japanese"].items(),
                               key=lambda x: -x[1]):
        pct = 100 * count / stats["n_japanese"] if stats["n_japanese"] > 0 else 0
        print(f"    {model}: {count} ({pct:.1f}%)")

    print("  Western texts:")
    for model, count in sorted(stats["best_fit_distribution"]["western"].items(),
                               key=lambda x: -x[1]):
        pct = 100 * count / stats["n_western"] if stats["n_western"] > 0 else 0
        print(f"    {model}: {count} ({pct:.1f}%)")

    # Save results
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "japanese_analyses": [r.to_dict() for r in japanese_results],
        "western_analyses": [r.to_dict() for r in western_results],
        "cross_cultural_statistics": stats
    }

    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    output_file = output_dir / "unified_detector_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_unified_analysis()
