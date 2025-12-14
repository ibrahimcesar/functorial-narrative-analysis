"""
Information Complexity Classes (ICC) Detector

A data-driven, falsifiable narrative classification model based on
information-theoretic properties discovered through unsupervised learning.

Classes:
    ICC-1: Low Complexity Rise (Japanese-typical)
    ICC-2: Low Complexity Fall (Japanese-typical)
    ICC-3: High Complexity Oscillation (Western-typical)
    ICC-4: High Complexity Rise (Western-typical)
    ICC-5: High Complexity Fall (Western-typical)
    ICC-0: Unclassified (doesn't fit clear patterns)

This model is designed to be FALSIFIABLE:
- Specific numerical thresholds
- Rejects random noise (especially ICC-1, ICC-2)
- Makes testable cultural predictions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.signal import find_peaks, savgol_filter
from scipy import stats


@dataclass
class ICCResult:
    """Result of ICC classification."""
    trajectory_id: str
    title: str
    icc_class: str  # "ICC-1" through "ICC-5" or "ICC-0"
    class_name: str
    confidence: float
    features: Dict[str, float]
    cultural_prediction: str  # "japanese", "western", "neutral"
    notes: List[str]

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "title": self.title,
            "icc_class": self.icc_class,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "features": {k: float(v) for k, v in self.features.items()},
            "cultural_prediction": self.cultural_prediction,
            "notes": self.notes,
        }


# ICC Class definitions with thresholds
# NEW: High-complexity classes now require structural features to reject noise
#
# Naming Convention:
#   - Classes named by narrative archetype, not technical features
#   - Japanese-typical: Contemplative, subtle information flow
#   - Western-typical: Dramatic, high-tension information dynamics
#
ICC_CLASSES = {
    "ICC-0": {
        "name": "Wandering Mist",
        "full_name": "The Uncharted Path",
        "cultural_prediction": "neutral",
        "archetype": "Polyphonic narrative; multiple interweaving arcs; experimental structure",
        "thresholds": {},  # Catch-all for unclassified
        "description": "Does not fit standard patterns. May indicate: (1) Complex multi-plot structures like Anna Karenina where parallel arcs interfere, (2) Experimental or avant-garde narratives, (3) Genuinely unique structures, or (4) Random/incoherent text. The absence of pattern IS the pattern.",
        "examples": ["Anna Karenina (dual plots)", "Ulysses", "Cloud Atlas", "Experimental fiction", "Multi-POV epics"],
    },
    "ICC-1": {
        "name": "Quiet Ascent",
        "full_name": "The Contemplative Rise",
        "cultural_prediction": "japanese",
        "archetype": "Bildungsroman without drama; kishōtenketsu enlightenment",
        "thresholds": {
            "net_change_min": 0.15,
            "net_change_max": 1.0,
            "peaks_max": 3,
            "volatility_max": 0.07,
        },
        "description": "Gradual, steady rise with minimal turbulence. The narrative builds quietly toward revelation or growth. Typical of kishōtenketsu, contemplative fiction, and philosophical coming-of-age.",
        "examples": ["Soseki's Kokoro", "Kawabata's Snow Country", "Quiet literary fiction"],
    },
    "ICC-2": {
        "name": "Gentle Descent",
        "full_name": "The Elegiac Fall",
        "cultural_prediction": "japanese",
        "archetype": "Mono no aware; beautiful sadness without melodrama",
        "thresholds": {
            "net_change_min": -1.0,
            "net_change_max": -0.15,
            "peaks_max": 4,
            "volatility_max": 0.07,
        },
        "description": "Gradual, controlled decline toward loss or ending. The tragedy unfolds with restraint rather than dramatic reversals. Typical of mono no aware aesthetics.",
        "examples": ["Tale of Genji's later chapters", "Tanizaki's The Makioka Sisters", "Quiet tragedies"],
    },
    "ICC-3": {
        "name": "Eternal Return",
        "full_name": "The Cyclical Journey",
        "cultural_prediction": "western",
        "archetype": "Episodic adventure; the hero returns transformed",
        "thresholds": {
            "net_change_min": -0.2,
            "net_change_max": 0.2,
            "peaks_min": 4,
            "volatility_min": 0.05,
            "symmetry_min": 0.3,
        },
        "description": "Oscillating pattern that returns to origin. Many dramatic peaks but overall equilibrium. The journey is the destination. Typical of picaresque, episodic adventures, and circular narratives.",
        "examples": ["The Odyssey", "Don Quixote", "Adventure serials", "Harmon's Story Circle"],
    },
    "ICC-4": {
        "name": "Triumphant Climb",
        "full_name": "The Dramatic Ascent",
        "cultural_prediction": "western",
        "archetype": "Rags-to-riches with setbacks; the hero overcomes",
        "thresholds": {
            "net_change_min": 0.15,
            "net_change_max": 1.0,
            "peaks_min": 4,
            "volatility_min": 0.05,
            "trend_r2_min": 0.15,
        },
        "description": "Rising trajectory with dramatic complications. Many setbacks but ultimate triumph. The Hollywood success story with tension. Typical of action-adventure, romance, and inspirational fiction.",
        "examples": ["Rocky", "Pride and Prejudice", "The Count of Monte Cristo", "Most Hollywood films"],
    },
    "ICC-5": {
        "name": "Tragic Fall",
        "full_name": "The Dramatic Descent",
        "cultural_prediction": "western",
        "archetype": "Shakespearean tragedy; the mighty are brought low",
        "thresholds": {
            "net_change_min": -1.0,
            "net_change_max": -0.15,
            "peaks_min": 4,
            "volatility_min": 0.05,
            "trend_r2_min": 0.15,
        },
        "description": "Falling trajectory with dramatic reversals. Many moments of hope before final doom. The tragedy is operatic, not quiet. Typical of classical tragedy, crime fiction endings, and cautionary tales.",
        "examples": ["Macbeth", "Breaking Bad", "The Great Gatsby", "Crime and Punishment"],
    },
}


class ICCDetector:
    """
    Information Complexity Classes (ICC) Detector.

    A falsifiable narrative classification system based on data-driven
    pattern discovery rather than pre-existing theoretical models.
    """

    def __init__(
        self,
        n_points: int = 100,
        smooth_window: int = 11,
        peak_prominence: float = 0.08,
    ):
        """
        Initialize ICC detector.

        Args:
            n_points: Points to resample trajectory to
            smooth_window: Savitzky-Golay filter window
            peak_prominence: Minimum peak prominence for detection
        """
        self.n_points = n_points
        self.smooth_window = smooth_window
        self.peak_prominence = peak_prominence
        self.classes = ICC_CLASSES

    def _preprocess(self, trajectory: np.ndarray) -> np.ndarray:
        """Preprocess trajectory: resample, normalize, smooth."""
        # Resample
        x_old = np.linspace(0, 1, len(trajectory))
        x_new = np.linspace(0, 1, self.n_points)
        resampled = np.interp(x_new, x_old, trajectory)

        # Normalize to [0, 1]
        min_val, max_val = resampled.min(), resampled.max()
        if max_val - min_val > 1e-8:
            normalized = (resampled - min_val) / (max_val - min_val)
        else:
            normalized = np.full_like(resampled, 0.5)

        # Smooth
        if len(normalized) > self.smooth_window:
            smoothed = savgol_filter(normalized, self.smooth_window, 3)
        else:
            smoothed = normalized

        return smoothed

    def _extract_features(self, trajectory: np.ndarray) -> Dict[str, float]:
        """Extract ICC-relevant features."""
        t = trajectory

        # Net change
        net_change = t[-1] - t[0]

        # Peak detection
        peaks, properties = find_peaks(
            t,
            distance=len(t)//10,
            prominence=self.peak_prominence
        )
        n_peaks = len(peaks)

        # Valley detection
        valleys, _ = find_peaks(
            -t,
            distance=len(t)//10,
            prominence=self.peak_prominence
        )
        n_valleys = len(valleys)

        # Volatility (std of differences)
        diff = np.diff(t)
        volatility = np.std(diff)

        # Complexity (direction changes)
        sign_changes = np.sum(np.diff(np.sign(diff)) != 0)

        # Peak positions
        if n_peaks > 0:
            first_peak_pos = peaks[0] / len(t)
            last_peak_pos = peaks[-1] / len(t)
            max_peak_pos = peaks[np.argmax(t[peaks])] / len(t)
        else:
            first_peak_pos = 0.5
            last_peak_pos = 0.5
            max_peak_pos = np.argmax(t) / len(t)

        # Range and variance
        value_range = t.max() - t.min()
        variance = np.var(t)

        # === NEW FEATURES TO DISTINGUISH STRUCTURED COMPLEXITY FROM NOISE ===

        # 1. Peak regularity: structured narratives have more regular peak spacing
        # Random noise has irregular spacing; intentional structure has rhythm
        if n_peaks >= 2:
            peak_intervals = np.diff(peaks)
            peak_regularity = 1.0 - (np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-8))
            peak_regularity = max(0, min(1, peak_regularity))
        else:
            peak_regularity = 0.0

        # 2. Autocorrelation at lag 1: structured data has temporal dependence
        # Random noise has ~0 autocorrelation; narratives have continuity
        if len(t) > 1:
            autocorr = np.corrcoef(t[:-1], t[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0

        # 3. Trend strength: fit linear regression and measure R²
        # Structured narratives follow discernible trends
        x = np.arange(len(t))
        slope, intercept = np.polyfit(x, t, 1)
        trend_line = slope * x + intercept
        ss_res = np.sum((t - trend_line) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        trend_r2 = 1 - (ss_res / (ss_tot + 1e-8)) if ss_tot > 0 else 0
        trend_r2 = max(0, trend_r2)

        # 4. Segment coherence: divide into segments, check if segments are internally smooth
        # Narratives have coherent "acts"; noise is uniformly chaotic
        n_segments = 4
        segment_size = len(t) // n_segments
        segment_volatilities = []
        for i in range(n_segments):
            seg = t[i * segment_size:(i + 1) * segment_size]
            if len(seg) > 1:
                seg_vol = np.std(np.diff(seg))
                segment_volatilities.append(seg_vol)
        if segment_volatilities:
            # Coherence = segments are smoother than overall
            segment_coherence = 1.0 - (np.mean(segment_volatilities) / (volatility + 1e-8))
            segment_coherence = max(0, min(1, segment_coherence))
        else:
            segment_coherence = 0.0

        # 5. Symmetry score: many narrative structures have rough symmetry
        # (beginning mirrors end, or central peak)
        mid = len(t) // 2
        first_half = t[:mid]
        second_half = t[mid:2*mid]  # Same length
        if len(first_half) == len(second_half) and len(first_half) > 0:
            # Check if second half mirrors first (inverted correlation = symmetry)
            mirror_corr = np.corrcoef(first_half, second_half[::-1])[0, 1]
            if np.isnan(mirror_corr):
                mirror_corr = 0.0
            symmetry = abs(mirror_corr)  # Both positive and negative correlation = structure
        else:
            symmetry = 0.0

        # 6. Structure score: combined measure of non-randomness
        # High value = definitely NOT random noise
        structure_score = (
            0.25 * peak_regularity +
            0.25 * abs(autocorr) +
            0.25 * trend_r2 +
            0.25 * segment_coherence
        )

        return {
            "net_change": float(net_change),
            "n_peaks": n_peaks,
            "n_valleys": n_valleys,
            "volatility": float(volatility),
            "complexity": sign_changes,
            "first_peak_pos": float(first_peak_pos),
            "last_peak_pos": float(last_peak_pos),
            "max_peak_pos": float(max_peak_pos),
            "value_range": float(value_range),
            "variance": float(variance),
            # New structural features
            "peak_regularity": float(peak_regularity),
            "autocorrelation": float(autocorr),
            "trend_r2": float(trend_r2),
            "segment_coherence": float(segment_coherence),
            "symmetry": float(symmetry),
            "structure_score": float(structure_score),
        }

    def _match_class(
        self,
        features: Dict[str, float]
    ) -> Tuple[str, float, List[str]]:
        """
        Match features to ICC class.

        Returns:
            (class_id, confidence, notes)
        """
        net_change = features["net_change"]
        n_peaks = features["n_peaks"]
        volatility = features["volatility"]
        structure_score = features.get("structure_score", 0)
        autocorrelation = features.get("autocorrelation", 0)
        peak_regularity = features.get("peak_regularity", 0)
        trend_r2 = features.get("trend_r2", 0)
        symmetry = features.get("symmetry", 0)

        best_class = "ICC-0"
        best_score = 0.0
        notes = []

        for class_id, class_def in self.classes.items():
            # Skip ICC-0 in matching loop (it's the fallback)
            if class_id == "ICC-0":
                continue

            thresholds = class_def["thresholds"]
            score = 0.0
            matches = 0
            total = 0

            # Check net_change range
            if "net_change_min" in thresholds and "net_change_max" in thresholds:
                total += 1
                if thresholds["net_change_min"] <= net_change <= thresholds["net_change_max"]:
                    matches += 1
                    score += 0.4

            # Check peaks
            if "peaks_max" in thresholds:
                total += 1
                if n_peaks <= thresholds["peaks_max"]:
                    matches += 1
                    score += 0.3
            if "peaks_min" in thresholds:
                total += 1
                if n_peaks >= thresholds["peaks_min"]:
                    matches += 1
                    score += 0.3

            # Check volatility
            if "volatility_max" in thresholds:
                total += 1
                if volatility <= thresholds["volatility_max"]:
                    matches += 1
                    score += 0.3
            if "volatility_min" in thresholds:
                total += 1
                if volatility >= thresholds["volatility_min"]:
                    matches += 1
                    score += 0.3

            # Check structure score (for ICC-3/4/5 to reject noise)
            if "structure_score_min" in thresholds:
                total += 1
                if structure_score >= thresholds["structure_score_min"]:
                    matches += 1
                    score += 0.3

            # Check autocorrelation (temporal coherence)
            if "autocorrelation_min" in thresholds:
                total += 1
                if abs(autocorrelation) >= thresholds["autocorrelation_min"]:
                    matches += 1
                    score += 0.3

            # Check peak regularity (regular spacing = intentional structure)
            if "peak_regularity_min" in thresholds:
                total += 1
                if peak_regularity >= thresholds["peak_regularity_min"]:
                    matches += 1
                    score += 0.3

            # Check trend R² (trajectory follows discernible trend)
            if "trend_r2_min" in thresholds:
                total += 1
                if trend_r2 >= thresholds["trend_r2_min"]:
                    matches += 1
                    score += 0.3

            # Check symmetry (beginning mirrors end)
            if "symmetry_min" in thresholds:
                total += 1
                if symmetry >= thresholds["symmetry_min"]:
                    matches += 1
                    score += 0.3

            # Normalize score
            if total > 0:
                score = score / (total * 0.33)  # Max possible per check
                score = min(1.0, score)

            if matches == total and score > best_score:
                best_score = score
                best_class = class_id

        # Generate notes
        if net_change > 0.15:
            notes.append("Rising overall arc")
        elif net_change < -0.15:
            notes.append("Falling overall arc")
        else:
            notes.append("Oscillating/stable arc")

        if n_peaks <= 3:
            notes.append(f"Low complexity ({n_peaks} peaks)")
        else:
            notes.append(f"High complexity ({n_peaks} peaks)")

        if volatility < 0.05:
            notes.append("Smooth transitions")
        elif volatility > 0.08:
            notes.append("Turbulent/volatile")

        if best_class == "ICC-0":
            notes.append("Unique structure - does not fit standard patterns")
            notes.append("May indicate polyphonic narrative, experimental structure, or interfering plotlines")

        return best_class, best_score, notes

    def detect(
        self,
        trajectory: np.ndarray,
        trajectory_id: str = "unknown",
        title: str = "Unknown"
    ) -> ICCResult:
        """
        Classify trajectory into ICC class.

        Args:
            trajectory: Surprisal/entropy trajectory
            trajectory_id: Identifier
            title: Title of work

        Returns:
            ICCResult with classification
        """
        # Preprocess
        processed = self._preprocess(trajectory)

        # Extract features
        features = self._extract_features(processed)

        # Match to class
        icc_class, confidence, notes = self._match_class(features)

        # Get class info
        if icc_class in self.classes:
            class_info = self.classes[icc_class]
            class_name = class_info["name"]
            cultural_prediction = class_info["cultural_prediction"]
            # Add archetype and examples to notes
            if "archetype" in class_info:
                notes.append(f"Archetype: {class_info['archetype']}")
            if "examples" in class_info:
                notes.append(f"Similar to: {', '.join(class_info['examples'][:2])}")
        else:
            class_name = "Unclassified"
            cultural_prediction = "neutral"

        return ICCResult(
            trajectory_id=trajectory_id,
            title=title,
            icc_class=icc_class,
            class_name=class_name,
            confidence=confidence,
            features=features,
            cultural_prediction=cultural_prediction,
            notes=notes,
        )

    def classify_batch(
        self,
        trajectories: List[np.ndarray],
        trajectory_ids: Optional[List[str]] = None,
        titles: Optional[List[str]] = None
    ) -> List[ICCResult]:
        """Classify multiple trajectories."""
        results = []

        for i, traj in enumerate(trajectories):
            tid = trajectory_ids[i] if trajectory_ids else f"traj_{i}"
            title = titles[i] if titles else f"Trajectory {i}"

            result = self.detect(traj, tid, title)
            results.append(result)

        return results

    def get_class_statistics(
        self,
        results: List[ICCResult],
        cultures: Optional[List[str]] = None
    ) -> Dict:
        """
        Compute statistics for classified trajectories.

        Args:
            results: List of ICCResult
            cultures: Optional list of actual cultures

        Returns:
            Statistics dict
        """
        stats = {
            "total": len(results),
            "by_class": {},
            "cultural_accuracy": None,
        }

        # Count by class
        for r in results:
            cls = r.icc_class
            if cls not in stats["by_class"]:
                stats["by_class"][cls] = {"count": 0, "avg_confidence": 0}
            stats["by_class"][cls]["count"] += 1

        # Average confidence per class
        for cls in stats["by_class"]:
            class_results = [r for r in results if r.icc_class == cls]
            stats["by_class"][cls]["avg_confidence"] = (
                np.mean([r.confidence for r in class_results])
            )

        # Cultural prediction accuracy
        if cultures:
            correct = 0
            total = 0
            for r, actual in zip(results, cultures):
                if r.cultural_prediction == "neutral":
                    continue
                total += 1
                if r.cultural_prediction == actual:
                    correct += 1

            if total > 0:
                stats["cultural_accuracy"] = correct / total

        return stats


def test_icc_falsifiability():
    """
    Test ICC model falsifiability by comparing to random trajectories.
    """
    detector = ICCDetector()

    print("=" * 60)
    print("ICC MODEL FALSIFIABILITY TEST (v2 - with structure requirements)")
    print("=" * 60)

    # Generate random trajectories
    n_random = 200
    random_results = []

    print("\nTesting on random trajectories...")
    for _ in range(n_random):
        # Smooth random trajectory (simulates noise passing through same preprocessing)
        raw = np.random.uniform(0, 1, 100)
        smoothed = savgol_filter(raw, 11, 3)
        result = detector.detect(smoothed)
        random_results.append(result)

    # Count classifications
    random_by_class = {}
    for r in random_results:
        random_by_class[r.icc_class] = random_by_class.get(r.icc_class, 0) + 1

    print("\nRandom trajectory classification:")
    for cls, count in sorted(random_by_class.items()):
        pct = 100 * count / n_random
        print(f"  {cls}: {count} ({pct:.1f}%)")

    # Calculate rejection rates for each class type
    icc1_rate = random_by_class.get("ICC-1", 0) / n_random
    icc2_rate = random_by_class.get("ICC-2", 0) / n_random
    icc3_rate = random_by_class.get("ICC-3", 0) / n_random
    icc4_rate = random_by_class.get("ICC-4", 0) / n_random
    icc5_rate = random_by_class.get("ICC-5", 0) / n_random
    icc0_rate = random_by_class.get("ICC-0", 0) / n_random

    print(f"\nClass-by-class random acceptance rates:")
    print(f"  ICC-0 (Unclassified):              {icc0_rate:.1%}")
    print(f"  ICC-1 (Low Complexity Rise):       {icc1_rate:.1%}  <- should be <5%")
    print(f"  ICC-2 (Low Complexity Fall):       {icc2_rate:.1%}  <- should be <5%")
    print(f"  ICC-3 (High Complexity Oscillation): {icc3_rate:.1%}")
    print(f"  ICC-4 (High Complexity Rise):      {icc4_rate:.1%}")
    print(f"  ICC-5 (High Complexity Fall):      {icc5_rate:.1%}")

    low_complexity_rate = icc1_rate + icc2_rate
    high_complexity_rate = icc3_rate + icc4_rate + icc5_rate

    print(f"\nSummary:")
    print(f"  Low-complexity (ICC-1/2) acceptance:  {low_complexity_rate:.1%}")
    print(f"  High-complexity (ICC-3/4/5) acceptance: {high_complexity_rate:.1%}")
    print(f"  Unclassified (ICC-0):                 {icc0_rate:.1%}")

    # Tests
    tests_passed = 0
    tests_total = 2

    if low_complexity_rate < 0.05:
        print("\n✓ PASS: Low-complexity classes reject >95% of random")
        tests_passed += 1
    else:
        print("\n✗ FAIL: Low-complexity classes accept too much random")

    if high_complexity_rate < 0.70:
        print("✓ PASS: High-complexity classes reject >30% of random")
        tests_passed += 1
    else:
        print("✗ FAIL: High-complexity classes accept too much random (need stricter structure requirements)")

    print(f"\nFalsifiability score: {tests_passed}/{tests_total} tests passed")


if __name__ == "__main__":
    test_icc_falsifiability()
