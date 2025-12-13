"""
Reagan Shape Classifier

Classifies narrative trajectories into the six fundamental story shapes
identified by Reagan et al. (2016) "The emotional arcs of stories are
dominated by six basic shapes."

The Six Shapes:
    1. RAGS TO RICHES (Rise)
       Continuous rise in sentiment throughout
       Shape: /

    2. RICHES TO RAGS (Fall / Tragedy)
       Continuous fall in sentiment
       Shape: \

    3. MAN IN A HOLE (Fall-Rise)
       Descent followed by recovery
       Shape: \_/

    4. ICARUS (Rise-Fall)
       Rise followed by fall
       Shape: /\

    5. CINDERELLA (Rise-Fall-Rise)
       Rise, setback, then triumph
       Shape: /\_/

    6. OEDIPUS (Fall-Rise-Fall)
       Fall, brief recovery, then final fall
       Shape: \_/\

Reference:
Reagan, A. J., Mitchell, L., Kiley, D., Danforth, C. M., & Dodds, P. S. (2016).
The emotional arcs of stories are dominated by six basic shapes.
EPJ Data Science, 5(1), 31.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr


@dataclass
class ReaganShape:
    """Represents one of Reagan's six story shapes."""
    name: str
    short_name: str
    pattern: str  # Visual representation
    description: str
    canonical_func: callable  # Function generating canonical shape


# Define the six canonical shapes
def _rags_to_riches(x):
    """Continuous rise."""
    return x

def _riches_to_rags(x):
    """Continuous fall."""
    return 1 - x

def _man_in_hole(x):
    """Fall then rise (U-shape)."""
    return 4 * (x - 0.5) ** 2

def _icarus(x):
    """Rise then fall (inverted U)."""
    return 1 - 4 * (x - 0.5) ** 2

def _cinderella(x):
    """Rise-fall-rise (N-shape or double valley)."""
    return 0.5 + 0.5 * np.sin(2 * np.pi * x - np.pi/2)

def _oedipus(x):
    """Fall-rise-fall (inverted N)."""
    return 0.5 - 0.5 * np.sin(2 * np.pi * x - np.pi/2)


REAGAN_SHAPES = {
    "rags_to_riches": ReaganShape(
        name="Rags to Riches",
        short_name="Rise",
        pattern="/",
        description="Continuous rise in fortune",
        canonical_func=_rags_to_riches
    ),
    "riches_to_rags": ReaganShape(
        name="Riches to Rags",
        short_name="Fall",
        pattern="\\",
        description="Continuous fall (tragedy)",
        canonical_func=_riches_to_rags
    ),
    "man_in_hole": ReaganShape(
        name="Man in a Hole",
        short_name="Fall-Rise",
        pattern="\\_/",
        description="Fall into trouble, then recovery",
        canonical_func=_man_in_hole
    ),
    "icarus": ReaganShape(
        name="Icarus",
        short_name="Rise-Fall",
        pattern="/\\",
        description="Rise to success, then fall",
        canonical_func=_icarus
    ),
    "cinderella": ReaganShape(
        name="Cinderella",
        short_name="Rise-Fall-Rise",
        pattern="/\\_/",
        description="Rise, setback, ultimate triumph",
        canonical_func=_cinderella
    ),
    "oedipus": ReaganShape(
        name="Oedipus",
        short_name="Fall-Rise-Fall",
        pattern="\\_/\\",
        description="Fall, brief hope, final tragedy",
        canonical_func=_oedipus
    ),
}


@dataclass
class ShapeClassification:
    """Result of classifying a trajectory into Reagan shapes."""
    trajectory_id: str
    title: str
    best_shape: str
    best_shape_name: str
    confidence: float  # Correlation with best match
    shape_scores: Dict[str, float]  # Correlation with each shape
    shape_ranking: List[str]  # Shapes ranked by fit
    trajectory_features: Dict[str, float]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "title": self.title,
            "best_shape": self.best_shape,
            "best_shape_name": self.best_shape_name,
            "confidence": self.confidence,
            "shape_scores": self.shape_scores,
            "shape_ranking": self.shape_ranking,
            "trajectory_features": self.trajectory_features,
            "notes": self.notes,
        }


class ReaganClassifier:
    """
    Classifies narrative trajectories into Reagan et al.'s six shapes.

    Uses correlation-based matching against canonical shape templates.
    """

    def __init__(self, smooth_sigma: float = 3.0, n_points: int = 100):
        """
        Initialize classifier.

        Args:
            smooth_sigma: Gaussian smoothing for preprocessing
            n_points: Number of points for resampling
        """
        self.smooth_sigma = smooth_sigma
        self.n_points = n_points
        self.shapes = REAGAN_SHAPES

        # Pre-compute canonical patterns
        self.x = np.linspace(0, 1, n_points)
        self.canonical_patterns = {
            name: shape.canonical_func(self.x)
            for name, shape in self.shapes.items()
        }

    def _preprocess_trajectory(self, values: np.ndarray) -> np.ndarray:
        """Smooth, normalize, and resample trajectory."""
        # Smooth
        smoothed = gaussian_filter1d(values, sigma=self.smooth_sigma)

        # Normalize to [0, 1]
        min_val, max_val = smoothed.min(), smoothed.max()
        if max_val - min_val > 1e-8:
            normalized = (smoothed - min_val) / (max_val - min_val)
        else:
            normalized = np.full_like(smoothed, 0.5)

        # Resample to fixed length
        x_orig = np.linspace(0, 1, len(normalized))
        resampled = np.interp(self.x, x_orig, normalized)

        return resampled

    def _extract_features(self, trajectory: np.ndarray) -> Dict[str, float]:
        """Extract trajectory features for analysis."""
        # Basic statistics
        start_val = trajectory[0]
        end_val = trajectory[-1]
        mean_val = np.mean(trajectory)

        # Trend
        slope, _ = np.polyfit(self.x, trajectory, 1)

        # Find peaks and valleys
        peaks, _ = find_peaks(trajectory, distance=len(trajectory)//10)
        valleys, _ = find_peaks(-trajectory, distance=len(trajectory)//10)

        n_peaks = len(peaks)
        n_valleys = len(valleys)

        # Position of global extrema
        global_max_pos = np.argmax(trajectory) / len(trajectory)
        global_min_pos = np.argmin(trajectory) / len(trajectory)

        # Change from start to end
        net_change = end_val - start_val

        # Volatility
        volatility = np.std(np.diff(trajectory))

        return {
            "start_value": float(start_val),
            "end_value": float(end_val),
            "mean_value": float(mean_val),
            "slope": float(slope),
            "n_peaks": n_peaks,
            "n_valleys": n_valleys,
            "global_max_position": float(global_max_pos),
            "global_min_position": float(global_min_pos),
            "net_change": float(net_change),
            "volatility": float(volatility),
        }

    def _compute_shape_scores(self, trajectory: np.ndarray) -> Dict[str, float]:
        """Compute correlation with each canonical shape."""
        scores = {}

        for name, pattern in self.canonical_patterns.items():
            corr, _ = pearsonr(trajectory, pattern)
            if np.isnan(corr):
                corr = 0.0
            scores[name] = float(corr)

        return scores

    def classify(
        self,
        trajectory: np.ndarray,
        trajectory_id: str = "unknown",
        title: str = "Unknown"
    ) -> ShapeClassification:
        """
        Classify a trajectory into one of Reagan's six shapes.

        Args:
            trajectory: Array of sentiment values
            trajectory_id: Identifier
            title: Title of the work

        Returns:
            ShapeClassification with results
        """
        # Preprocess
        processed = self._preprocess_trajectory(trajectory)

        # Extract features
        features = self._extract_features(processed)

        # Compute shape scores
        shape_scores = self._compute_shape_scores(processed)

        # Rank shapes by correlation
        ranking = sorted(shape_scores.keys(), key=lambda k: shape_scores[k], reverse=True)

        # Best match
        best_shape = ranking[0]
        best_score = shape_scores[best_shape]
        best_shape_name = self.shapes[best_shape].name

        # Generate notes
        notes = []

        # Confidence assessment
        if best_score > 0.8:
            notes.append(f"Strong {best_shape_name} pattern")
        elif best_score > 0.5:
            notes.append(f"Moderate {best_shape_name} pattern")
        else:
            notes.append(f"Weak shape match (best: {best_shape_name})")

        # Check for ambiguity
        second_best = ranking[1]
        second_score = shape_scores[second_best]
        if best_score - second_score < 0.1:
            notes.append(f"Ambiguous: also resembles {self.shapes[second_best].name}")

        # Feature-based notes
        if features["net_change"] > 0.3:
            notes.append("Overall positive arc")
        elif features["net_change"] < -0.3:
            notes.append("Overall negative arc")

        if features["n_peaks"] >= 2:
            notes.append(f"Multiple peaks ({features['n_peaks']})")

        return ShapeClassification(
            trajectory_id=trajectory_id,
            title=title,
            best_shape=best_shape,
            best_shape_name=best_shape_name,
            confidence=best_score,
            shape_scores=shape_scores,
            shape_ranking=ranking,
            trajectory_features=features,
            notes=notes,
        )

    def classify_batch(
        self,
        trajectories: List[Dict]
    ) -> List[ShapeClassification]:
        """Classify multiple trajectories."""
        results = []
        for traj in trajectories:
            values = np.array(traj.get("values", traj.get("trajectory", [])))
            traj_id = traj.get("id", traj.get("trajectory_id", "unknown"))
            title = traj.get("title", "Unknown")

            result = self.classify(values, traj_id, title)
            results.append(result)

        return results


def correlate_harmon_shapes(
    harmon_scores: Dict[str, float],
    shape_classifications: Dict[str, str],
    shape_scores: Dict[str, Dict[str, float]]
) -> Dict:
    """
    Analyze correlation between Harmon Circle scores and Reagan shapes.

    Tests the hypothesis: High Harmon score → Man-in-a-Hole shape

    Args:
        harmon_scores: {trajectory_id: harmon_conformance_score}
        shape_classifications: {trajectory_id: best_shape}
        shape_scores: {trajectory_id: {shape: score}}

    Returns:
        Correlation analysis results
    """
    # Group by shape
    shape_groups = {}
    for traj_id, shape in shape_classifications.items():
        if shape not in shape_groups:
            shape_groups[shape] = []
        shape_groups[shape].append(harmon_scores.get(traj_id, 0))

    # Mean Harmon score per shape
    mean_harmon_by_shape = {
        shape: np.mean(scores) if scores else 0
        for shape, scores in shape_groups.items()
    }

    # Correlation: Harmon score vs Man-in-Hole score
    harmon_list = []
    mih_list = []
    for traj_id in harmon_scores:
        if traj_id in shape_scores:
            harmon_list.append(harmon_scores[traj_id])
            mih_list.append(shape_scores[traj_id].get("man_in_hole", 0))

    if len(harmon_list) >= 3:
        harmon_mih_corr, harmon_mih_p = pearsonr(harmon_list, mih_list)
    else:
        harmon_mih_corr, harmon_mih_p = 0, 1

    return {
        "mean_harmon_by_shape": mean_harmon_by_shape,
        "harmon_mih_correlation": float(harmon_mih_corr),
        "harmon_mih_p_value": float(harmon_mih_p),
        "n_samples": len(harmon_list),
        "shape_distribution": {s: len(g) for s, g in shape_groups.items()},
    }
