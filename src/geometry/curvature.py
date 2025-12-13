"""
Narrative Curvature and Geometric Features.

Computes differential geometric properties of narrative trajectories:
    - Curvature: Rate of change of direction (surprise acceleration)
    - Fisher Information: Local sensitivity of belief distribution
    - Arc Length: Total information traversed
    - Geodesic Deviation: How far from "straight" narrative paths

These features characterize the "shape" of stories in information space.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid


@dataclass
class GeometricFeatures:
    """
    Collection of geometric features for a narrative.

    These features can be used for:
        - Genre classification
        - Story shape characterization
        - Cross-cultural comparison
        - Quality/engagement prediction
    """
    # Basic statistics
    mean_surprisal: float
    std_surprisal: float
    max_surprisal: float
    min_surprisal: float

    # Curvature features
    mean_curvature: float
    max_curvature: float
    curvature_variance: float
    total_curvature: float  # Integral of absolute curvature

    # Arc length (total information)
    arc_length: float

    # Derivative features (information rate)
    mean_derivative: float
    max_derivative: float
    derivative_variance: float

    # Peak analysis
    n_peaks: int
    peak_positions: List[float]
    peak_heights: List[float]

    # Shape descriptors
    skewness: float  # Asymmetry of surprisal distribution
    kurtosis: float  # Tail heaviness

    # Entropy dynamics
    entropy_initial: float
    entropy_final: float
    entropy_change: float  # Resolution pattern

    # Additional metadata
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "mean_surprisal": self.mean_surprisal,
            "std_surprisal": self.std_surprisal,
            "max_surprisal": self.max_surprisal,
            "min_surprisal": self.min_surprisal,
            "mean_curvature": self.mean_curvature,
            "max_curvature": self.max_curvature,
            "curvature_variance": self.curvature_variance,
            "total_curvature": self.total_curvature,
            "arc_length": self.arc_length,
            "mean_derivative": self.mean_derivative,
            "max_derivative": self.max_derivative,
            "derivative_variance": self.derivative_variance,
            "n_peaks": self.n_peaks,
            "peak_positions": self.peak_positions,
            "peak_heights": self.peak_heights,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "entropy_initial": self.entropy_initial,
            "entropy_final": self.entropy_final,
            "entropy_change": self.entropy_change,
            "metadata": self.metadata,
        }

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML."""
        return np.array([
            self.mean_surprisal,
            self.std_surprisal,
            self.max_surprisal,
            self.min_surprisal,
            self.mean_curvature,
            self.max_curvature,
            self.curvature_variance,
            self.total_curvature,
            self.arc_length,
            self.mean_derivative,
            self.max_derivative,
            self.derivative_variance,
            self.n_peaks,
            self.skewness,
            self.kurtosis,
            self.entropy_initial,
            self.entropy_final,
            self.entropy_change,
        ])

    @staticmethod
    def feature_names() -> List[str]:
        """Get names of features in vector form."""
        return [
            "mean_surprisal",
            "std_surprisal",
            "max_surprisal",
            "min_surprisal",
            "mean_curvature",
            "max_curvature",
            "curvature_variance",
            "total_curvature",
            "arc_length",
            "mean_derivative",
            "max_derivative",
            "derivative_variance",
            "n_peaks",
            "skewness",
            "kurtosis",
            "entropy_initial",
            "entropy_final",
            "entropy_change",
        ]


class NarrativeCurvature:
    """
    Compute curvature and geometric properties of narrative trajectories.

    In information geometry, curvature measures how rapidly the
    probability distribution is changing - narratively, this corresponds
    to how quickly reader beliefs are being updated.

    High curvature = dramatic reveals, plot twists
    Low curvature = steady development, predictable progression
    """

    def __init__(
        self,
        smooth_sigma: float = 2.0,
        derivative_order: int = 2,
    ):
        """
        Initialize curvature analyzer.

        Args:
            smooth_sigma: Gaussian smoothing parameter
            derivative_order: Order for Savitzky-Golay filter
        """
        self.smooth_sigma = smooth_sigma
        self.derivative_order = derivative_order

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing."""
        if len(values) < 5:
            return values
        return gaussian_filter1d(values, sigma=self.smooth_sigma)

    def _compute_derivatives(
        self,
        values: np.ndarray,
        positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute first and second derivatives."""
        if len(values) < 5:
            return np.zeros_like(values), np.zeros_like(values)

        # Use finite differences
        dt = np.diff(positions)
        dt = np.where(dt > 0, dt, 1e-6)  # Avoid division by zero

        # First derivative (velocity)
        dv = np.diff(values)
        first_derivative = np.zeros_like(values)
        first_derivative[:-1] = dv / dt
        first_derivative[-1] = first_derivative[-2]  # Pad

        # Second derivative (acceleration)
        dv2 = np.diff(first_derivative)
        second_derivative = np.zeros_like(values)
        second_derivative[:-1] = dv2 / dt
        second_derivative[-1] = second_derivative[-2]  # Pad

        return first_derivative, second_derivative

    def compute_curvature(
        self,
        values: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """
        Compute curvature at each point.

        For a 1D trajectory, curvature κ = |y''| / (1 + y'^2)^(3/2)

        This measures how sharply the narrative is "turning" in
        information space.
        """
        smoothed = self._smooth(values)
        first_deriv, second_deriv = self._compute_derivatives(smoothed, positions)

        # Curvature formula
        numerator = np.abs(second_deriv)
        denominator = np.power(1 + first_deriv ** 2, 1.5)
        denominator = np.where(denominator > 0, denominator, 1e-6)

        curvature = numerator / denominator
        return curvature

    def compute_arc_length(
        self,
        values: np.ndarray,
        positions: np.ndarray,
    ) -> float:
        """
        Compute arc length of trajectory.

        This measures total "information distance" traversed,
        independent of direction.
        """
        smoothed = self._smooth(values)
        first_deriv, _ = self._compute_derivatives(smoothed, positions)

        # Arc length element: ds = sqrt(1 + (dy/dx)^2) dx
        integrand = np.sqrt(1 + first_deriv ** 2)

        # Integrate
        arc_length = trapezoid(integrand, positions)
        return float(arc_length)

    def compute_fisher_information(
        self,
        values: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate Fisher information at each point.

        In information geometry, Fisher information measures
        how sensitive the distribution is to parameter changes.

        For surprisal trajectories, we approximate this as the
        squared first derivative - high Fisher information means
        small narrative changes produce large belief updates.
        """
        smoothed = self._smooth(values)
        first_deriv, _ = self._compute_derivatives(smoothed, positions)

        # Fisher information approximation
        fisher = first_deriv ** 2
        return fisher

    def find_peaks(
        self,
        values: np.ndarray,
        positions: np.ndarray,
        prominence: float = 0.5,
    ) -> Tuple[List[float], List[float]]:
        """
        Find peaks (local maxima) in trajectory.

        Peaks correspond to narrative climaxes, reveals, or
        moments of maximum information.
        """
        from scipy.signal import find_peaks as scipy_find_peaks

        smoothed = self._smooth(values)

        # Find peaks with minimum prominence
        peak_indices, properties = scipy_find_peaks(
            smoothed,
            prominence=prominence * np.std(smoothed),
        )

        peak_positions = [float(positions[i]) for i in peak_indices]
        peak_heights = [float(smoothed[i]) for i in peak_indices]

        return peak_positions, peak_heights

    def compute_shape_moments(
        self,
        values: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute skewness and kurtosis of value distribution."""
        from scipy.stats import skew, kurtosis

        if len(values) < 4:
            return 0.0, 0.0

        return float(skew(values)), float(kurtosis(values))

    def extract_features(
        self,
        values: np.ndarray,
        positions: Optional[np.ndarray] = None,
    ) -> GeometricFeatures:
        """
        Extract all geometric features from a trajectory.

        Args:
            values: Surprisal or other trajectory values
            positions: Normalized positions (0-1), auto-generated if None

        Returns:
            GeometricFeatures object with all computed features
        """
        if positions is None:
            positions = np.linspace(0, 1, len(values))

        # Basic statistics
        mean_surprisal = float(np.mean(values))
        std_surprisal = float(np.std(values))
        max_surprisal = float(np.max(values))
        min_surprisal = float(np.min(values))

        # Curvature
        curvature = self.compute_curvature(values, positions)
        mean_curvature = float(np.mean(curvature))
        max_curvature = float(np.max(curvature))
        curvature_variance = float(np.var(curvature))
        total_curvature = float(trapezoid(np.abs(curvature), positions))

        # Arc length
        arc_length = self.compute_arc_length(values, positions)

        # Derivatives
        first_deriv, _ = self._compute_derivatives(values, positions)
        mean_derivative = float(np.mean(first_deriv))
        max_derivative = float(np.max(np.abs(first_deriv)))
        derivative_variance = float(np.var(first_deriv))

        # Peaks
        peak_positions, peak_heights = self.find_peaks(values, positions)
        n_peaks = len(peak_positions)

        # Shape moments
        skewness, kurt = self.compute_shape_moments(values)

        # Entropy dynamics (using first and last 10%)
        n = len(values)
        window = max(1, n // 10)
        entropy_initial = float(np.mean(values[:window]))
        entropy_final = float(np.mean(values[-window:]))
        entropy_change = entropy_final - entropy_initial

        return GeometricFeatures(
            mean_surprisal=mean_surprisal,
            std_surprisal=std_surprisal,
            max_surprisal=max_surprisal,
            min_surprisal=min_surprisal,
            mean_curvature=mean_curvature,
            max_curvature=max_curvature,
            curvature_variance=curvature_variance,
            total_curvature=total_curvature,
            arc_length=arc_length,
            mean_derivative=mean_derivative,
            max_derivative=max_derivative,
            derivative_variance=derivative_variance,
            n_peaks=n_peaks,
            peak_positions=peak_positions,
            peak_heights=peak_heights,
            skewness=skewness,
            kurtosis=kurt,
            entropy_initial=entropy_initial,
            entropy_final=entropy_final,
            entropy_change=entropy_change,
        )


def classify_information_shape(features: GeometricFeatures) -> Dict[str, float]:
    """
    Classify narrative into information-geometric shape categories.

    Returns scores for different shape archetypes based on
    the geometric features.

    Shape categories (extending Reagan's 6 with geometric interpretation):
        - geodesic_tragedy: Low curvature descent (Greek tragedy)
        - high_curvature_mystery: High Fisher info, localized spikes
        - random_walk_comedy: High variance, oscillating
        - compression_progress: Steady entropy reduction
        - discontinuous_twist: Large curvature spike late

    Thresholds calibrated from empirical corpus analysis (n=125 works):
        mean_curvature: p25=52, p50=64, p75=72
        max_curvature: p25=699, p50=988, p75=1337
        n_peaks: p25=7, p50=11, p75=13
        entropy_change: p25=-0.08, p50=0.0, p75=0.10
        std_surprisal/mean_surprisal (CV): typically 0.1-0.3
    """
    scores = {}

    # Empirical percentile thresholds
    CURVATURE_LOW = 52.0   # p25
    CURVATURE_MED = 64.0   # p50
    CURVATURE_HIGH = 72.0  # p75
    MAX_CURV_HIGH = 1000.0 # ~p50
    PEAKS_MED = 11         # p50
    PEAKS_HIGH = 13        # p75

    # Coefficient of variation (std/mean)
    cv = features.std_surprisal / features.mean_surprisal if features.mean_surprisal > 0 else 0

    # === Geodesic Tragedy ===
    # Smooth descent: low curvature + entropy decreasing + negative skew
    # Greek tragedy = inevitable fall, minimal surprise
    curvature_smooth = max(0, 1.0 - (features.mean_curvature - CURVATURE_LOW) / (CURVATURE_HIGH - CURVATURE_LOW))
    entropy_descent = 1.0 if features.entropy_change < -0.05 else 0.3 if features.entropy_change < 0 else 0.1
    negative_arc = 1.0 if features.skewness < -0.5 else 0.5 if features.skewness < 0 else 0.2
    scores["geodesic_tragedy"] = curvature_smooth * entropy_descent * negative_arc

    # === High Curvature Mystery ===
    # Sustained high information rate: high mean curvature + many peaks
    # Detective fiction = constant revelation, sustained engagement
    curvature_high = max(0, (features.mean_curvature - CURVATURE_MED) / (CURVATURE_HIGH - CURVATURE_MED))
    peak_density = min(1.0, features.n_peaks / PEAKS_HIGH)
    scores["high_curvature_mystery"] = (curvature_high * 0.6 + peak_density * 0.4)

    # === Random Walk Comedy ===
    # High variance, oscillating: high CV + many peaks + near-zero net change
    # Comedy/picaresque = unpredictable, oscillating fortunes
    oscillation = min(1.0, cv / 0.25)  # CV > 0.25 is very oscillatory
    peak_score = min(1.0, features.n_peaks / PEAKS_MED)
    mean_reversion = 1.0 if abs(features.entropy_change) < 0.1 else 0.5
    scores["random_walk_comedy"] = (oscillation * 0.4 + peak_score * 0.3 + mean_reversion * 0.3)

    # === Compression Progress ===
    # Steady entropy reduction: entropy decreases + low variance + positive kurtosis
    # Bildungsroman/mystery resolution = learning, uncertainty reduction
    if features.entropy_change < -0.05:
        compression_strength = min(1.0, abs(features.entropy_change) / 0.3)
        steadiness = max(0, 1.0 - cv / 0.3)  # Lower CV = steadier
        scores["compression_progress"] = compression_strength * 0.7 + steadiness * 0.3
    else:
        scores["compression_progress"] = 0.1

    # === Discontinuous Twist ===
    # Late spike: high max_curvature + peaks concentrated late (>60%)
    # Twist ending = O. Henry, surprise revelation
    late_peaks = sum(1 for p in features.peak_positions if p > 0.6)
    early_peaks = sum(1 for p in features.peak_positions if p <= 0.6)
    late_concentration = late_peaks / max(1, late_peaks + early_peaks)
    max_curv_extreme = min(1.0, features.max_curvature / MAX_CURV_HIGH)
    scores["discontinuous_twist"] = late_concentration * 0.6 + max_curv_extreme * 0.4

    # Normalize scores to sum to 1
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    return scores
