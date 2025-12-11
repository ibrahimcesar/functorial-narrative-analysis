"""
Aristotle's Three-Act Structure Detector

Detects the classical dramatic structure from Aristotle's Poetics:

    ACT I: PROTASIS (Beginning/Setup)
        - Introduction of characters, setting, conflict
        - Establishes the status quo
        - Ends with the "inciting incident"

    ACT II: EPITASIS (Middle/Confrontation)
        - Rising action and complications
        - Character struggles with obstacles
        - The longest act (typically 50% of narrative)
        - Contains the "crisis" or "turning point"

    ACT III: CATASTROPHE (End/Resolution)
        - Climax and falling action
        - Resolution of conflict
        - Denouement (untying of the knot)

Aristotle emphasized unity of action, time, and place.
The detector evaluates structural balance and dramatic arc.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


@dataclass
class AristotleAct:
    """Represents an act in the three-act structure."""
    number: int
    name: str
    greek_name: str
    expected_proportion: float  # Expected proportion of narrative
    expected_sentiment: str  # "stable", "rising", "falling", "peak"
    description: str


# Define the 3 acts
ARISTOTLE_ACTS = [
    AristotleAct(1, "Beginning", "Protasis", 0.25, "stable",
                 "Setup: characters, setting, initial conflict"),
    AristotleAct(2, "Middle", "Epitasis", 0.50, "rising",
                 "Confrontation: complications, obstacles, crisis"),
    AristotleAct(3, "End", "Catastrophe", 0.25, "falling",
                 "Resolution: climax, falling action, denouement"),
]


@dataclass
class AristotleMatch:
    """Result of matching a trajectory to Aristotle's structure."""
    trajectory_id: str
    title: str
    conformance_score: float
    act_boundaries: List[float]  # [0, act1_end, act2_end, 1.0]
    act_means: List[float]  # Mean sentiment in each act
    act_variances: List[float]  # Variance in each act
    inciting_incident: float  # Position of first major change
    crisis_point: float  # Position of maximum tension
    climax_point: float  # Position of climax
    pattern_type: str
    unity_score: float  # Measure of structural unity
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "title": self.title,
            "conformance_score": self.conformance_score,
            "act_boundaries": self.act_boundaries,
            "act_means": self.act_means,
            "act_variances": self.act_variances,
            "inciting_incident": self.inciting_incident,
            "crisis_point": self.crisis_point,
            "climax_point": self.climax_point,
            "pattern_type": self.pattern_type,
            "unity_score": self.unity_score,
            "notes": self.notes,
        }


class AristotleDetector:
    """
    Detects Aristotle's Three-Act Structure in narrative trajectories.

    Key principles from Poetics:
    1. Beginning: has nothing necessarily before it
    2. Middle: has something before and after
    3. End: has something before but nothing after
    4. Proper magnitude: long enough to show change of fortune
    5. Unity: single, complete action
    """

    # Canonical three-act pattern (normalized 0-1)
    # Act I: stable/slight rise, Act II: rise to peak, Act III: fall then resolve
    CANONICAL_PATTERN = {
        'positions': np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]),
        'values': np.array([0.5, 0.5, 0.55, 0.6, 0.7, 0.8, 0.6, 0.5, 0.55])
    }

    def __init__(self, smooth_sigma: float = 3.0):
        self.smooth_sigma = smooth_sigma
        self.acts = ARISTOTLE_ACTS

    def _preprocess_trajectory(self, values: np.ndarray) -> np.ndarray:
        """Smooth and normalize trajectory."""
        smoothed = gaussian_filter1d(values, sigma=self.smooth_sigma)
        min_val, max_val = smoothed.min(), smoothed.max()
        if max_val - min_val > 1e-8:
            normalized = (smoothed - min_val) / (max_val - min_val)
        else:
            normalized = np.full_like(smoothed, 0.5)
        return normalized

    def _find_act_boundaries(self, trajectory: np.ndarray) -> List[float]:
        """
        Find optimal act boundaries based on structural analysis.

        Returns [0.0, act1_end, act2_end, 1.0]
        """
        n = len(trajectory)

        # Find significant changes in trajectory
        derivative = np.gradient(trajectory)
        abs_derivative = np.abs(derivative)

        # Find peaks in derivative (major changes)
        peaks, properties = find_peaks(abs_derivative, distance=n//10, prominence=0.01)

        if len(peaks) >= 2:
            # Sort by prominence/significance
            peak_values = abs_derivative[peaks]
            sorted_indices = np.argsort(peak_values)[::-1]

            # Take top 2 peaks as act boundaries
            boundary_candidates = sorted(peaks[sorted_indices[:2]])

            act1_end = boundary_candidates[0] / (n - 1)
            act2_end = boundary_candidates[1] / (n - 1) if len(boundary_candidates) > 1 else 0.75

            # Ensure reasonable proportions (Act II should be largest)
            if act1_end < 0.15:
                act1_end = 0.25
            if act2_end < 0.5:
                act2_end = 0.75
            if act2_end - act1_end < 0.3:  # Act II too short
                act1_end = 0.25
                act2_end = 0.75
        else:
            # Default proportions
            act1_end = 0.25
            act2_end = 0.75

        return [0.0, act1_end, act2_end, 1.0]

    def _find_key_points(self, trajectory: np.ndarray) -> Dict[str, float]:
        """Find inciting incident, crisis, and climax positions."""
        n = len(trajectory)

        # Derivative for finding changes
        derivative = np.gradient(trajectory)

        # Inciting incident: first significant rise (usually in Act I)
        first_quarter = n // 4
        first_quarter_deriv = derivative[:first_quarter]
        if len(first_quarter_deriv) > 0:
            inciting_idx = np.argmax(first_quarter_deriv)
            inciting_incident = inciting_idx / (n - 1)
        else:
            inciting_incident = 0.2

        # Crisis: point of maximum tension (usually mid-Act II)
        # Look for maximum arousal/derivative in middle section
        middle_start = n // 4
        middle_end = 3 * n // 4
        middle_tension = np.abs(derivative[middle_start:middle_end])
        if len(middle_tension) > 0:
            crisis_idx = middle_start + np.argmax(middle_tension)
            crisis_point = crisis_idx / (n - 1)
        else:
            crisis_point = 0.5

        # Climax: global maximum (usually end of Act II / start of Act III)
        climax_idx = np.argmax(trajectory)
        climax_point = climax_idx / (n - 1)

        return {
            'inciting_incident': inciting_incident,
            'crisis_point': crisis_point,
            'climax_point': climax_point,
        }

    def _compute_unity_score(self, trajectory: np.ndarray, boundaries: List[float]) -> float:
        """
        Compute Aristotle's "unity of action" score.

        High unity = single dramatic arc, coherent structure
        Low unity = fragmented, multiple unrelated arcs
        """
        n = len(trajectory)

        # 1. Check for single dominant arc (not multiple peaks)
        peaks, _ = find_peaks(trajectory, distance=n//10)
        if len(peaks) == 0:
            n_peaks = 1  # Monotonic
        else:
            n_peaks = len(peaks)

        # Penalty for multiple peaks (multiple arcs)
        peak_penalty = max(0, 1 - (n_peaks - 1) * 0.2)

        # 2. Check that middle section has rising action
        act1_end_idx = int(boundaries[1] * (n - 1))
        act2_end_idx = int(boundaries[2] * (n - 1))

        act2 = trajectory[act1_end_idx:act2_end_idx]
        if len(act2) > 1:
            # Rising action in Act II
            rising_ratio = np.sum(np.diff(act2) > 0) / len(np.diff(act2))
        else:
            rising_ratio = 0.5

        # 3. Check for resolution in Act III
        act3 = trajectory[act2_end_idx:]
        if len(act3) > 1:
            # Should show some falling then stabilization
            act3_range = np.max(act3) - np.min(act3)
            resolution = 1 - min(1, act3_range * 2)  # Less variance = more resolved
        else:
            resolution = 0.5

        unity = (peak_penalty * 0.4 + rising_ratio * 0.3 + resolution * 0.3)
        return max(0, min(1, unity))

    def _compute_conformance(
        self,
        trajectory: np.ndarray,
        boundaries: List[float],
        key_points: Dict[str, float]
    ) -> Tuple[float, str, List[str]]:
        """Compute conformance to Aristotle's structure."""
        notes = []
        n = len(trajectory)

        # Get act segments
        act_indices = [int(b * (n - 1)) for b in boundaries]
        act1 = trajectory[:act_indices[1]]
        act2 = trajectory[act_indices[1]:act_indices[2]]
        act3 = trajectory[act_indices[2]:]

        act_means = [np.mean(act1), np.mean(act2), np.mean(act3)]
        act_variances = [np.var(act1), np.var(act2), np.var(act3)]

        scores = []

        # 1. Act proportion score (Act II should be ~50%)
        act2_proportion = boundaries[2] - boundaries[1]
        proportion_score = 1 - abs(act2_proportion - 0.5)
        scores.append(proportion_score)

        # 2. Rising action in Act II
        if act_means[1] > act_means[0]:
            notes.append("Rising action in Act II")
            scores.append(0.8)
        else:
            scores.append(0.3)

        # 3. Climax position (should be late Act II or early Act III)
        climax = key_points['climax_point']
        if 0.6 < climax < 0.85:
            notes.append(f"Well-positioned climax at {climax:.0%}")
            scores.append(1.0)
        elif 0.5 < climax < 0.9:
            scores.append(0.7)
        else:
            notes.append(f"Unusual climax position at {climax:.0%}")
            scores.append(0.3)

        # 4. Resolution in Act III
        if len(act3) > 1:
            act3_trend = np.polyfit(range(len(act3)), act3, 1)[0]
            if act3_trend < 0:  # Falling action
                notes.append("Clear falling action in Act III")
                scores.append(0.9)
            else:
                scores.append(0.5)
        else:
            scores.append(0.5)

        # 5. Beginning stability
        if act_variances[0] < act_variances[1]:
            notes.append("Stable beginning (setup)")
            scores.append(0.8)
        else:
            scores.append(0.5)

        conformance = np.mean(scores)

        # Classify pattern
        if conformance > 0.7:
            pattern_type = "classic_three_act"
            notes.append("Strong Aristotelian structure")
        elif conformance > 0.5:
            pattern_type = "modified_three_act"
            notes.append("Modified three-act structure")
        elif act_means[0] > act_means[2]:
            pattern_type = "tragic"
            notes.append("Tragic arc (decline)")
        elif act_means[0] < act_means[2]:
            pattern_type = "comedic"
            notes.append("Comedic arc (rise)")
        else:
            pattern_type = "non_conforming"

        return conformance, pattern_type, notes, act_means, act_variances

    def detect(
        self,
        trajectory: np.ndarray,
        trajectory_id: str = "unknown",
        title: str = "Unknown"
    ) -> AristotleMatch:
        """Detect Aristotle's three-act structure in a trajectory."""
        processed = self._preprocess_trajectory(trajectory)

        boundaries = self._find_act_boundaries(processed)
        key_points = self._find_key_points(processed)
        unity_score = self._compute_unity_score(processed, boundaries)

        conformance, pattern_type, notes, act_means, act_variances = \
            self._compute_conformance(processed, boundaries, key_points)

        return AristotleMatch(
            trajectory_id=trajectory_id,
            title=title,
            conformance_score=conformance,
            act_boundaries=boundaries,
            act_means=act_means,
            act_variances=act_variances,
            inciting_incident=key_points['inciting_incident'],
            crisis_point=key_points['crisis_point'],
            climax_point=key_points['climax_point'],
            pattern_type=pattern_type,
            unity_score=unity_score,
            notes=notes,
        )
