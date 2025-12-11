"""
Freytag's Pyramid Detector

Detects Gustav Freytag's five-act dramatic structure (1863):

                        CLIMAX (3)
                          /\
                         /  \
                        /    \
            RISING (2) /      \ FALLING (4)
                      /        \
                     /          \
    EXPOSITION (1)  /            \ DÉNOUEMENT (5)
    _______________/              \_______________

    1. EXPOSITION (Einleitung)
       - Introduction of setting, characters, basic conflict
       - Establishes the "normal world"

    2. RISING ACTION (Steigerung)
       - Complications, building tension
       - Series of events that create suspense
       - Includes "exciting force" (erregendes Moment)

    3. CLIMAX (Höhepunkt)
       - Turning point of the drama
       - Moment of highest tension
       - Protagonist's fate is decided

    4. FALLING ACTION (Fallende Handlung)
       - Events unfold from climax toward resolution
       - Includes "moment of final suspense" (Moment der letzten Spannung)

    5. DÉNOUEMENT/CATASTROPHE (Katastrophe)
       - Final resolution
       - Tragedy: protagonist's downfall
       - Comedy: protagonist's triumph

Freytag developed this primarily for tragedy but it applies to drama broadly.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks, argrelextrema
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


@dataclass
class FreytagStage:
    """Represents a stage in Freytag's Pyramid."""
    number: int
    name: str
    german_name: str
    expected_position: float
    expected_slope: str  # "flat", "rising", "peak", "falling"
    description: str


FREYTAG_STAGES = [
    FreytagStage(1, "Exposition", "Einleitung", 0.1, "flat",
                 "Introduction of setting, characters, conflict"),
    FreytagStage(2, "Rising Action", "Steigerung", 0.35, "rising",
                 "Complications and building tension"),
    FreytagStage(3, "Climax", "Höhepunkt", 0.5, "peak",
                 "Turning point, highest tension"),
    FreytagStage(4, "Falling Action", "Fallende Handlung", 0.7, "falling",
                 "Events unfold toward resolution"),
    FreytagStage(5, "Dénouement", "Katastrophe", 0.9, "flat",
                 "Final resolution or catastrophe"),
]


@dataclass
class FreytagMatch:
    """Result of matching a trajectory to Freytag's Pyramid."""
    trajectory_id: str
    title: str
    conformance_score: float
    stage_positions: List[float]
    stage_values: List[float]
    climax_position: float
    climax_value: float
    pyramid_height: float  # Difference between climax and exposition
    symmetry_score: float  # How symmetric the pyramid is
    resolution_type: str  # "tragic", "comic", "neutral"
    pattern_type: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "title": self.title,
            "conformance_score": self.conformance_score,
            "stage_positions": self.stage_positions,
            "stage_values": self.stage_values,
            "climax_position": self.climax_position,
            "climax_value": self.climax_value,
            "pyramid_height": self.pyramid_height,
            "symmetry_score": self.symmetry_score,
            "resolution_type": self.resolution_type,
            "pattern_type": self.pattern_type,
            "notes": self.notes,
        }


class FreytagDetector:
    """
    Detects Freytag's Pyramid structure in narrative trajectories.

    The pyramid model expects:
    1. Rising action from exposition to climax
    2. Single clear climax near the middle
    3. Falling action from climax to resolution
    4. Roughly symmetric structure
    """

    # Canonical pyramid pattern (normalized 0-1)
    CANONICAL_PATTERN = np.array([
        0.3,   # 1. Exposition - low/baseline
        0.5,   # Rising toward climax
        0.7,   # 2. Rising Action midpoint
        0.9,   # Approaching climax
        1.0,   # 3. Climax - peak
        0.85,  # Beginning fall
        0.6,   # 4. Falling Action midpoint
        0.4,   # Approaching resolution
        0.35,  # 5. Dénouement - resolved
    ])

    def __init__(self, smooth_sigma: float = 3.0):
        self.smooth_sigma = smooth_sigma
        self.stages = FREYTAG_STAGES

    def _preprocess_trajectory(self, values: np.ndarray) -> np.ndarray:
        """Smooth and normalize trajectory."""
        smoothed = gaussian_filter1d(values, sigma=self.smooth_sigma)
        min_val, max_val = smoothed.min(), smoothed.max()
        if max_val - min_val > 1e-8:
            normalized = (smoothed - min_val) / (max_val - min_val)
        else:
            normalized = np.full_like(smoothed, 0.5)
        return normalized

    def _find_climax(self, trajectory: np.ndarray) -> Tuple[int, float]:
        """Find the climax (highest point) in the trajectory."""
        n = len(trajectory)

        # Look for global maximum
        global_max_idx = np.argmax(trajectory)

        # Also look for local maxima
        peaks, properties = find_peaks(trajectory, distance=n//10, prominence=0.05)

        if len(peaks) > 0:
            # Find the most prominent peak
            peak_values = trajectory[peaks]
            most_prominent_idx = peaks[np.argmax(peak_values)]

            # Prefer peaks closer to middle (Freytag expects climax around middle)
            middle = n // 2
            distances = np.abs(peaks - middle)
            middle_peaks = peaks[distances < n // 4]

            if len(middle_peaks) > 0:
                # Take highest peak near middle
                climax_idx = middle_peaks[np.argmax(trajectory[middle_peaks])]
            else:
                climax_idx = most_prominent_idx
        else:
            climax_idx = global_max_idx

        climax_position = climax_idx / (n - 1)
        climax_value = trajectory[climax_idx]

        return climax_idx, climax_position, climax_value

    def _map_to_stages(self, trajectory: np.ndarray, climax_idx: int) -> Tuple[List[float], List[float]]:
        """Map trajectory to Freytag's 5 stages based on climax position."""
        n = len(trajectory)
        climax_pos = climax_idx / (n - 1)

        # Stage positions relative to climax
        stage_positions = [
            0.0,  # Exposition start
            climax_pos * 0.5,  # Rising Action midpoint
            climax_pos,  # Climax
            climax_pos + (1 - climax_pos) * 0.5,  # Falling Action midpoint
            1.0,  # Dénouement end
        ]

        # Get values at these positions
        x_norm = np.linspace(0, 1, n)
        interp_func = interp1d(x_norm, trajectory, kind='linear', fill_value='extrapolate')
        stage_values = [float(interp_func(pos)) for pos in stage_positions]

        return stage_positions, stage_values

    def _compute_symmetry(self, trajectory: np.ndarray, climax_idx: int) -> float:
        """Compute how symmetric the pyramid is around the climax."""
        n = len(trajectory)

        # Get rising and falling portions
        rising = trajectory[:climax_idx + 1]
        falling = trajectory[climax_idx:]

        if len(rising) < 2 or len(falling) < 2:
            return 0.5

        # Normalize both to same length for comparison
        target_len = min(len(rising), len(falling), 50)

        rising_resampled = np.interp(
            np.linspace(0, 1, target_len),
            np.linspace(0, 1, len(rising)),
            rising
        )
        falling_resampled = np.interp(
            np.linspace(0, 1, target_len),
            np.linspace(0, 1, len(falling)),
            falling[::-1]  # Reverse for comparison
        )

        # Correlation between rising and reversed falling
        correlation = np.corrcoef(rising_resampled, falling_resampled)[0, 1]

        if np.isnan(correlation):
            return 0.5

        # Convert to 0-1 score
        return max(0, (correlation + 1) / 2)

    def _determine_resolution_type(self, trajectory: np.ndarray, stage_values: List[float]) -> str:
        """Determine if the resolution is tragic, comic, or neutral."""
        exposition_value = stage_values[0]
        denouement_value = stage_values[-1]

        diff = denouement_value - exposition_value

        if diff < -0.15:
            return "tragic"  # Ends lower than it started
        elif diff > 0.15:
            return "comic"  # Ends higher than it started
        else:
            return "neutral"  # Returns to similar level

    def _compute_conformance(
        self,
        trajectory: np.ndarray,
        stage_positions: List[float],
        stage_values: List[float],
        climax_position: float,
        symmetry_score: float
    ) -> Tuple[float, str, List[str]]:
        """Compute conformance to Freytag's Pyramid."""
        notes = []
        scores = []

        # 1. Climax position (should be near middle, Freytag said 50%)
        climax_deviation = abs(climax_position - 0.5)
        if climax_deviation < 0.1:
            notes.append("Well-centered climax")
            scores.append(1.0)
        elif climax_deviation < 0.2:
            scores.append(0.7)
        else:
            notes.append(f"Off-center climax at {climax_position:.0%}")
            scores.append(0.4)

        # 2. Rising action (exposition < climax)
        if stage_values[0] < stage_values[2]:
            rise_magnitude = stage_values[2] - stage_values[0]
            notes.append(f"Rising action (Δ={rise_magnitude:.2f})")
            scores.append(min(1.0, 0.5 + rise_magnitude))
        else:
            notes.append("No clear rising action")
            scores.append(0.2)

        # 3. Falling action (climax > denouement)
        if stage_values[2] > stage_values[4]:
            fall_magnitude = stage_values[2] - stage_values[4]
            notes.append(f"Falling action (Δ={fall_magnitude:.2f})")
            scores.append(min(1.0, 0.5 + fall_magnitude))
        else:
            notes.append("No clear falling action")
            scores.append(0.2)

        # 4. Symmetry score
        scores.append(symmetry_score)
        if symmetry_score > 0.7:
            notes.append("Symmetric pyramid structure")

        # 5. Single climax (check for multiple peaks)
        peaks, _ = find_peaks(trajectory, distance=len(trajectory)//10, prominence=0.1)
        if len(peaks) <= 1:
            notes.append("Single clear climax")
            scores.append(1.0)
        elif len(peaks) <= 3:
            scores.append(0.6)
        else:
            notes.append(f"Multiple peaks ({len(peaks)})")
            scores.append(0.3)

        # 6. Correlation with canonical pattern
        # Resample trajectory to match canonical length
        resampled = np.interp(
            np.linspace(0, 1, len(self.CANONICAL_PATTERN)),
            np.linspace(0, 1, len(trajectory)),
            trajectory
        )
        correlation = np.corrcoef(resampled, self.CANONICAL_PATTERN)[0, 1]
        if not np.isnan(correlation):
            scores.append(max(0, correlation))

        conformance = np.mean(scores)

        # Classify pattern
        if conformance > 0.7:
            pattern_type = "classic_pyramid"
            notes.append("Strong Freytag pyramid structure")
        elif conformance > 0.5:
            pattern_type = "modified_pyramid"
        elif climax_position < 0.3:
            pattern_type = "early_climax"
            notes.append("Climax too early for classic pyramid")
        elif climax_position > 0.7:
            pattern_type = "late_climax"
            notes.append("Climax too late for classic pyramid")
        elif symmetry_score < 0.4:
            pattern_type = "asymmetric"
            notes.append("Asymmetric structure")
        else:
            pattern_type = "non_conforming"

        return conformance, pattern_type, notes

    def detect(
        self,
        trajectory: np.ndarray,
        trajectory_id: str = "unknown",
        title: str = "Unknown"
    ) -> FreytagMatch:
        """Detect Freytag's Pyramid structure in a trajectory."""
        processed = self._preprocess_trajectory(trajectory)

        # Find climax
        climax_idx, climax_position, climax_value = self._find_climax(processed)

        # Map to stages
        stage_positions, stage_values = self._map_to_stages(processed, climax_idx)

        # Compute symmetry
        symmetry_score = self._compute_symmetry(processed, climax_idx)

        # Determine resolution type
        resolution_type = self._determine_resolution_type(processed, stage_values)

        # Compute pyramid height
        pyramid_height = climax_value - stage_values[0]

        # Compute conformance
        conformance, pattern_type, notes = self._compute_conformance(
            processed, stage_positions, stage_values, climax_position, symmetry_score
        )

        return FreytagMatch(
            trajectory_id=trajectory_id,
            title=title,
            conformance_score=conformance,
            stage_positions=stage_positions,
            stage_values=stage_values,
            climax_position=climax_position,
            climax_value=climax_value,
            pyramid_height=pyramid_height,
            symmetry_score=symmetry_score,
            resolution_type=resolution_type,
            pattern_type=pattern_type,
            notes=notes,
        )
