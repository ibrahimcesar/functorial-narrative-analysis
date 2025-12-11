"""
Campbell's Hero's Journey (Monomyth) Detector

Detects Joseph Campbell's 17-stage Hero's Journey from "The Hero with a Thousand Faces" (1949).

The journey is divided into three acts:

    ═══════════════════════════════════════════════════════════════════════
    ACT I: DEPARTURE (Separation)
    ═══════════════════════════════════════════════════════════════════════
    1. THE ORDINARY WORLD
       Hero's normal life before the adventure
    2. CALL TO ADVENTURE
       Hero receives invitation to journey
    3. REFUSAL OF THE CALL
       Hero hesitates, fears the unknown
    4. MEETING THE MENTOR
       Hero gains guidance, confidence, tools
    5. CROSSING THE FIRST THRESHOLD
       Hero commits to the adventure, enters special world

    ═══════════════════════════════════════════════════════════════════════
    ACT II: INITIATION (Descent)
    ═══════════════════════════════════════════════════════════════════════
    6. TESTS, ALLIES, ENEMIES
       Hero learns rules of special world
    7. APPROACH TO THE INMOST CAVE
       Hero prepares for major challenge
    8. THE ORDEAL
       Hero faces greatest fear (death and rebirth)
    9. REWARD (SEIZING THE SWORD)
       Hero gains what they sought

    ═══════════════════════════════════════════════════════════════════════
    ACT III: RETURN (Integration)
    ═══════════════════════════════════════════════════════════════════════
    10. THE ROAD BACK
        Hero begins return, may be pursued
    11. RESURRECTION
        Final test, hero transformed
    12. RETURN WITH THE ELIXIR
        Hero returns home changed, with boon

Campbell's full 17-stage version includes additional substages. This implementation
uses the commonly-taught 12-stage version (Christopher Vogler's adaptation).
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
class HeroStage:
    """Represents a stage in the Hero's Journey."""
    number: int
    name: str
    act: str  # "departure", "initiation", "return"
    expected_position: float
    expected_sentiment: str
    description: str


# 12-stage Hero's Journey (Vogler adaptation)
HERO_STAGES = [
    # ACT I: DEPARTURE
    HeroStage(1, "Ordinary World", "departure", 0.0, "neutral",
              "Hero's normal life before adventure"),
    HeroStage(2, "Call to Adventure", "departure", 0.08, "rising",
              "Herald announces the need for change"),
    HeroStage(3, "Refusal of the Call", "departure", 0.12, "falling",
              "Hero hesitates, shows fear"),
    HeroStage(4, "Meeting the Mentor", "departure", 0.17, "rising",
              "Hero gains guidance and confidence"),
    HeroStage(5, "Crossing the Threshold", "departure", 0.25, "neutral",
              "Hero commits, enters special world"),

    # ACT II: INITIATION
    HeroStage(6, "Tests, Allies, Enemies", "initiation", 0.35, "fluctuating",
              "Hero learns rules, makes friends/foes"),
    HeroStage(7, "Approach to Inmost Cave", "initiation", 0.45, "falling",
              "Hero prepares for major challenge"),
    HeroStage(8, "The Ordeal", "initiation", 0.55, "nadir",
              "Hero faces greatest fear, death-rebirth"),
    HeroStage(9, "Reward", "initiation", 0.65, "rising",
              "Hero seizes the sword, gains prize"),

    # ACT III: RETURN
    HeroStage(10, "The Road Back", "return", 0.75, "rising",
              "Hero returns, may face pursuit"),
    HeroStage(11, "Resurrection", "return", 0.85, "peak",
              "Final test, hero transformed"),
    HeroStage(12, "Return with Elixir", "return", 1.0, "resolved",
              "Hero returns changed, brings boon"),
]


@dataclass
class HeroMatch:
    """Result of matching a trajectory to the Hero's Journey."""
    trajectory_id: str
    title: str
    conformance_score: float
    stage_positions: List[float]
    stage_values: List[float]
    detected_stages: List[int]
    ordeal_position: float  # The Ordeal (death-rebirth)
    ordeal_depth: float  # How deep the nadir goes
    resurrection_position: float
    transformation_score: float  # End vs beginning
    act_scores: Dict[str, float]  # Per-act conformance
    pattern_type: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "title": self.title,
            "conformance_score": self.conformance_score,
            "stage_positions": self.stage_positions,
            "stage_values": self.stage_values,
            "detected_stages": self.detected_stages,
            "ordeal_position": self.ordeal_position,
            "ordeal_depth": self.ordeal_depth,
            "resurrection_position": self.resurrection_position,
            "transformation_score": self.transformation_score,
            "act_scores": self.act_scores,
            "pattern_type": self.pattern_type,
            "notes": self.notes,
        }


class CampbellDetector:
    """
    Detects Campbell's Hero's Journey (Monomyth) structure.

    Key characteristics:
    1. Departure from ordinary world
    2. Descent into special world (ordeal at nadir)
    3. Ascent and return transformed

    The pattern resembles Harmon Circle but with more stages and
    emphasis on transformation/rebirth.
    """

    # Canonical Hero's Journey pattern (12 points, normalized 0-1)
    CANONICAL_PATTERN = np.array([
        0.50,  # 1. Ordinary World - neutral
        0.55,  # 2. Call to Adventure - slight rise
        0.45,  # 3. Refusal - fear, hesitation
        0.60,  # 4. Meeting Mentor - confidence boost
        0.55,  # 5. Crossing Threshold - committed
        0.45,  # 6. Tests - ups and downs
        0.35,  # 7. Approach Cave - tension builds
        0.20,  # 8. Ordeal - nadir, death-rebirth
        0.50,  # 9. Reward - achievement
        0.60,  # 10. Road Back - complications
        0.80,  # 11. Resurrection - final peak
        0.70,  # 12. Return with Elixir - resolved, transformed
    ])

    def __init__(self, smooth_sigma: float = 3.0):
        self.smooth_sigma = smooth_sigma
        self.stages = HERO_STAGES

    def _preprocess_trajectory(self, values: np.ndarray) -> np.ndarray:
        """Smooth and normalize trajectory."""
        smoothed = gaussian_filter1d(values, sigma=self.smooth_sigma)
        min_val, max_val = smoothed.min(), smoothed.max()
        if max_val - min_val > 1e-8:
            normalized = (smoothed - min_val) / (max_val - min_val)
        else:
            normalized = np.full_like(smoothed, 0.5)
        return normalized

    def _find_ordeal(self, trajectory: np.ndarray) -> Tuple[int, float, float]:
        """
        Find the Ordeal (deepest point, death-rebirth moment).

        Returns: (index, position, depth)
        """
        n = len(trajectory)

        # Look for global minimum in middle portion (40-70% typically)
        middle_start = int(0.3 * n)
        middle_end = int(0.8 * n)

        middle_section = trajectory[middle_start:middle_end]
        if len(middle_section) == 0:
            return n // 2, 0.5, 0.5

        local_min_idx = np.argmin(middle_section)
        ordeal_idx = middle_start + local_min_idx
        ordeal_position = ordeal_idx / (n - 1)
        ordeal_depth = 1 - trajectory[ordeal_idx]  # Lower = deeper

        return ordeal_idx, ordeal_position, ordeal_depth

    def _find_resurrection(self, trajectory: np.ndarray, ordeal_idx: int) -> Tuple[int, float]:
        """
        Find the Resurrection (peak after ordeal).

        Returns: (index, position)
        """
        n = len(trajectory)

        # Look for maximum after the ordeal
        after_ordeal = trajectory[ordeal_idx:]
        if len(after_ordeal) == 0:
            return n - 1, 1.0

        resurrection_local_idx = np.argmax(after_ordeal)
        resurrection_idx = ordeal_idx + resurrection_local_idx
        resurrection_position = resurrection_idx / (n - 1)

        return resurrection_idx, resurrection_position

    def _map_to_stages(self, trajectory: np.ndarray, ordeal_idx: int) -> Tuple[List[float], List[float]]:
        """Map trajectory to 12 Hero's Journey stages."""
        n = len(trajectory)
        ordeal_pos = ordeal_idx / (n - 1)

        # Adjust canonical positions based on actual ordeal location
        # Ordeal should be at stage 8 (canonical ~0.55)
        scale_factor = ordeal_pos / 0.55 if ordeal_pos > 0 else 1.0

        stage_positions = []
        for stage in HERO_STAGES:
            if stage.number <= 8:
                # Scale stages before ordeal
                pos = stage.expected_position * scale_factor
            else:
                # Stages after ordeal scale from ordeal to end
                remaining = stage.expected_position - 0.55
                pos = ordeal_pos + remaining * ((1 - ordeal_pos) / 0.45)

            stage_positions.append(min(1.0, max(0.0, pos)))

        # Get values at stage positions
        x_norm = np.linspace(0, 1, n)
        interp_func = interp1d(x_norm, trajectory, kind='linear', fill_value='extrapolate')
        stage_values = [float(interp_func(pos)) for pos in stage_positions]

        return stage_positions, stage_values

    def _compute_act_scores(
        self,
        trajectory: np.ndarray,
        stage_values: List[float]
    ) -> Dict[str, float]:
        """Compute conformance score for each act."""
        act_scores = {}

        # ACT I: DEPARTURE (stages 1-5)
        # Should show: neutral -> slight variation -> committed
        departure_vals = stage_values[:5]
        # Check for refusal (dip at stage 3) and mentor boost (rise at stage 4)
        has_refusal = departure_vals[2] < departure_vals[1]
        has_mentor_boost = departure_vals[3] > departure_vals[2]
        departure_variance = np.var(departure_vals)
        act_scores['departure'] = (
            0.3 * (1 if has_refusal else 0.3) +
            0.3 * (1 if has_mentor_boost else 0.3) +
            0.4 * min(1, departure_variance * 10)  # Some variation expected
        )

        # ACT II: INITIATION (stages 6-9)
        # Should show: tests -> descent to ordeal -> reward rise
        initiation_vals = stage_values[5:9]
        has_descent = initiation_vals[2] < initiation_vals[0]  # Ordeal lower than tests
        has_reward = initiation_vals[3] > initiation_vals[2]  # Reward higher than ordeal
        ordeal_is_nadir = initiation_vals[2] == min(stage_values)
        act_scores['initiation'] = (
            0.3 * (1 if has_descent else 0.2) +
            0.4 * (1 if has_reward else 0.2) +
            0.3 * (1 if ordeal_is_nadir else 0.5)
        )

        # ACT III: RETURN (stages 10-12)
        # Should show: road back -> resurrection peak -> return resolved
        return_vals = stage_values[9:]
        has_resurrection_peak = len(return_vals) >= 2 and return_vals[1] > return_vals[0]
        ends_high = return_vals[-1] > 0.5
        transformation = return_vals[-1] - stage_values[0]
        act_scores['return'] = (
            0.3 * (1 if has_resurrection_peak else 0.3) +
            0.3 * (1 if ends_high else 0.3) +
            0.4 * max(0, min(1, transformation + 0.5))  # Positive transformation
        )

        return act_scores

    def _compute_conformance(
        self,
        trajectory: np.ndarray,
        stage_values: List[float],
        ordeal_position: float,
        ordeal_depth: float,
        act_scores: Dict[str, float]
    ) -> Tuple[float, str, List[str]]:
        """Compute overall conformance to Hero's Journey."""
        notes = []
        scores = []

        # 1. Correlation with canonical pattern
        resampled = np.interp(
            np.linspace(0, 1, 12),
            np.linspace(0, 1, len(trajectory)),
            trajectory
        )
        correlation = np.corrcoef(resampled, self.CANONICAL_PATTERN)[0, 1]
        if not np.isnan(correlation):
            scores.append(max(0, correlation))
            if correlation > 0.5:
                notes.append("Strong pattern correlation")

        # 2. Ordeal position (should be 45-65% through)
        if 0.4 < ordeal_position < 0.7:
            notes.append(f"Well-positioned ordeal at {ordeal_position:.0%}")
            scores.append(1.0)
        elif 0.3 < ordeal_position < 0.8:
            scores.append(0.6)
        else:
            notes.append(f"Unusual ordeal position at {ordeal_position:.0%}")
            scores.append(0.3)

        # 3. Ordeal depth (should be significant descent)
        if ordeal_depth > 0.5:
            notes.append("Deep ordeal (significant descent)")
            scores.append(1.0)
        elif ordeal_depth > 0.3:
            scores.append(0.7)
        else:
            notes.append("Shallow ordeal")
            scores.append(0.4)

        # 4. Transformation (end should be higher/different than start)
        transformation = stage_values[-1] - stage_values[0]
        if transformation > 0.1:
            notes.append(f"Positive transformation (+{transformation:.2f})")
            scores.append(1.0)
        elif transformation > -0.1:
            scores.append(0.7)
        else:
            notes.append("Negative transformation (tragic journey)")
            scores.append(0.5)  # Not wrong, just different

        # 5. Act scores
        for act, score in act_scores.items():
            scores.append(score)

        conformance = np.mean(scores)

        # Classify pattern
        if conformance > 0.7 and ordeal_depth > 0.4:
            pattern_type = "classic_hero_journey"
            notes.append("Strong Hero's Journey structure")
        elif conformance > 0.55:
            pattern_type = "modified_journey"
            notes.append("Modified Hero's Journey")
        elif ordeal_depth < 0.2:
            pattern_type = "flat_journey"
            notes.append("Journey without significant ordeal")
        elif transformation < -0.2:
            pattern_type = "tragic_journey"
            notes.append("Tragic hero journey (no return)")
        elif act_scores.get('return', 0) < 0.3:
            pattern_type = "incomplete_journey"
            notes.append("Incomplete journey (weak return)")
        else:
            pattern_type = "non_conforming"

        return conformance, pattern_type, notes

    def detect(
        self,
        trajectory: np.ndarray,
        trajectory_id: str = "unknown",
        title: str = "Unknown"
    ) -> HeroMatch:
        """Detect Campbell's Hero's Journey structure in a trajectory."""
        processed = self._preprocess_trajectory(trajectory)

        # Find key moments
        ordeal_idx, ordeal_position, ordeal_depth = self._find_ordeal(processed)
        resurrection_idx, resurrection_position = self._find_resurrection(processed, ordeal_idx)

        # Map to stages
        stage_positions, stage_values = self._map_to_stages(processed, ordeal_idx)

        # Compute act scores
        act_scores = self._compute_act_scores(processed, stage_values)

        # Compute transformation
        transformation_score = stage_values[-1] - stage_values[0]

        # Compute conformance
        conformance, pattern_type, notes = self._compute_conformance(
            processed, stage_values, ordeal_position, ordeal_depth, act_scores
        )

        # All stages detected by default
        detected_stages = list(range(1, 13))

        return HeroMatch(
            trajectory_id=trajectory_id,
            title=title,
            conformance_score=conformance,
            stage_positions=stage_positions,
            stage_values=stage_values,
            detected_stages=detected_stages,
            ordeal_position=ordeal_position,
            ordeal_depth=ordeal_depth,
            resurrection_position=resurrection_position,
            transformation_score=transformation_score,
            act_scores=act_scores,
            pattern_type=pattern_type,
            notes=notes,
        )
