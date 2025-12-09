"""
Campbell's Monomyth / Hero's Journey Detector

Detects the 17-stage Hero's Journey structure described by Joseph Campbell
in "The Hero with a Thousand Faces" (1949). This structure identifies a
universal pattern in mythological narratives across cultures.

The three main acts and 17 stages:

    ACT I - DEPARTURE (Separation)
        1. The Ordinary World - Hero's normal life
        2. Call to Adventure - Challenge or quest presented
        3. Refusal of the Call - Hero hesitates or refuses
        4. Meeting the Mentor - Helper appears with guidance
        5. Crossing the First Threshold - Hero commits to adventure

    ACT II - INITIATION (Descent)
        6. Tests, Allies, Enemies - Hero faces challenges, gains companions
        7. Approach to the Inmost Cave - Preparation for major ordeal
        8. The Ordeal - Supreme crisis, death and rebirth
        9. Reward (Seizing the Sword) - Hero gains prize or knowledge
        10. The Road Back - Begin journey home

    ACT III - RETURN (Integration)
        11. Resurrection - Final test, transformation complete
        12. Return with the Elixir - Hero brings boon to ordinary world

Note: Campbell's original formulation includes additional nuances (17 stages),
but this detector uses the commonly-adopted 12-stage synthesis popularized
by Christopher Vogler in "The Writer's Journey."

Geometric representation:

    ORDINARY WORLD ─────────────────────────────── RETURN WITH ELIXIR
          │                                               ↑
          ↓ Call to Adventure                             │
    ┌─────────────────────────────────────────────────────┤
    │                    SPECIAL WORLD                    │
    │                                                     │
    │   Tests ──→ Approach ──→ ORDEAL ──→ Reward ──→ Road Back
    │              Cave         (nadir)                   │
    └─────────────────────────────────────────────────────┘

Category-theoretic formalization:
    Define diagram category Journey-17 (or Journey-12):
        Objects: {OrdinaryWorld, Call, Refusal, Mentor, Threshold,
                  Tests, Approach, Ordeal, Reward, RoadBack,
                  Resurrection, Return}
        Morphisms: Sequential transitions between stages

    A hero's journey is a functor H: Journey-12 → Narr that:
        - Maps each stage to a narrative segment H(i)
        - Maps transitions to plot beats
        - Preserves the Departure-Initiation-Return structure

    The descent/ascent functor D: Journey-12 → 2 where 2 = {Ordinary, Special}:
        D(1-4, 11-12) = Ordinary
        D(5-10) = Special
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks, argrelextrema
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import click
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class JourneyStage:
    """Represents a stage in the Hero's Journey."""
    number: int
    name: str
    act: str  # "departure", "initiation", "return"
    world: str  # "ordinary" or "special"
    expected_tension: str  # "baseline", "rising", "peak", "falling", "transformed"
    description: str


# Define the 12 stages (Vogler synthesis)
JOURNEY_STAGES = [
    # ACT I - DEPARTURE
    JourneyStage(1, "Ordinary World", "departure", "ordinary", "baseline",
                 "Hero's normal life before adventure"),
    JourneyStage(2, "Call to Adventure", "departure", "ordinary", "rising",
                 "Challenge or quest is presented"),
    JourneyStage(3, "Refusal of the Call", "departure", "ordinary", "baseline",
                 "Hero hesitates or refuses the call"),
    JourneyStage(4, "Meeting the Mentor", "departure", "ordinary", "rising",
                 "Helper provides guidance, training, or gifts"),
    JourneyStage(5, "Crossing the Threshold", "departure", "threshold", "rising",
                 "Hero commits and enters the special world"),

    # ACT II - INITIATION
    JourneyStage(6, "Tests, Allies, Enemies", "initiation", "special", "rising",
                 "Hero faces challenges and gains companions"),
    JourneyStage(7, "Approach to Inmost Cave", "initiation", "special", "rising",
                 "Preparation for major ordeal"),
    JourneyStage(8, "The Ordeal", "initiation", "special", "peak",
                 "Supreme crisis - death and rebirth moment"),
    JourneyStage(9, "Reward", "initiation", "special", "falling",
                 "Hero seizes the prize or gains knowledge"),
    JourneyStage(10, "The Road Back", "initiation", "special", "falling",
                  "Begin journey back to ordinary world"),

    # ACT III - RETURN
    JourneyStage(11, "Resurrection", "return", "threshold", "peak",
                  "Final test, climactic moment, transformation complete"),
    JourneyStage(12, "Return with Elixir", "return", "ordinary", "transformed",
                  "Hero returns with boon for ordinary world"),
]


@dataclass
class JourneyMatch:
    """Result of matching a trajectory to the Hero's Journey."""
    trajectory_id: str
    title: str
    conformance_score: float  # 0-1, how well it matches
    stage_positions: List[float]  # Normalized positions [0,1] for each stage
    stage_values: List[float]  # Trajectory values at each stage
    detected_stages: List[int]  # Which stages were clearly detected
    threshold_crossing: float  # Position of first threshold (entering special world)
    ordeal_position: float  # Position of the supreme ordeal
    ordeal_depth: float  # How deep the ordeal goes (crisis intensity)
    resurrection_position: float  # Position of the resurrection/final test
    return_threshold: float  # Position of return to ordinary world
    transformation_score: float  # How much hero changed (end vs start)
    pattern_type: str  # "full_journey", "truncated", "inverted", etc.
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "title": self.title,
            "conformance_score": self.conformance_score,
            "stage_positions": self.stage_positions,
            "stage_values": self.stage_values,
            "detected_stages": self.detected_stages,
            "threshold_crossing": self.threshold_crossing,
            "ordeal_position": self.ordeal_position,
            "ordeal_depth": self.ordeal_depth,
            "resurrection_position": self.resurrection_position,
            "return_threshold": self.return_threshold,
            "transformation_score": self.transformation_score,
            "pattern_type": self.pattern_type,
            "notes": self.notes,
        }


class HerosJourneyDetector:
    """
    Detects Campbell's Hero's Journey / Monomyth structure in narratives.

    The detector identifies:
    1. The three-act structure (Departure, Initiation, Return)
    2. Key structural points (thresholds, ordeal, resurrection)
    3. The descent-ascent pattern into and out of the "special world"
    4. Transformation of the hero (difference between start and end)

    This is closely related to the Harmon Story Circle, which is a
    simplified 8-step derivative of the Hero's Journey.
    """

    # Canonical journey pattern (normalized 0-1, representing tension/conflict)
    # Shows the characteristic descent into ordeal and return
    CANONICAL_PATTERN = np.array([
        0.3,   # 1. Ordinary World - stable baseline
        0.4,   # 2. Call to Adventure - initial tension
        0.35,  # 3. Refusal - slight dip
        0.45,  # 4. Meeting Mentor - rising hope
        0.5,   # 5. Crossing Threshold - commitment
        0.55,  # 6. Tests - increasing challenge
        0.65,  # 7. Approach - building tension
        0.25,  # 8. Ordeal - crisis/nadir (valley)
        0.6,   # 9. Reward - recovery, prize gained
        0.7,   # 10. Road Back - new challenges
        0.85,  # 11. Resurrection - final test/climax
        0.75,  # 12. Return - transformed equilibrium
    ])

    # Default stage positions (can be adjusted based on detection)
    DEFAULT_POSITIONS = np.array([
        0.00,  # Ordinary World
        0.08,  # Call to Adventure
        0.12,  # Refusal of Call
        0.17,  # Meeting Mentor
        0.25,  # Crossing Threshold
        0.35,  # Tests, Allies, Enemies
        0.45,  # Approach to Cave
        0.55,  # Ordeal
        0.65,  # Reward
        0.75,  # Road Back
        0.85,  # Resurrection
        0.95,  # Return with Elixir
    ])

    def __init__(self, smooth_sigma: float = 3.0):
        """
        Initialize detector.

        Args:
            smooth_sigma: Gaussian smoothing for trajectory preprocessing
        """
        self.smooth_sigma = smooth_sigma
        self.stages = JOURNEY_STAGES

    def _preprocess_trajectory(self, values: np.ndarray) -> np.ndarray:
        """Smooth and normalize trajectory."""
        smoothed = gaussian_filter1d(values, sigma=self.smooth_sigma)
        min_val, max_val = smoothed.min(), smoothed.max()
        if max_val - min_val > 1e-8:
            normalized = (smoothed - min_val) / (max_val - min_val)
        else:
            normalized = np.full_like(smoothed, 0.5)
        return normalized

    def _find_structural_points(self, trajectory: np.ndarray) -> Dict:
        """
        Find key structural points in the trajectory.

        Returns dict with valleys (potential ordeals), peaks (potential
        resurrections), and other structural features.
        """
        n = len(trajectory)

        # Find peaks and valleys
        peaks, peak_props = find_peaks(trajectory, distance=n//8, prominence=0.05)
        valleys, valley_props = find_peaks(-trajectory, distance=n//8, prominence=0.05)

        # Global extrema
        global_max = np.argmax(trajectory)
        global_min = np.argmin(trajectory)

        # Compute derivative for threshold detection
        derivative = np.gradient(trajectory)

        return {
            "peaks": peaks,
            "valleys": valleys,
            "global_max": global_max,
            "global_min": global_min,
            "derivative": derivative,
        }

    def _find_ordeal(self, trajectory: np.ndarray, struct: Dict) -> Tuple[float, float]:
        """
        Find the Ordeal - the supreme crisis point (typically a valley).

        The ordeal is characterized by:
        - A significant low point (valley)
        - Located in the middle section (40-70% of narrative)
        - Represents the "death and rebirth" moment

        Returns:
            ordeal_position: Normalized position of ordeal
            ordeal_depth: How deep the ordeal goes (0-1, 1 = deepest)
        """
        n = len(trajectory)
        valleys = struct["valleys"]

        # Look for valleys in the expected ordeal zone (40-70%)
        ordeal_zone_start = int(0.35 * n)
        ordeal_zone_end = int(0.75 * n)

        # Find valleys in the zone
        zone_valleys = [v for v in valleys if ordeal_zone_start <= v <= ordeal_zone_end]

        if len(zone_valleys) > 0:
            # Find the deepest valley in the zone
            valley_depths = [trajectory[v] for v in zone_valleys]
            deepest_idx = np.argmin(valley_depths)
            ordeal_idx = zone_valleys[deepest_idx]
        elif len(valleys) > 0:
            # Use the global minimum valley
            valley_depths = [trajectory[v] for v in valleys]
            deepest_idx = np.argmin(valley_depths)
            ordeal_idx = valleys[deepest_idx]
        else:
            # Use global minimum
            ordeal_idx = struct["global_min"]

        ordeal_position = ordeal_idx / n

        # Compute depth relative to surrounding context
        window = max(1, n // 10)
        local_max = np.max(trajectory[max(0, ordeal_idx-window):min(n, ordeal_idx+window)])
        ordeal_value = trajectory[ordeal_idx]
        ordeal_depth = local_max - ordeal_value

        return ordeal_position, ordeal_depth

    def _find_resurrection(self, trajectory: np.ndarray, struct: Dict, ordeal_pos: float) -> float:
        """
        Find the Resurrection - the final climactic test after the ordeal.

        The resurrection should be:
        - A peak after the ordeal
        - Near the end of the narrative (75-95%)

        Returns:
            resurrection_position: Normalized position
        """
        n = len(trajectory)
        ordeal_idx = int(ordeal_pos * n)
        peaks = struct["peaks"]

        # Look for peaks after the ordeal
        post_ordeal_peaks = [p for p in peaks if p > ordeal_idx]

        if len(post_ordeal_peaks) > 0:
            # Prefer peaks in the 75-95% range
            late_peaks = [p for p in post_ordeal_peaks if 0.70 * n <= p <= 0.95 * n]

            if late_peaks:
                # Take the highest late peak
                resurrection_idx = late_peaks[np.argmax(trajectory[late_peaks])]
            else:
                # Take the last post-ordeal peak
                resurrection_idx = post_ordeal_peaks[-1]
        else:
            # Default to 85% if no clear peak
            resurrection_idx = int(0.85 * n)

        return resurrection_idx / n

    def _find_thresholds(
        self,
        trajectory: np.ndarray,
        struct: Dict,
        ordeal_pos: float
    ) -> Tuple[float, float]:
        """
        Find the threshold crossings:
        1. First threshold: entering the special world (~20-30%)
        2. Return threshold: leaving the special world (~80-90%)

        These are points of significant change in the derivative.

        Returns:
            first_threshold: Position of entering special world
            return_threshold: Position of returning to ordinary world
        """
        n = len(trajectory)
        derivative = struct["derivative"]
        ordeal_idx = int(ordeal_pos * n)

        # First threshold: significant positive change in first third
        first_third = derivative[:n // 3]
        first_threshold_idx = n // 4  # default

        if len(first_third) > 0:
            # Look for significant derivative spikes
            threshold = np.mean(np.abs(first_third)) + np.std(np.abs(first_third))
            significant = np.where(np.abs(first_third) > threshold)[0]
            if len(significant) > 0:
                first_threshold_idx = significant[-1]  # Last significant change in first third

        first_threshold = max(0.15, min(0.35, first_threshold_idx / n))

        # Return threshold: after ordeal, when trajectory stabilizes
        post_ordeal = derivative[ordeal_idx:]
        return_threshold_local = len(post_ordeal) * 3 // 4  # default

        if len(post_ordeal) > 5:
            # Look for where derivative approaches zero (stabilization)
            abs_deriv = np.abs(post_ordeal)
            stable_threshold = np.mean(abs_deriv) * 0.5
            stable_points = np.where(abs_deriv < stable_threshold)[0]
            if len(stable_points) > 0:
                return_threshold_local = stable_points[-1]

        return_threshold = ordeal_pos + (return_threshold_local / n) * (1 - ordeal_pos)
        return_threshold = max(ordeal_pos + 0.1, min(0.95, return_threshold))

        return first_threshold, return_threshold

    def _map_to_stages(
        self,
        trajectory: np.ndarray,
        ordeal_pos: float,
        resurrection_pos: float,
        threshold_pos: float,
        return_pos: float
    ) -> Tuple[List[float], List[float]]:
        """
        Map trajectory to the 12 Hero's Journey stages.

        Returns:
            stage_positions: Normalized positions for each stage
            stage_values: Trajectory values at each stage
        """
        n = len(trajectory)

        # Build stage positions based on detected key points
        positions = [
            0.0,                                    # 1. Ordinary World
            threshold_pos * 0.4,                    # 2. Call to Adventure
            threshold_pos * 0.6,                    # 3. Refusal
            threshold_pos * 0.8,                    # 4. Meeting Mentor
            threshold_pos,                          # 5. Crossing Threshold
            threshold_pos + (ordeal_pos - threshold_pos) * 0.3,  # 6. Tests
            threshold_pos + (ordeal_pos - threshold_pos) * 0.7,  # 7. Approach
            ordeal_pos,                             # 8. Ordeal
            ordeal_pos + (resurrection_pos - ordeal_pos) * 0.3,  # 9. Reward
            ordeal_pos + (resurrection_pos - ordeal_pos) * 0.7,  # 10. Road Back
            resurrection_pos,                       # 11. Resurrection
            min(0.98, return_pos + 0.05),          # 12. Return with Elixir
        ]

        # Ensure monotonic increase
        for i in range(1, len(positions)):
            if positions[i] <= positions[i-1]:
                positions[i] = positions[i-1] + 0.02

        # Get values at each position
        x_norm = np.linspace(0, 1, n)
        interp_func = interp1d(x_norm, trajectory, kind='linear', bounds_error=False, fill_value='extrapolate')
        stage_values = [float(interp_func(pos)) for pos in positions]

        return positions, stage_values

    def _compute_transformation(self, trajectory: np.ndarray, stage_values: List[float]) -> float:
        """
        Compute the transformation score - how much the hero changed.

        A true hero's journey should show transformation: the hero returns
        different from how they left.

        Returns:
            Transformation score (0-1)
        """
        start_value = stage_values[0]
        end_value = stage_values[-1]

        # Simple difference (positive transformation = higher at end)
        raw_transform = end_value - start_value

        # Also consider if the hero went through trial (ordeal depth)
        ordeal_value = stage_values[7]  # Ordeal stage
        journey_depth = start_value - ordeal_value

        # Combined: transformation includes both change and trial
        if journey_depth > 0.1:  # Went through significant trial
            transformation = 0.5 + 0.5 * min(1.0, raw_transform / 0.5)
        else:
            transformation = 0.3 * min(1.0, abs(raw_transform) / 0.5)

        return max(0.0, min(1.0, transformation))

    def _compute_conformance(
        self,
        trajectory: np.ndarray,
        stage_values: List[float],
        ordeal_depth: float,
        transformation: float
    ) -> Tuple[float, str, List[str]]:
        """
        Compute how well trajectory conforms to Hero's Journey.

        Returns:
            conformance_score: 0-1 measure of fit
            pattern_type: classification
            notes: observations
        """
        notes = []
        stage_vals = np.array(stage_values)

        # 1. Correlation with canonical pattern
        correlation = np.corrcoef(stage_vals, self.CANONICAL_PATTERN)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        # 2. Descent-Ascent structure (key to monomyth)
        # Departure should be higher than ordeal, return should be higher than ordeal
        departure_mean = np.mean(stage_vals[:5])
        initiation_nadir = np.min(stage_vals[5:10])
        return_mean = np.mean(stage_vals[10:])

        has_descent = departure_mean > initiation_nadir + 0.1
        has_ascent = return_mean > initiation_nadir + 0.1

        descent_ascent_score = (
            (0.3 if has_descent else 0.0) +
            (0.3 if has_ascent else 0.0)
        )

        if has_descent:
            notes.append("Clear descent into special world")
        else:
            notes.append("Weak or absent descent")

        if has_ascent:
            notes.append("Clear return/ascent from ordeal")
        else:
            notes.append("Weak or absent return ascent")

        # 3. Ordeal prominence
        if ordeal_depth > 0.3:
            ordeal_score = 0.2
            notes.append(f"Strong ordeal (depth: {ordeal_depth:.2f})")
        elif ordeal_depth > 0.15:
            ordeal_score = 0.1
            notes.append("Moderate ordeal")
        else:
            ordeal_score = 0.0
            notes.append("Weak or absent ordeal")

        # 4. Transformation
        if transformation > 0.6:
            transform_score = 0.2
            notes.append("Clear hero transformation")
        elif transformation > 0.3:
            transform_score = 0.1
            notes.append("Partial transformation")
        else:
            transform_score = 0.0
            notes.append("Limited transformation")

        # 5. Three-act structure presence
        act1_mean = np.mean(stage_vals[:5])    # Departure
        act2_mean = np.mean(stage_vals[5:10])  # Initiation
        act3_mean = np.mean(stage_vals[10:])   # Return

        # Act 2 should be lower (in the special world)
        act_structure = act2_mean < act1_mean and act2_mean < act3_mean
        act_score = 0.1 if act_structure else 0.0

        # Combined score
        conformance = (
            0.2 * max(0, correlation) +
            descent_ascent_score +
            ordeal_score +
            transform_score +
            act_score
        )
        conformance = max(0.0, min(1.0, conformance))

        # Classify pattern type
        if conformance > 0.65 and has_descent and has_ascent:
            pattern_type = "full_journey"
            notes.append("Strong Hero's Journey conformance")
        elif has_descent and not has_ascent:
            pattern_type = "truncated_journey"
            notes.append("Journey without return (tragedy)")
        elif not has_descent and has_ascent:
            pattern_type = "ascent_only"
            notes.append("Rise without descent (wish fulfillment)")
        elif correlation < -0.2:
            pattern_type = "inverted_journey"
            notes.append("Inverted pattern (anti-hero journey)")
        elif conformance > 0.4:
            pattern_type = "partial_journey"
            notes.append("Partial journey elements present")
        else:
            pattern_type = "non_conforming"
            notes.append("Does not follow journey structure")

        return conformance, pattern_type, notes

    def detect(
        self,
        trajectory: np.ndarray,
        trajectory_id: str = "unknown",
        title: str = "Unknown"
    ) -> JourneyMatch:
        """
        Detect Hero's Journey structure in a trajectory.

        Args:
            trajectory: Array of sentiment/arousal values
            trajectory_id: Identifier for the trajectory
            title: Title of the work

        Returns:
            JourneyMatch with detection results
        """
        # Preprocess
        processed = self._preprocess_trajectory(trajectory)

        # Find structural points
        struct = self._find_structural_points(processed)

        # Find key journey points
        ordeal_pos, ordeal_depth = self._find_ordeal(processed, struct)
        resurrection_pos = self._find_resurrection(processed, struct, ordeal_pos)
        threshold_pos, return_pos = self._find_thresholds(processed, struct, ordeal_pos)

        # Map to stages
        stage_positions, stage_values = self._map_to_stages(
            processed, ordeal_pos, resurrection_pos, threshold_pos, return_pos
        )

        # Compute transformation
        transformation = self._compute_transformation(processed, stage_values)

        # Compute conformance
        conformance, pattern_type, notes = self._compute_conformance(
            processed, stage_values, ordeal_depth, transformation
        )

        # Determine which stages were "detected"
        detected_stages = list(range(1, 13))  # All stages by default

        return JourneyMatch(
            trajectory_id=trajectory_id,
            title=title,
            conformance_score=conformance,
            stage_positions=stage_positions,
            stage_values=stage_values,
            detected_stages=detected_stages,
            threshold_crossing=threshold_pos,
            ordeal_position=ordeal_pos,
            ordeal_depth=ordeal_depth,
            resurrection_position=resurrection_pos,
            return_threshold=return_pos,
            transformation_score=transformation,
            pattern_type=pattern_type,
            notes=notes,
        )

    def detect_batch(self, trajectories: List[Dict]) -> List[JourneyMatch]:
        """
        Detect Hero's Journey in multiple trajectories.

        Args:
            trajectories: List of dicts with 'values', 'id', 'title' keys

        Returns:
            List of JourneyMatch results
        """
        results = []
        for traj in trajectories:
            values = np.array(traj.get("values", traj.get("trajectory", [])))
            traj_id = traj.get("id", traj.get("trajectory_id", "unknown"))
            title = traj.get("title", "Unknown")

            match = self.detect(values, traj_id, title)
            results.append(match)

        return results


def analyze_corpus(
    trajectories_dir: Path,
    output_dir: Path,
    trajectory_suffix: str = "_sentiment.json"
):
    """
    Analyze a corpus for Hero's Journey conformance.

    Args:
        trajectories_dir: Directory containing trajectory JSON files
        output_dir: Output directory for results
        trajectory_suffix: Suffix for trajectory files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = HerosJourneyDetector()

    # Load trajectories
    traj_files = list(Path(trajectories_dir).glob(f"*{trajectory_suffix}"))
    traj_files = [f for f in traj_files if f.name != "manifest.json"]

    console.print(f"[blue]Analyzing {len(traj_files)} trajectories for Hero's Journey...[/blue]")

    results = []
    pattern_counts = {}

    for traj_file in traj_files:
        with open(traj_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        values = np.array(data["values"])
        traj_id = data["metadata"].get("source_id", traj_file.stem)
        title = data["metadata"].get("title", "Unknown")

        match = detector.detect(values, traj_id, title)
        results.append(match)

        pattern_counts[match.pattern_type] = pattern_counts.get(match.pattern_type, 0) + 1

    # Sort by conformance
    results.sort(key=lambda x: x.conformance_score, reverse=True)

    # Save detailed results
    results_data = {
        "total_texts": len(results),
        "pattern_distribution": pattern_counts,
        "mean_conformance": float(np.mean([r.conformance_score for r in results])),
        "mean_ordeal_depth": float(np.mean([r.ordeal_depth for r in results])),
        "mean_transformation": float(np.mean([r.transformation_score for r in results])),
        "results": [r.to_dict() for r in results],
    }

    with open(output_dir / "heros_journey_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    # Print summary table
    table = Table(title="Hero's Journey Analysis Results")
    table.add_column("Pattern Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results) if results else 0
        table.add_row(pattern, str(count), f"{pct:.1f}%")

    console.print(table)

    # Top conforming texts
    console.print("\n[bold]Top 5 Hero's Journey Conforming Texts:[/bold]")
    for match in results[:5]:
        console.print(f"  {match.conformance_score:.2f}: {match.title} ({match.pattern_type})")

    console.print(f"\n[green]Results saved to {output_dir}[/green]")

    return results


def visualize_journey(
    match: JourneyMatch,
    trajectory: np.ndarray,
    output_file: Optional[Path] = None
):
    """
    Create visualization of Hero's Journey match.

    Args:
        match: JourneyMatch result
        trajectory: Original trajectory values
        output_file: Path to save figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(16, 6))

    # Left: Trajectory with stage markers
    ax1 = fig.add_subplot(1, 2, 1)

    processed = gaussian_filter1d(trajectory, sigma=3.0)
    normalized = (processed - processed.min()) / (processed.max() - processed.min() + 1e-8)
    x = np.linspace(0, 1, len(normalized))

    ax1.plot(x, normalized, 'b-', linewidth=2, label='Trajectory')

    # Mark stages
    colors_act = {'departure': 'green', 'initiation': 'red', 'return': 'blue'}
    for i, (pos, val) in enumerate(zip(match.stage_positions, match.stage_values)):
        stage = JOURNEY_STAGES[i]
        color = colors_act[stage.act]
        ax1.scatter([pos], [val], color=color, s=80, zorder=5)
        if i % 2 == 0:  # Label every other stage to avoid crowding
            ax1.annotate(f"{i+1}", (pos, val), xytext=(3, 3),
                        textcoords='offset points', fontsize=7)

    # Shade worlds
    ax1.axvspan(0, match.threshold_crossing, alpha=0.1, color='green', label='Ordinary World')
    ax1.axvspan(match.threshold_crossing, match.return_threshold, alpha=0.1, color='red', label='Special World')
    ax1.axvspan(match.return_threshold, 1, alpha=0.1, color='blue', label='Return')

    # Mark ordeal
    ax1.axvline(x=match.ordeal_position, color='darkred', linestyle='--', alpha=0.7, label='Ordeal')
    ax1.axvline(x=match.resurrection_position, color='gold', linestyle='--', alpha=0.7, label='Resurrection')

    ax1.set_xlabel('Narrative Time')
    ax1.set_ylabel('Tension (normalized)')
    ax1.set_title(f'{match.title}\nConformance: {match.conformance_score:.2f} ({match.pattern_type})')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Right: Journey circle diagram
    ax2 = fig.add_subplot(1, 2, 2, aspect='equal')

    # Draw the journey as a circle
    theta = np.linspace(0, 2*np.pi, 13)[:-1]  # 12 stages
    radius = 0.4
    center = (0.5, 0.5)

    # Draw circle
    circle_theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(center[0] + radius*np.cos(circle_theta),
             center[1] + radius*np.sin(circle_theta), 'gray', linewidth=1)

    # Draw horizontal line (ordinary/special world divide)
    ax2.plot([0.1, 0.9], [0.5, 0.5], 'k--', alpha=0.5)
    ax2.text(0.5, 0.92, 'ORDINARY WORLD', ha='center', fontsize=9, fontweight='bold')
    ax2.text(0.5, 0.08, 'SPECIAL WORLD', ha='center', fontsize=9, fontweight='bold')

    # Plot stages
    for i, angle in enumerate(theta):
        x_pos = center[0] + radius * np.cos(angle + np.pi/2)
        y_pos = center[1] + radius * np.sin(angle + np.pi/2)

        stage = JOURNEY_STAGES[i]
        color = colors_act[stage.act]

        # Size based on trajectory value
        size = 50 + 150 * match.stage_values[i]

        ax2.scatter([x_pos], [y_pos], s=size, c=color, alpha=0.7, zorder=5)

        # Label
        label_radius = radius + 0.15
        lx = center[0] + label_radius * np.cos(angle + np.pi/2)
        ly = center[1] + label_radius * np.sin(angle + np.pi/2)
        ax2.text(lx, ly, f"{i+1}", ha='center', va='center', fontsize=8)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title("Hero's Journey Circle\n(Green=Departure, Red=Initiation, Blue=Return)")
    ax2.axis('off')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        console.print(f"[green]Saved journey visualization to {output_file}[/green]")
    else:
        plt.show()

    plt.close()


@click.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True),
              help='Trajectory directory')
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(),
              help='Output directory')
@click.option('--suffix', '-s', default='_sentiment.json', help='Trajectory file suffix')
@click.option('--visualize', '-v', is_flag=True, help='Generate visualizations')
def main(input_dir: str, output_dir: str, suffix: str, visualize: bool):
    """Detect Hero's Journey patterns in trajectories."""
    results = analyze_corpus(Path(input_dir), Path(output_dir), suffix)

    if visualize:
        viz_dir = Path(output_dir) / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        for match in results[:5]:
            traj_file = Path(input_dir) / f"{match.trajectory_id}{suffix}"
            if traj_file.exists():
                with open(traj_file) as f:
                    data = json.load(f)
                trajectory = np.array(data["values"])
                viz_file = viz_dir / f"{match.trajectory_id}_journey.png"
                visualize_journey(match, trajectory, viz_file)


if __name__ == "__main__":
    main()
