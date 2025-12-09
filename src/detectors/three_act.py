"""
Aristotelian Three-Act Structure Detector

Detects the classical three-act dramatic structure in narrative trajectories.
Derived from Aristotle's Poetics, this is the foundational Western dramatic form.

The three acts:
    Act I - PROTASIS (Setup/Beginning): ~25% of narrative
        - Introduces characters, setting, and situation
        - Establishes the world before conflict
        - Ends with the inciting incident / first plot point

    Act II - EPITASIS (Confrontation/Middle): ~50% of narrative
        - Rising action and complications
        - Protagonist pursues goal, faces obstacles
        - Contains midpoint reversal and escalating tension
        - Ends with second plot point / crisis

    Act III - CATASTROPHE (Resolution/End): ~25% of narrative
        - Climax and falling action
        - Resolution of conflicts
        - New equilibrium established

Key characteristics:
- Unity of action (single through-line)
- Clear beginning, middle, and end
- Rising tension through Act II
- Climax near Act III opening
- Proportions roughly 1:2:1

Category-theoretic formalization:
    Define diagram category 3-Act:
        Objects: {I, II, III}
        Morphisms: setup: I→II, resolution: II→III

    A three-act story is a functor T: 3-Act → Narr that:
        - Maps each act to a narrative segment T(i)
        - Maps transitions to plot points
        - Respects dramatic unity
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
class ThreeActStage:
    """Represents an act in the three-act structure."""
    number: int
    name_latin: str
    name_english: str
    expected_proportion: float  # Expected proportion of narrative
    expected_tension: str  # "low", "rising", "peak", "falling"
    description: str


# Define the 3 acts with their characteristics
THREE_ACT_STAGES = [
    ThreeActStage(
        1, "Protasis", "Setup",
        0.25, "low",
        "Introduction of characters, setting, and initial situation"
    ),
    ThreeActStage(
        2, "Epitasis", "Confrontation",
        0.50, "rising",
        "Rising action, complications, and escalating conflict"
    ),
    ThreeActStage(
        3, "Catastrophe", "Resolution",
        0.25, "peak",
        "Climax, falling action, and resolution"
    ),
]


@dataclass
class ThreeActMatch:
    """Result of matching a trajectory to three-act structure."""
    trajectory_id: str
    title: str
    conformance_score: float  # 0-1, how well it matches
    act_boundaries: List[float]  # Normalized positions [0,1] for act transitions
    act_values: List[float]  # Mean trajectory values for each act
    first_plot_point: float  # Position of Act I → II transition
    midpoint: float  # Position of midpoint reversal
    second_plot_point: float  # Position of Act II → III transition
    climax_position: float  # Position of climax
    climax_intensity: float  # Strength of climax
    rising_action_slope: float  # Rate of tension increase in Act II
    pattern_type: str  # "classic_three_act", "front_loaded", "back_loaded", etc.
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "title": self.title,
            "conformance_score": self.conformance_score,
            "act_boundaries": self.act_boundaries,
            "act_values": self.act_values,
            "first_plot_point": self.first_plot_point,
            "midpoint": self.midpoint,
            "second_plot_point": self.second_plot_point,
            "climax_position": self.climax_position,
            "climax_intensity": self.climax_intensity,
            "rising_action_slope": self.rising_action_slope,
            "pattern_type": self.pattern_type,
            "notes": self.notes,
        }


class ThreeActDetector:
    """
    Detects Aristotelian three-act structure in narrative trajectories.

    The detector works by:
    1. Identifying key structural points (plot points, midpoint, climax)
    2. Evaluating proportions against ideal 1:2:1 ratio
    3. Measuring rising action through Act II
    4. Assessing climax placement and intensity
    """

    # Canonical three-act tension pattern (normalized 0-1)
    # Act I: stable low, Act II: rising, Act III: peak then resolution
    CANONICAL_PATTERN = np.array([
        0.3,   # Act I start - low tension
        0.35,  # Act I end - slight rise (inciting incident)
        0.4,   # Act II early - beginning rise
        0.5,   # Midpoint
        0.65,  # Act II late - significant rise
        0.75,  # Second plot point - approaching peak
        0.9,   # Climax - maximum tension
        0.5,   # Resolution - tension release
    ])

    # Ideal proportions
    IDEAL_ACT_I = 0.25
    IDEAL_ACT_II = 0.50
    IDEAL_ACT_III = 0.25

    def __init__(self, smooth_sigma: float = 3.0, proportion_tolerance: float = 0.1):
        """
        Initialize detector.

        Args:
            smooth_sigma: Gaussian smoothing for trajectory preprocessing
            proportion_tolerance: Allowed deviation from ideal proportions
        """
        self.smooth_sigma = smooth_sigma
        self.proportion_tolerance = proportion_tolerance
        self.stages = THREE_ACT_STAGES

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

        Returns dict with:
            - peaks: indices of local maxima
            - valleys: indices of local minima
            - global_max: index of global maximum (potential climax)
            - inflection_points: where derivative changes sign
            - derivative: rate of change array
        """
        n = len(trajectory)

        # Find peaks and valleys
        peaks, peak_props = find_peaks(trajectory, distance=n//10, prominence=0.05)
        valleys, _ = find_peaks(-trajectory, distance=n//10)

        # Global extrema
        global_max = np.argmax(trajectory)
        global_min = np.argmin(trajectory)

        # Compute derivative for slope analysis
        derivative = np.gradient(trajectory)

        # Find inflection points
        second_derivative = np.gradient(derivative)
        inflection_points = []
        for i in range(1, len(second_derivative)):
            if second_derivative[i-1] * second_derivative[i] < 0:
                inflection_points.append(i)

        return {
            "peaks": peaks,
            "valleys": valleys,
            "global_max": global_max,
            "global_min": global_min,
            "inflection_points": np.array(inflection_points),
            "derivative": derivative,
        }

    def _find_plot_points(self, trajectory: np.ndarray, struct: Dict) -> Tuple[float, float, float]:
        """
        Find the two major plot points and midpoint.

        Returns:
            first_plot_point: End of Act I (normalized position)
            midpoint: Center of Act II
            second_plot_point: End of Act II (normalized position)
        """
        n = len(trajectory)
        derivative = struct["derivative"]

        # Default positions based on ideal proportions
        default_pp1 = 0.25
        default_mid = 0.50
        default_pp2 = 0.75

        # Look for significant changes in derivative (plot points)
        # First plot point: significant change in first third
        first_third_deriv = np.abs(derivative[:n//3])
        if len(first_third_deriv) > 0:
            max_change_idx = np.argmax(first_third_deriv)
            if first_third_deriv[max_change_idx] > np.std(derivative) * 0.5:
                first_plot_point = max_change_idx / n
            else:
                first_plot_point = default_pp1
        else:
            first_plot_point = default_pp1

        # Second plot point: look for change in last third leading to climax
        last_third_start = 2 * n // 3
        last_third_deriv = np.abs(derivative[last_third_start:])
        if len(last_third_deriv) > 0:
            max_change_idx = np.argmax(last_third_deriv)
            second_plot_point = (last_third_start + max_change_idx) / n
        else:
            second_plot_point = default_pp2

        # Midpoint: look for reversal or peak in middle section
        mid_section_start = n // 3
        mid_section_end = 2 * n // 3
        mid_section = trajectory[mid_section_start:mid_section_end]

        if len(mid_section) > 0:
            # Look for local extremum
            mid_peaks, _ = find_peaks(mid_section, distance=len(mid_section)//4)
            if len(mid_peaks) > 0:
                midpoint = (mid_section_start + mid_peaks[len(mid_peaks)//2]) / n
            else:
                midpoint = default_mid
        else:
            midpoint = default_mid

        # Ensure logical ordering
        first_plot_point = max(0.15, min(0.35, first_plot_point))
        midpoint = max(first_plot_point + 0.1, min(0.6, midpoint))
        second_plot_point = max(midpoint + 0.1, min(0.85, second_plot_point))

        return first_plot_point, midpoint, second_plot_point

    def _find_climax(self, trajectory: np.ndarray, struct: Dict, second_pp: float) -> Tuple[float, float]:
        """
        Find the climax position and intensity.

        The climax should be near the start of Act III, representing
        the peak of dramatic tension before resolution.

        Returns:
            climax_position: Normalized position of climax
            climax_intensity: Relative intensity (0-1)
        """
        n = len(trajectory)
        global_max = struct["global_max"]
        global_max_pos = global_max / n

        # Climax should ideally be between second plot point and ~85% of narrative
        ideal_climax_zone = (second_pp, min(0.9, second_pp + 0.15))

        # Check if global max is in the expected zone
        if ideal_climax_zone[0] <= global_max_pos <= ideal_climax_zone[1]:
            climax_position = global_max_pos
        else:
            # Look for highest peak in the expected zone
            zone_start = int(ideal_climax_zone[0] * n)
            zone_end = int(ideal_climax_zone[1] * n)
            zone_values = trajectory[zone_start:zone_end]

            if len(zone_values) > 0:
                local_max = np.argmax(zone_values)
                climax_position = (zone_start + local_max) / n
            else:
                climax_position = global_max_pos

        # Compute intensity as normalized value at climax
        climax_idx = int(climax_position * n)
        climax_value = trajectory[min(climax_idx, n-1)]

        # Intensity relative to the narrative range
        min_val = np.min(trajectory)
        max_val = np.max(trajectory)
        if max_val - min_val > 1e-8:
            climax_intensity = (climax_value - min_val) / (max_val - min_val)
        else:
            climax_intensity = 0.5

        return climax_position, climax_intensity

    def _compute_rising_action(self, trajectory: np.ndarray, pp1: float, pp2: float) -> float:
        """
        Compute the slope of rising action through Act II.

        Returns:
            Slope coefficient (positive = proper rising action)
        """
        n = len(trajectory)
        act2_start = int(pp1 * n)
        act2_end = int(pp2 * n)

        act2_values = trajectory[act2_start:act2_end]
        if len(act2_values) < 2:
            return 0.0

        # Fit linear regression to get slope
        x = np.arange(len(act2_values))
        slope = np.polyfit(x, act2_values, 1)[0]

        # Normalize slope to trajectory scale
        return float(slope * len(act2_values))

    def _compute_act_values(
        self,
        trajectory: np.ndarray,
        pp1: float,
        pp2: float
    ) -> Tuple[List[float], List[float]]:
        """
        Compute act boundaries and mean values.

        Returns:
            act_boundaries: [pp1, pp2] (normalized)
            act_values: Mean values for each act
        """
        n = len(trajectory)

        act1_end = int(pp1 * n)
        act2_end = int(pp2 * n)

        act1_values = trajectory[:act1_end] if act1_end > 0 else trajectory[:n//4]
        act2_values = trajectory[act1_end:act2_end] if act2_end > act1_end else trajectory[n//4:3*n//4]
        act3_values = trajectory[act2_end:] if act2_end < n else trajectory[3*n//4:]

        return (
            [pp1, pp2],
            [
                float(np.mean(act1_values)) if len(act1_values) > 0 else 0.5,
                float(np.mean(act2_values)) if len(act2_values) > 0 else 0.5,
                float(np.mean(act3_values)) if len(act3_values) > 0 else 0.5,
            ]
        )

    def _compute_conformance(
        self,
        trajectory: np.ndarray,
        pp1: float,
        pp2: float,
        midpoint: float,
        climax_pos: float,
        climax_intensity: float,
        rising_slope: float,
        act_values: List[float]
    ) -> Tuple[float, str, List[str]]:
        """
        Compute how well trajectory conforms to three-act structure.

        Returns:
            conformance_score: 0-1 measure of fit
            pattern_type: classification of pattern
            notes: observations about the match
        """
        notes = []

        # 1. Proportion scoring (ideal is 25-50-25)
        act1_prop = pp1
        act2_prop = pp2 - pp1
        act3_prop = 1.0 - pp2

        prop_deviation = (
            abs(act1_prop - self.IDEAL_ACT_I) +
            abs(act2_prop - self.IDEAL_ACT_II) +
            abs(act3_prop - self.IDEAL_ACT_III)
        )
        proportion_score = max(0, 1 - prop_deviation / 0.6)

        if prop_deviation < 0.15:
            notes.append("Near-ideal act proportions")
        elif act1_prop > 0.35:
            notes.append("Extended setup (front-loaded)")
        elif act3_prop > 0.35:
            notes.append("Extended resolution (back-loaded)")

        # 2. Rising action scoring
        has_rising_action = rising_slope > 0.05
        rising_score = min(1.0, max(0, rising_slope * 5)) if has_rising_action else 0

        if has_rising_action:
            notes.append(f"Clear rising action (slope: {rising_slope:.2f})")
        else:
            notes.append("Weak or absent rising action")

        # 3. Climax placement scoring
        # Ideal climax is at ~75-85% of narrative
        climax_ideal_distance = abs(climax_pos - 0.80)
        climax_placement_score = max(0, 1 - climax_ideal_distance * 3)

        if 0.70 <= climax_pos <= 0.90:
            notes.append(f"Well-placed climax at {climax_pos:.0%}")
        elif climax_pos < 0.50:
            notes.append("Early climax (unconventional structure)")
        elif climax_pos > 0.90:
            notes.append("Late climax (rushed resolution)")

        # 4. Climax intensity scoring
        intensity_score = climax_intensity
        if climax_intensity > 0.7:
            notes.append("Strong climax intensity")
        elif climax_intensity < 0.4:
            notes.append("Weak climax intensity")

        # 5. Act progression scoring
        # Act III should have higher peak than Act I
        progression_score = 0.5
        if act_values[2] > act_values[0]:
            progression_score = 0.8
            notes.append("Proper tension progression")
        if act_values[1] > act_values[0] and act_values[2] >= act_values[1] * 0.8:
            progression_score = 1.0

        # Combined score
        conformance = (
            0.20 * proportion_score +
            0.25 * rising_score +
            0.20 * climax_placement_score +
            0.15 * intensity_score +
            0.20 * progression_score
        )
        conformance = max(0.0, min(1.0, conformance))

        # Classify pattern type
        if conformance > 0.65 and has_rising_action:
            if abs(act1_prop - 0.25) < 0.1 and abs(act3_prop - 0.25) < 0.1:
                pattern_type = "classic_three_act"
            else:
                pattern_type = "modified_three_act"
        elif act1_prop > 0.35:
            pattern_type = "front_loaded"
        elif act3_prop > 0.35:
            pattern_type = "back_loaded"
        elif not has_rising_action:
            pattern_type = "flat_structure"
        elif climax_pos < 0.5:
            pattern_type = "inverted_structure"
        else:
            pattern_type = "non_conforming"

        return conformance, pattern_type, notes

    def detect(
        self,
        trajectory: np.ndarray,
        trajectory_id: str = "unknown",
        title: str = "Unknown"
    ) -> ThreeActMatch:
        """
        Detect three-act structure in a trajectory.

        Args:
            trajectory: Array of sentiment/arousal values
            trajectory_id: Identifier for the trajectory
            title: Title of the work

        Returns:
            ThreeActMatch with detection results
        """
        # Preprocess
        processed = self._preprocess_trajectory(trajectory)

        # Find structural points
        struct = self._find_structural_points(processed)

        # Find plot points and midpoint
        pp1, midpoint, pp2 = self._find_plot_points(processed, struct)

        # Find climax
        climax_pos, climax_intensity = self._find_climax(processed, struct, pp2)

        # Compute rising action slope
        rising_slope = self._compute_rising_action(processed, pp1, pp2)

        # Compute act boundaries and values
        act_boundaries, act_values = self._compute_act_values(processed, pp1, pp2)

        # Compute conformance
        conformance, pattern_type, notes = self._compute_conformance(
            processed, pp1, pp2, midpoint, climax_pos,
            climax_intensity, rising_slope, act_values
        )

        return ThreeActMatch(
            trajectory_id=trajectory_id,
            title=title,
            conformance_score=conformance,
            act_boundaries=act_boundaries,
            act_values=act_values,
            first_plot_point=pp1,
            midpoint=midpoint,
            second_plot_point=pp2,
            climax_position=climax_pos,
            climax_intensity=climax_intensity,
            rising_action_slope=rising_slope,
            pattern_type=pattern_type,
            notes=notes,
        )

    def detect_batch(self, trajectories: List[Dict]) -> List[ThreeActMatch]:
        """
        Detect three-act structure in multiple trajectories.

        Args:
            trajectories: List of dicts with 'values', 'id', 'title' keys

        Returns:
            List of ThreeActMatch results
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
    Analyze a corpus for three-act structure conformance.

    Args:
        trajectories_dir: Directory containing trajectory JSON files
        output_dir: Output directory for results
        trajectory_suffix: Suffix for trajectory files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = ThreeActDetector()

    # Load trajectories
    traj_files = list(Path(trajectories_dir).glob(f"*{trajectory_suffix}"))
    traj_files = [f for f in traj_files if f.name != "manifest.json"]

    console.print(f"[blue]Analyzing {len(traj_files)} trajectories for Three-Act Structure...[/blue]")

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
        "mean_climax_position": float(np.mean([r.climax_position for r in results])),
        "mean_rising_slope": float(np.mean([r.rising_action_slope for r in results])),
        "results": [r.to_dict() for r in results],
    }

    with open(output_dir / "three_act_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    # Print summary table
    table = Table(title="Three-Act Structure Analysis Results")
    table.add_column("Pattern Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results) if results else 0
        table.add_row(pattern, str(count), f"{pct:.1f}%")

    console.print(table)

    # Top conforming texts
    console.print("\n[bold]Top 5 Three-Act Structure Conforming Texts:[/bold]")
    for match in results[:5]:
        console.print(f"  {match.conformance_score:.2f}: {match.title} ({match.pattern_type})")

    console.print(f"\n[green]Results saved to {output_dir}[/green]")

    return results


@click.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True),
              help='Trajectory directory')
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(),
              help='Output directory')
@click.option('--suffix', '-s', default='_sentiment.json', help='Trajectory file suffix')
def main(input_dir: str, output_dir: str, suffix: str):
    """Detect Aristotelian three-act structure patterns in trajectories."""
    analyze_corpus(Path(input_dir), Path(output_dir), suffix)


if __name__ == "__main__":
    main()
