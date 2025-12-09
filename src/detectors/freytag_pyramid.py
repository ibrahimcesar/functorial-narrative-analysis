"""
Freytag's Pyramid Detector

Detects the five-act dramatic structure described by Gustav Freytag in
"Die Technik des Dramas" (1863). This structure was derived from analysis
of classical Greek and Shakespearean drama.

The five acts:
    1. EXPOSITION (Einleitung): ~10-15% of narrative
        - Introduction of setting, characters, and situation
        - Establishes the status quo
        - Contains the "exciting force" (erregendes Moment)

    2. RISING ACTION (Steigerung): ~25-30% of narrative
        - Complications and obstacles introduced
        - Tension builds progressively
        - Series of crises leading toward climax

    3. CLIMAX (Höhepunkt): ~5-10% of narrative
        - The turning point / peripeteia
        - Maximum dramatic tension
        - The point of no return

    4. FALLING ACTION (Fall/Umkehr): ~25-30% of narrative
        - Consequences unfold
        - Movement toward resolution
        - Contains the "force of final suspense" (Moment der letzten Spannung)

    5. DENOUEMENT/CATASTROPHE (Katastrophe): ~10-15% of narrative
        - Final resolution
        - New equilibrium established
        - In tragedy: catastrophe; in comedy: resolution

Geometric representation (the "pyramid"):

                    CLIMAX
                      /\\
                     /  \\
                    /    \\
          RISING  /      \\ FALLING
          ACTION /        \\ ACTION
                /          \\
               /            \\
    EXPOSITION              DENOUEMENT
    ──────────────────────────────────

Category-theoretic formalization:
    Define diagram category Freytag-5:
        Objects: {Exp, Rise, Climax, Fall, Den}
        Morphisms: excite: Exp→Rise, peak: Rise→Climax,
                   reverse: Climax→Fall, resolve: Fall→Den

    A Freytag story is a functor F: Freytag-5 → Narr that:
        - Maps each phase to a narrative segment
        - Maps transitions to dramatic beats
        - The climax object maps to a single point (not a range)
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
class FreytagStage:
    """Represents a phase in Freytag's Pyramid."""
    number: int
    name_german: str
    name_english: str
    expected_proportion: Tuple[float, float]  # Min-max proportion
    expected_tension: str  # "baseline", "rising", "peak", "falling", "resolved"
    description: str


# Define the 5 phases
FREYTAG_STAGES = [
    FreytagStage(
        1, "Einleitung", "Exposition",
        (0.10, 0.15), "baseline",
        "Introduction of setting, characters, and initial situation"
    ),
    FreytagStage(
        2, "Steigerung", "Rising Action",
        (0.25, 0.30), "rising",
        "Complications build, tension increases progressively"
    ),
    FreytagStage(
        3, "Höhepunkt", "Climax",
        (0.05, 0.10), "peak",
        "Turning point, maximum tension, point of no return"
    ),
    FreytagStage(
        4, "Fall", "Falling Action",
        (0.25, 0.30), "falling",
        "Consequences unfold, movement toward resolution"
    ),
    FreytagStage(
        5, "Katastrophe", "Denouement",
        (0.10, 0.15), "resolved",
        "Final resolution, new equilibrium established"
    ),
]


@dataclass
class FreytagMatch:
    """Result of matching a trajectory to Freytag's Pyramid."""
    trajectory_id: str
    title: str
    conformance_score: float  # 0-1, how well it matches
    phase_boundaries: List[float]  # Normalized positions for phase transitions
    phase_values: List[float]  # Mean trajectory values for each phase
    climax_position: float  # Position of the climax peak
    climax_value: float  # Value at climax
    pyramid_symmetry: float  # How symmetric the rise/fall is
    exciting_force_position: float  # Position of inciting incident
    final_suspense_position: float  # Position of last tension spike
    ascending_slope: float  # Slope of rising action
    descending_slope: float  # Slope of falling action
    pattern_type: str  # "classic_pyramid", "asymmetric", "double_peak", etc.
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "title": self.title,
            "conformance_score": self.conformance_score,
            "phase_boundaries": self.phase_boundaries,
            "phase_values": self.phase_values,
            "climax_position": self.climax_position,
            "climax_value": self.climax_value,
            "pyramid_symmetry": self.pyramid_symmetry,
            "exciting_force_position": self.exciting_force_position,
            "final_suspense_position": self.final_suspense_position,
            "ascending_slope": self.ascending_slope,
            "descending_slope": self.descending_slope,
            "pattern_type": self.pattern_type,
            "notes": self.notes,
        }


class FreytagPyramidDetector:
    """
    Detects Freytag's Pyramid structure in narrative trajectories.

    The detector identifies:
    1. The pyramidal shape (rise → peak → fall)
    2. The five distinct phases
    3. Symmetry between rising and falling action
    4. Key dramatic moments (exciting force, climax, final suspense)
    """

    # Canonical pyramid pattern (normalized 0-1)
    CANONICAL_PATTERN = np.array([
        0.2,   # Exposition start
        0.25,  # Exposition end / exciting force
        0.4,   # Rising action midpoint
        0.6,   # Approaching climax
        0.9,   # Climax peak
        0.7,   # Post-climax, beginning fall
        0.5,   # Falling action midpoint
        0.35,  # Final suspense moment
        0.2,   # Denouement
    ])

    def __init__(self, smooth_sigma: float = 3.0):
        """
        Initialize detector.

        Args:
            smooth_sigma: Gaussian smoothing for trajectory preprocessing
        """
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
        """
        Find the climax (highest point) in the trajectory.

        Returns:
            climax_idx: Index of the climax
            climax_value: Normalized value at climax
        """
        n = len(trajectory)

        # Find all significant peaks
        peaks, properties = find_peaks(
            trajectory,
            distance=n // 8,
            prominence=0.1
        )

        if len(peaks) == 0:
            # No clear peaks - use global max
            climax_idx = np.argmax(trajectory)
        else:
            # Prefer peaks in the middle section (40-70% of narrative)
            mid_start = int(0.35 * n)
            mid_end = int(0.75 * n)

            mid_peaks = [p for p in peaks if mid_start <= p <= mid_end]

            if mid_peaks:
                # Take highest peak in middle section
                climax_idx = mid_peaks[np.argmax(trajectory[mid_peaks])]
            else:
                # Take global maximum peak
                climax_idx = peaks[np.argmax(trajectory[peaks])]

        climax_value = float(trajectory[climax_idx])
        return climax_idx, climax_value

    def _find_exciting_force(self, trajectory: np.ndarray, climax_idx: int) -> float:
        """
        Find the "exciting force" - the inciting incident that begins rising action.

        This is typically a significant change in the first 25% of the narrative.

        Returns:
            Position of exciting force (normalized)
        """
        n = len(trajectory)
        first_quarter = trajectory[:n // 4]

        # Look for first significant derivative change
        derivative = np.gradient(first_quarter)
        abs_deriv = np.abs(derivative)

        # Find points where derivative exceeds threshold
        threshold = np.mean(abs_deriv) + 0.5 * np.std(abs_deriv)
        significant_points = np.where(abs_deriv > threshold)[0]

        if len(significant_points) > 0:
            exciting_idx = significant_points[0]
        else:
            # Default to ~10-15% of narrative
            exciting_idx = n // 8

        return exciting_idx / n

    def _find_final_suspense(self, trajectory: np.ndarray, climax_idx: int) -> float:
        """
        Find the "moment of final suspense" in the falling action.

        This is often a brief uptick in tension before final resolution.

        Returns:
            Position of final suspense moment (normalized)
        """
        n = len(trajectory)
        post_climax = trajectory[climax_idx:]

        if len(post_climax) < 5:
            return climax_idx / n + 0.1

        # Look for local peaks in the falling section
        local_peaks, _ = find_peaks(post_climax, distance=len(post_climax) // 4)

        if len(local_peaks) > 0:
            # Take the last significant peak
            final_suspense_local = local_peaks[-1]
            final_suspense_idx = climax_idx + final_suspense_local
        else:
            # Default to ~80% of falling action
            final_suspense_idx = climax_idx + int(0.6 * len(post_climax))

        return min(final_suspense_idx / n, 0.95)

    def _compute_slopes(
        self,
        trajectory: np.ndarray,
        exciting_pos: float,
        climax_idx: int
    ) -> Tuple[float, float]:
        """
        Compute the ascending and descending slopes.

        Returns:
            ascending_slope: Slope of rising action
            descending_slope: Slope of falling action (negative for proper descent)
        """
        n = len(trajectory)
        exciting_idx = int(exciting_pos * n)

        # Rising action: from exciting force to climax
        rising_section = trajectory[exciting_idx:climax_idx+1]
        if len(rising_section) > 1:
            x_rise = np.arange(len(rising_section))
            ascending_slope = np.polyfit(x_rise, rising_section, 1)[0]
        else:
            ascending_slope = 0.0

        # Falling action: from climax to end
        falling_section = trajectory[climax_idx:]
        if len(falling_section) > 1:
            x_fall = np.arange(len(falling_section))
            descending_slope = np.polyfit(x_fall, falling_section, 1)[0]
        else:
            descending_slope = 0.0

        # Normalize by section length
        ascending_slope = ascending_slope * len(rising_section) if len(rising_section) > 0 else 0
        descending_slope = descending_slope * len(falling_section) if len(falling_section) > 0 else 0

        return float(ascending_slope), float(descending_slope)

    def _compute_symmetry(
        self,
        trajectory: np.ndarray,
        climax_idx: int,
        ascending_slope: float,
        descending_slope: float
    ) -> float:
        """
        Compute the symmetry of the pyramid.

        Perfect symmetry = 1.0, complete asymmetry = 0.0

        Returns:
            Symmetry score (0-1)
        """
        n = len(trajectory)

        # Method 1: Compare absolute slopes
        if abs(ascending_slope) > 0.01 and abs(descending_slope) > 0.01:
            slope_ratio = min(abs(ascending_slope), abs(descending_slope)) / \
                         max(abs(ascending_slope), abs(descending_slope))
        else:
            slope_ratio = 0.5

        # Method 2: Compare lengths of rise vs fall
        rise_length = climax_idx
        fall_length = n - climax_idx
        length_ratio = min(rise_length, fall_length) / max(rise_length, fall_length) if max(rise_length, fall_length) > 0 else 0.5

        # Method 3: Compare start and end values
        value_symmetry = 1.0 - abs(trajectory[0] - trajectory[-1])

        # Combined symmetry
        symmetry = 0.4 * slope_ratio + 0.4 * length_ratio + 0.2 * value_symmetry
        return float(max(0, min(1, symmetry)))

    def _map_to_phases(
        self,
        trajectory: np.ndarray,
        climax_idx: int,
        exciting_pos: float
    ) -> Tuple[List[float], List[float]]:
        """
        Map trajectory to 5 Freytag phases.

        Returns:
            phase_boundaries: Positions of phase transitions
            phase_values: Mean values for each phase
        """
        n = len(trajectory)
        climax_pos = climax_idx / n

        # Calculate boundaries based on climax position
        # Exposition: 0 to exciting force
        # Rising action: exciting force to climax
        # Climax: narrow band around peak
        # Falling action: after climax to near end
        # Denouement: final section

        expo_end = exciting_pos
        rise_end = climax_pos - 0.02
        climax_end = climax_pos + 0.02
        fall_end = climax_end + (1.0 - climax_end) * 0.75

        phase_boundaries = [expo_end, rise_end, climax_end, fall_end]

        # Compute mean values for each phase
        boundaries = [0.0] + phase_boundaries + [1.0]
        phase_values = []

        for i in range(5):
            start = int(boundaries[i] * n)
            end = int(boundaries[i + 1] * n)
            if end > start:
                phase_values.append(float(np.mean(trajectory[start:end])))
            else:
                phase_values.append(float(trajectory[min(start, n-1)]))

        return phase_boundaries, phase_values

    def _compute_conformance(
        self,
        trajectory: np.ndarray,
        climax_idx: int,
        climax_value: float,
        symmetry: float,
        ascending_slope: float,
        descending_slope: float,
        phase_values: List[float]
    ) -> Tuple[float, str, List[str]]:
        """
        Compute how well trajectory conforms to Freytag's Pyramid.

        Returns:
            conformance_score: 0-1 measure of fit
            pattern_type: classification
            notes: observations
        """
        notes = []
        n = len(trajectory)
        climax_pos = climax_idx / n

        # 1. Pyramid shape score
        has_rise = ascending_slope > 0.05
        has_fall = descending_slope < -0.05

        if has_rise and has_fall:
            pyramid_score = 0.9
            notes.append("Clear pyramidal shape")
        elif has_rise and not has_fall:
            pyramid_score = 0.4
            notes.append("Rising action without clear fall")
        elif not has_rise and has_fall:
            pyramid_score = 0.4
            notes.append("Falling action without clear rise")
        else:
            pyramid_score = 0.2
            notes.append("No clear pyramid shape")

        # 2. Climax position score (ideal is 40-60% of narrative)
        if 0.40 <= climax_pos <= 0.60:
            climax_score = 1.0
            notes.append(f"Well-centered climax at {climax_pos:.0%}")
        elif 0.30 <= climax_pos <= 0.70:
            climax_score = 0.7
            notes.append(f"Acceptable climax position at {climax_pos:.0%}")
        else:
            climax_score = 0.3
            if climax_pos < 0.30:
                notes.append("Early climax (compressed rising action)")
            else:
                notes.append("Late climax (compressed falling action)")

        # 3. Climax prominence score
        baseline = (trajectory[0] + trajectory[-1]) / 2
        prominence = climax_value - baseline
        if prominence > 0.4:
            prominence_score = 1.0
            notes.append("Strong climax prominence")
        elif prominence > 0.2:
            prominence_score = 0.7
        else:
            prominence_score = 0.3
            notes.append("Weak climax prominence")

        # 4. Symmetry score
        if symmetry > 0.7:
            notes.append("Good pyramid symmetry")
        elif symmetry < 0.4:
            notes.append("Asymmetric structure")

        # 5. Phase progression score
        # Should go: low → rising → peak → falling → low
        expo_val = phase_values[0]
        rise_val = phase_values[1]
        climax_val = phase_values[2]
        fall_val = phase_values[3]
        denou_val = phase_values[4]

        progression_correct = (
            rise_val > expo_val + 0.05 and
            climax_val > rise_val and
            climax_val > fall_val and
            fall_val > denou_val - 0.1
        )
        progression_score = 0.8 if progression_correct else 0.3

        # Combined score
        conformance = (
            0.30 * pyramid_score +
            0.20 * climax_score +
            0.15 * prominence_score +
            0.15 * symmetry +
            0.20 * progression_score
        )
        conformance = max(0.0, min(1.0, conformance))

        # Classify pattern
        if conformance > 0.65 and has_rise and has_fall:
            if symmetry > 0.6:
                pattern_type = "classic_pyramid"
            else:
                pattern_type = "asymmetric_pyramid"
        elif has_rise and has_fall:
            if climax_pos < 0.35:
                pattern_type = "left_skewed"
            elif climax_pos > 0.65:
                pattern_type = "right_skewed"
            else:
                pattern_type = "weak_pyramid"
        elif has_rise:
            pattern_type = "rising_only"
        elif has_fall:
            pattern_type = "falling_only"
        else:
            pattern_type = "non_conforming"

        # Check for double-peak
        peaks, _ = find_peaks(trajectory, distance=n//6, prominence=0.15)
        if len(peaks) >= 2:
            notes.append(f"Multiple peaks detected ({len(peaks)})")
            if conformance < 0.5:
                pattern_type = "multi_peak"

        return conformance, pattern_type, notes

    def detect(
        self,
        trajectory: np.ndarray,
        trajectory_id: str = "unknown",
        title: str = "Unknown"
    ) -> FreytagMatch:
        """
        Detect Freytag's Pyramid structure in a trajectory.

        Args:
            trajectory: Array of sentiment/arousal values
            trajectory_id: Identifier for the trajectory
            title: Title of the work

        Returns:
            FreytagMatch with detection results
        """
        # Preprocess
        processed = self._preprocess_trajectory(trajectory)

        # Find climax
        climax_idx, climax_value = self._find_climax(processed)
        climax_position = climax_idx / len(processed)

        # Find key moments
        exciting_pos = self._find_exciting_force(processed, climax_idx)
        final_suspense_pos = self._find_final_suspense(processed, climax_idx)

        # Compute slopes
        ascending_slope, descending_slope = self._compute_slopes(
            processed, exciting_pos, climax_idx
        )

        # Compute symmetry
        symmetry = self._compute_symmetry(
            processed, climax_idx, ascending_slope, descending_slope
        )

        # Map to phases
        phase_boundaries, phase_values = self._map_to_phases(
            processed, climax_idx, exciting_pos
        )

        # Compute conformance
        conformance, pattern_type, notes = self._compute_conformance(
            processed, climax_idx, climax_value, symmetry,
            ascending_slope, descending_slope, phase_values
        )

        return FreytagMatch(
            trajectory_id=trajectory_id,
            title=title,
            conformance_score=conformance,
            phase_boundaries=phase_boundaries,
            phase_values=phase_values,
            climax_position=climax_position,
            climax_value=climax_value,
            pyramid_symmetry=symmetry,
            exciting_force_position=exciting_pos,
            final_suspense_position=final_suspense_pos,
            ascending_slope=ascending_slope,
            descending_slope=descending_slope,
            pattern_type=pattern_type,
            notes=notes,
        )

    def detect_batch(self, trajectories: List[Dict]) -> List[FreytagMatch]:
        """
        Detect Freytag's Pyramid in multiple trajectories.

        Args:
            trajectories: List of dicts with 'values', 'id', 'title' keys

        Returns:
            List of FreytagMatch results
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
    Analyze a corpus for Freytag's Pyramid conformance.

    Args:
        trajectories_dir: Directory containing trajectory JSON files
        output_dir: Output directory for results
        trajectory_suffix: Suffix for trajectory files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = FreytagPyramidDetector()

    # Load trajectories
    traj_files = list(Path(trajectories_dir).glob(f"*{trajectory_suffix}"))
    traj_files = [f for f in traj_files if f.name != "manifest.json"]

    console.print(f"[blue]Analyzing {len(traj_files)} trajectories for Freytag's Pyramid...[/blue]")

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
        "mean_symmetry": float(np.mean([r.pyramid_symmetry for r in results])),
        "results": [r.to_dict() for r in results],
    }

    with open(output_dir / "freytag_pyramid_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    # Print summary table
    table = Table(title="Freytag's Pyramid Analysis Results")
    table.add_column("Pattern Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results) if results else 0
        table.add_row(pattern, str(count), f"{pct:.1f}%")

    console.print(table)

    # Top conforming texts
    console.print("\n[bold]Top 5 Freytag's Pyramid Conforming Texts:[/bold]")
    for match in results[:5]:
        console.print(f"  {match.conformance_score:.2f}: {match.title} ({match.pattern_type})")

    console.print(f"\n[green]Results saved to {output_dir}[/green]")

    return results


def visualize_pyramid(
    match: FreytagMatch,
    trajectory: np.ndarray,
    output_file: Optional[Path] = None
):
    """
    Create visualization of Freytag's Pyramid match.

    Args:
        match: FreytagMatch result
        trajectory: Original trajectory values
        output_file: Path to save figure
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Trajectory with phase markers
    processed = gaussian_filter1d(trajectory, sigma=3.0)
    normalized = (processed - processed.min()) / (processed.max() - processed.min() + 1e-8)
    x = np.linspace(0, 1, len(normalized))

    ax1.plot(x, normalized, 'b-', linewidth=2, label='Trajectory')

    # Mark phases
    colors = ['green', 'orange', 'red', 'purple', 'blue']
    phase_names = ['Exposition', 'Rising Action', 'Climax', 'Falling Action', 'Denouement']
    boundaries = [0.0] + match.phase_boundaries + [1.0]

    for i in range(5):
        ax1.axvspan(boundaries[i], boundaries[i+1], alpha=0.2, color=colors[i], label=phase_names[i])

    # Mark climax
    ax1.axvline(x=match.climax_position, color='red', linestyle='--', linewidth=2, label='Climax')

    ax1.set_xlabel('Narrative Time')
    ax1.set_ylabel('Tension (normalized)')
    ax1.set_title(f'{match.title}\nConformance: {match.conformance_score:.2f} ({match.pattern_type})')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: Pyramid diagram
    pyramid_x = [0, 0.5, 1]
    pyramid_y = [0, 1, 0]
    ax2.fill(pyramid_x, pyramid_y, alpha=0.3, color='blue')
    ax2.plot(pyramid_x, pyramid_y, 'b-', linewidth=2)

    # Plot actual trajectory shape on pyramid
    actual_climax = match.climax_position
    actual_x = [0, actual_climax, 1]
    actual_y = [match.phase_values[0], match.climax_value, match.phase_values[4]]
    ax2.plot(actual_x, actual_y, 'r-', linewidth=2, label='Actual shape')

    # Labels
    ax2.text(0, -0.1, 'Exposition', ha='center', fontsize=9)
    ax2.text(0.5, 1.05, 'Climax', ha='center', fontsize=9, fontweight='bold')
    ax2.text(1, -0.1, 'Denouement', ha='center', fontsize=9)
    ax2.text(0.25, 0.5, 'Rising\nAction', ha='center', fontsize=8, rotation=45)
    ax2.text(0.75, 0.5, 'Falling\nAction', ha='center', fontsize=8, rotation=-45)

    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_title("Freytag's Pyramid")
    ax2.legend(loc='upper right')
    ax2.axis('off')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        console.print(f"[green]Saved pyramid visualization to {output_file}[/green]")
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
    """Detect Freytag's Pyramid patterns in trajectories."""
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
                viz_file = viz_dir / f"{match.trajectory_id}_pyramid.png"
                visualize_pyramid(match, trajectory, viz_file)


if __name__ == "__main__":
    main()
