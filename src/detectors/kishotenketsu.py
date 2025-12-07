"""
Kishōtenketsu (起承転結) Detector

Detects the 4-act East Asian narrative structure in trajectories.
Unlike Western conflict-driven narrative, kishōtenketsu creates
tension through juxtaposition and reframing rather than conflict.

The four acts:
    起 (Ki) - Introduction: Establish setting and characters
    承 (Shō) - Development: Expand on the introduction
    転 (Ten) - Twist/Turn: Unexpected element, change of perspective
    結 (Ketsu) - Reconciliation: Tie together, resolve through reframing

Key characteristics:
- No central conflict required
- Tension from juxtaposition, not opposition
- The "ten" (twist) reframes what came before
- Resolution through understanding, not victory

Trajectory signatures:
- Flat or gradual sentiment through Ki-Shō (acts 1-2)
- Spike or sudden change at Ten (act 3)
- Return to baseline or new equilibrium at Ketsu (act 4)

Contrast with Harmon Circle:
- Harmon: descent → nadir → ascent (valley shape)
- Kishōtenketsu: plateau → spike → resolution (spike shape)
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
import click
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class KishotenketsuStage:
    """Represents a stage in kishōtenketsu."""
    name_jp: str
    name_en: str
    description: str
    expected_pattern: str  # "stable", "spike", "resolution"


KISHOTENKETSU_STAGES = [
    KishotenketsuStage("起", "Ki (Introduction)",
                       "Establish setting, characters, situation", "stable"),
    KishotenketsuStage("承", "Shō (Development)",
                       "Expand and develop the introduction", "stable"),
    KishotenketsuStage("転", "Ten (Twist)",
                       "Unexpected element, change of perspective", "spike"),
    KishotenketsuStage("結", "Ketsu (Reconciliation)",
                       "Tie together, new understanding", "resolution"),
]


@dataclass
class KishotenketsuMatch:
    """Result of matching a trajectory to kishōtenketsu."""
    trajectory_id: str
    title: str
    conformance_score: float
    stage_boundaries: List[float]  # Normalized positions [0,1] for act boundaries
    stage_values: List[float]  # Mean trajectory values for each act
    ten_position: float  # Position of the twist (0-1)
    ten_magnitude: float  # How strong the twist is
    has_twist: bool  # Whether a clear twist was detected
    twist_type: str  # "spike_up", "spike_down", "shift", "none"
    pattern_type: str  # "classic", "subtle", "western_hybrid", "non_conforming"
    stability_score: float  # How stable Ki-Shō are
    notes: List[str] = field(default_factory=list)

    @property
    def is_kishotenketsu(self) -> bool:
        """Whether this text conforms to kishōtenketsu structure."""
        return self.pattern_type in ("classic_kishotenketsu", "subtle_kishotenketsu")

    @property
    def twist_strength(self) -> float:
        """Alias for ten_magnitude for consistency."""
        return self.ten_magnitude

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "title": self.title,
            "conformance_score": self.conformance_score,
            "stage_boundaries": self.stage_boundaries,
            "stage_values": self.stage_values,
            "ten_position": self.ten_position,
            "ten_magnitude": self.ten_magnitude,
            "has_twist": self.has_twist,
            "twist_type": self.twist_type,
            "pattern_type": self.pattern_type,
            "stability_score": self.stability_score,
            "notes": self.notes,
        }


class KishotenketsuDetector:
    """
    Detects Kishōtenketsu (起承転結) structure in narrative trajectories.

    The detector identifies:
    1. Stable introduction/development (Ki-Shō)
    2. A twist or perspective shift (Ten)
    3. Resolution/reconciliation (Ketsu)

    Unlike Harmon Circle which looks for descent-ascent,
    this looks for stability-spike-resolution patterns.
    """

    def __init__(
        self,
        smooth_sigma: float = 3.0,
        spike_threshold: float = 1.5,  # Z-score threshold for twist detection
    ):
        """
        Initialize detector.

        Args:
            smooth_sigma: Gaussian smoothing for preprocessing
            spike_threshold: Z-score threshold for detecting the Ten (twist)
        """
        self.smooth_sigma = smooth_sigma
        self.spike_threshold = spike_threshold
        self.stages = KISHOTENKETSU_STAGES

    def _preprocess_trajectory(self, values: np.ndarray) -> np.ndarray:
        """Smooth and normalize trajectory."""
        smoothed = gaussian_filter1d(values, sigma=self.smooth_sigma)
        min_val, max_val = smoothed.min(), smoothed.max()
        if max_val - min_val > 1e-8:
            normalized = (smoothed - min_val) / (max_val - min_val)
        else:
            normalized = np.full_like(smoothed, 0.5)
        return normalized

    def _find_twist_point(self, trajectory: np.ndarray) -> Tuple[int, float, str]:
        """
        Find the Ten (twist) point in the trajectory.

        The twist is characterized by:
        - A sudden change in value
        - A local extremum (peak or valley)
        - High derivative magnitude

        Returns:
            twist_idx: Index of twist point
            magnitude: Strength of the twist (z-score)
            twist_type: "spike_up", "spike_down", "shift", or "none"
        """
        n = len(trajectory)

        # Compute derivative (rate of change)
        derivative = np.gradient(trajectory)
        derivative_z = zscore(np.abs(derivative))

        # Find peaks in derivative magnitude (sudden changes)
        peaks, properties = find_peaks(
            np.abs(derivative),
            height=np.std(derivative) * 0.5,
            distance=n // 8,
        )

        if len(peaks) == 0:
            # No clear twist - check for gradual shift
            first_half = np.mean(trajectory[:n//2])
            second_half = np.mean(trajectory[n//2:])
            shift = abs(second_half - first_half)

            if shift > 0.15:
                return n // 2, shift * 3, "shift"
            return n // 2, 0.0, "none"

        # Find the most significant change (highest derivative)
        # Prefer peaks in the second or third quarter (typical Ten position)
        scores = []
        for peak in peaks:
            position_score = 1.0
            # Prefer middle-to-late position (25%-75% of narrative)
            if 0.25 < peak / n < 0.75:
                position_score = 1.5
            if 0.4 < peak / n < 0.6:
                position_score = 2.0

            magnitude = np.abs(derivative[peak])
            scores.append((peak, magnitude * position_score, derivative[peak]))

        # Select best twist point
        best_peak, _, deriv_value = max(scores, key=lambda x: x[1])

        # Determine twist type
        if deriv_value > 0:
            twist_type = "spike_up"
        else:
            twist_type = "spike_down"

        # Calculate magnitude as z-score of the change
        local_window = max(1, n // 10)
        local_mean = np.mean(trajectory[max(0, best_peak-local_window):min(n, best_peak+local_window)])
        local_std = np.std(trajectory[max(0, best_peak-local_window):min(n, best_peak+local_window)])

        if local_std > 0:
            magnitude = abs(trajectory[best_peak] - local_mean) / local_std
        else:
            magnitude = 0.0

        return best_peak, magnitude, twist_type

    def _compute_stability(self, trajectory: np.ndarray, end_idx: int) -> float:
        """
        Compute stability score for Ki-Shō section.

        High stability = consistent values, low variation
        Low stability = fluctuating values (more Western-style)
        """
        if end_idx < 2:
            return 0.5

        section = trajectory[:end_idx]
        cv = np.std(section) / (np.mean(section) + 1e-8)

        # Convert to 0-1 score where 1 = very stable
        stability = 1.0 / (1.0 + cv * 5)

        return stability

    def _map_to_stages(
        self,
        trajectory: np.ndarray,
        twist_idx: int
    ) -> Tuple[List[float], List[float]]:
        """
        Map trajectory to 4 kishōtenketsu stages.

        Args:
            trajectory: Processed trajectory
            twist_idx: Index of the twist point

        Returns:
            stage_boundaries: Normalized positions for act boundaries
            stage_values: Mean values for each act
        """
        n = len(trajectory)

        # Standard division: Ki=25%, Shō=25%, Ten=25%, Ketsu=25%
        # But adjust based on twist position
        ten_pos = twist_idx / n

        # Adjust boundaries around the twist
        if 0.3 < ten_pos < 0.7:
            # Twist is well-positioned - distribute evenly around it
            ki_end = ten_pos * 0.5
            sho_end = ten_pos
            ten_end = ten_pos + (1 - ten_pos) * 0.4
        else:
            # Twist is early/late - use standard quarters
            ki_end = 0.25
            sho_end = 0.50
            ten_end = 0.75

        stage_boundaries = [0.0, ki_end, sho_end, ten_end, 1.0]

        # Compute mean values for each stage
        stage_values = []
        for i in range(4):
            start_idx = int(stage_boundaries[i] * n)
            end_idx = int(stage_boundaries[i + 1] * n)
            if end_idx > start_idx:
                stage_values.append(float(np.mean(trajectory[start_idx:end_idx])))
            else:
                stage_values.append(0.5)

        return stage_boundaries[1:], stage_values

    def _compute_conformance(
        self,
        trajectory: np.ndarray,
        twist_idx: int,
        twist_magnitude: float,
        twist_type: str,
        stability_score: float,
        stage_values: List[float]
    ) -> Tuple[float, str, List[str]]:
        """
        Compute how well trajectory conforms to kishōtenketsu.

        Returns:
            conformance_score: 0-1 measure of fit
            pattern_type: classification
            notes: observations
        """
        notes = []
        n = len(trajectory)

        # 1. Stability of Ki-Shō (first half should be stable)
        ki_sho_stability = stability_score
        if ki_sho_stability > 0.6:
            notes.append("Stable Ki-Shō section")
        elif ki_sho_stability < 0.3:
            notes.append("Unstable Ki-Shō (Western-style conflict)")

        # 2. Presence and strength of Ten (twist)
        has_twist = twist_magnitude > self.spike_threshold
        if has_twist:
            notes.append(f"Clear Ten (twist) detected at {twist_idx/n:.0%}")
        else:
            notes.append("Weak or absent Ten (twist)")

        # 3. Resolution pattern - Ketsu should return toward baseline
        ketsu_start = int(0.75 * n)
        ketsu_section = trajectory[ketsu_start:]
        ki_sho_mean = np.mean(trajectory[:int(0.5 * n)])
        ketsu_mean = np.mean(ketsu_section)

        # Ketsu should be closer to Ki-Shō than the twist extreme
        twist_value = trajectory[twist_idx]
        resolution_distance = abs(ketsu_mean - ki_sho_mean)
        twist_distance = abs(twist_value - ki_sho_mean)

        has_resolution = resolution_distance < twist_distance * 0.7
        if has_resolution:
            notes.append("Clear Ketsu (reconciliation)")
        else:
            notes.append("Incomplete resolution")

        # 4. Compare to Harmon pattern (valley vs spike)
        # Kishōtenketsu should NOT have strong descent-ascent
        first_quarter = np.mean(trajectory[:n//4])
        mid_point = np.mean(trajectory[n//4:3*n//4])
        last_quarter = np.mean(trajectory[3*n//4:])

        is_valley = mid_point < first_quarter - 0.1 and mid_point < last_quarter - 0.1
        if is_valley:
            notes.append("Warning: Valley pattern (more Harmon-like)")

        # Compute score
        stability_component = ki_sho_stability * 0.3
        twist_component = min(1.0, twist_magnitude / 3.0) * 0.35 if has_twist else 0.0
        resolution_component = 0.25 if has_resolution else 0.0
        anti_valley_component = 0.1 if not is_valley else 0.0

        conformance = stability_component + twist_component + resolution_component + anti_valley_component
        conformance = max(0.0, min(1.0, conformance))

        # Classify pattern
        if conformance > 0.6 and has_twist and ki_sho_stability > 0.4:
            pattern_type = "classic_kishotenketsu"
        elif conformance > 0.4 and has_twist:
            pattern_type = "subtle_kishotenketsu"
        elif is_valley:
            pattern_type = "western_hybrid"
        elif ki_sho_stability > 0.6 and not has_twist:
            pattern_type = "flat_narrative"
        else:
            pattern_type = "non_conforming"

        return conformance, pattern_type, notes

    def detect(
        self,
        trajectory: np.ndarray,
        trajectory_id: str = "unknown",
        title: str = "Unknown"
    ) -> KishotenketsuMatch:
        """
        Detect kishōtenketsu structure in a trajectory.

        Args:
            trajectory: Array of values (sentiment, entropy, etc.)
            trajectory_id: Identifier
            title: Title of the work

        Returns:
            KishotenketsuMatch with detection results
        """
        processed = self._preprocess_trajectory(trajectory)
        n = len(processed)

        # Find twist point
        twist_idx, twist_magnitude, twist_type = self._find_twist_point(processed)
        ten_position = twist_idx / n

        # Compute Ki-Shō stability
        ki_sho_end = int(min(twist_idx, n // 2))
        stability_score = self._compute_stability(processed, ki_sho_end)

        # Map to stages
        stage_boundaries, stage_values = self._map_to_stages(processed, twist_idx)

        # Compute conformance
        conformance, pattern_type, notes = self._compute_conformance(
            processed, twist_idx, twist_magnitude, twist_type,
            stability_score, stage_values
        )

        has_twist = twist_magnitude > self.spike_threshold

        return KishotenketsuMatch(
            trajectory_id=trajectory_id,
            title=title,
            conformance_score=conformance,
            stage_boundaries=stage_boundaries,
            stage_values=stage_values,
            ten_position=ten_position,
            ten_magnitude=twist_magnitude,
            has_twist=has_twist,
            twist_type=twist_type,
            pattern_type=pattern_type,
            stability_score=stability_score,
            notes=notes,
        )


def analyze_corpus(
    trajectories_dir: Path,
    output_dir: Path,
    trajectory_suffix: str = "_sentiment.json"
):
    """
    Analyze a corpus for kishōtenketsu conformance.

    Args:
        trajectories_dir: Directory containing trajectory JSON files
        output_dir: Output directory for results
        trajectory_suffix: Suffix for trajectory files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = KishotenketsuDetector()

    # Load trajectories
    traj_files = list(Path(trajectories_dir).glob(f"*{trajectory_suffix}"))
    traj_files = [f for f in traj_files if f.name != "manifest.json"]

    console.print(f"[blue]Analyzing {len(traj_files)} trajectories for Kishōtenketsu...[/blue]")

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

    # Save results
    results_data = {
        "total_texts": len(results),
        "pattern_distribution": pattern_counts,
        "mean_conformance": float(np.mean([r.conformance_score for r in results])),
        "mean_stability": float(np.mean([r.stability_score for r in results])),
        "texts_with_twist": sum(1 for r in results if r.has_twist),
        "results": [r.to_dict() for r in results],
    }

    with open(output_dir / "kishotenketsu_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    # Print summary
    table = Table(title="Kishōtenketsu Analysis Results")
    table.add_column("Pattern Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results)
        table.add_row(pattern, str(count), f"{pct:.1f}%")

    console.print(table)

    console.print(f"\n[bold]Top 5 Kishōtenketsu Conforming Texts:[/bold]")
    for match in results[:5]:
        console.print(f"  {match.conformance_score:.2f}: {match.title} ({match.pattern_type})")

    console.print(f"\n[green]✓ Results saved to {output_dir}[/green]")

    return results


@click.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True))
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path())
@click.option('--suffix', '-s', default='_sentiment.json', help='Trajectory file suffix')
def main(input_dir: str, output_dir: str, suffix: str):
    """Detect Kishōtenketsu patterns in trajectories."""
    analyze_corpus(Path(input_dir), Path(output_dir), suffix)


if __name__ == "__main__":
    main()
