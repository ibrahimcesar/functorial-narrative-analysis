"""
Harmon Story Circle Detector

Detects the 8-stage Dan Harmon Story Circle structure in narrative trajectories.
The circle maps protagonist transformation through:

    1. COMFORT (You) - Character in zone of comfort
    2. NEED (Need) - They want something
    3. GO (Go) - Enter unfamiliar situation
    4. SEARCH (Search) - Adapt to new world
    5. FIND (Find) - Get what they wanted
    6. TAKE (Take) - Pay heavy price
    7. RETURN (Return) - Return to familiar
    8. CHANGE (Change) - Having changed

Geometrically represented as:

           ORDER (Known World)
        ___1_________8___
       /                  \\
      2                    7
     |                      |
     |    ←—— DESCENT ——→   |
     |                      |
      3                    6
       \\____4_____5______ /
           CHAOS (Unknown)

The detector maps sentiment/arousal trajectories to circle positions
and evaluates conformance to the canonical pattern.
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
class CircleStage:
    """Represents a stage in the Harmon Circle."""
    number: int
    name: str
    short_name: str
    realm: str  # "order" or "chaos"
    expected_sentiment: str  # "high", "low", "falling", "rising"
    description: str


# Define the 8 stages
HARMON_STAGES = [
    CircleStage(1, "Comfort", "YOU", "order", "high",
                "Character in zone of comfort"),
    CircleStage(2, "Need", "NEED", "order", "high",
                "They want something"),
    CircleStage(3, "Go", "GO", "chaos", "falling",
                "Enter unfamiliar situation"),
    CircleStage(4, "Search", "SEARCH", "chaos", "low",
                "Adapt to new world"),
    CircleStage(5, "Find", "FIND", "chaos", "low",
                "Get what they wanted"),
    CircleStage(6, "Take", "TAKE", "chaos", "rising",
                "Pay heavy price"),
    CircleStage(7, "Return", "RETURN", "order", "rising",
                "Return to familiar situation"),
    CircleStage(8, "Change", "CHANGE", "order", "high",
                "Having changed"),
]


@dataclass
class CircleMatch:
    """Result of matching a trajectory to the Harmon Circle."""
    trajectory_id: str
    title: str
    conformance_score: float  # 0-1, how well it matches
    stage_positions: List[float]  # Normalized positions [0,1] for each stage
    stage_values: List[float]  # Trajectory values at each stage
    detected_stages: List[int]  # Which stages were detected
    descent_point: float  # Where descent into chaos begins
    nadir_point: float  # Lowest point (Find/Take)
    ascent_point: float  # Where return begins
    pattern_type: str  # "full_circle", "truncated", "inverted", etc.
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "title": self.title,
            "conformance_score": self.conformance_score,
            "stage_positions": self.stage_positions,
            "stage_values": self.stage_values,
            "detected_stages": self.detected_stages,
            "descent_point": self.descent_point,
            "nadir_point": self.nadir_point,
            "ascent_point": self.ascent_point,
            "pattern_type": self.pattern_type,
            "notes": self.notes,
        }


class HarmonCircleDetector:
    """
    Detects Harmon Story Circle structure in narrative trajectories.

    The detector works by:
    1. Identifying key structural points (peaks, valleys, inflection points)
    2. Mapping these to the 8 circle stages
    3. Evaluating how well the trajectory conforms to the canonical pattern
    """

    # Canonical circle pattern (sentiment at each stage, normalized 0-1)
    # Stages 1-2 are high, 3-4-5 descend to low, 6-7-8 ascend back
    CANONICAL_PATTERN = np.array([
        0.8,   # 1. Comfort - high
        0.75,  # 2. Need - slightly lower (tension introduced)
        0.5,   # 3. Go - crossing threshold, falling
        0.3,   # 4. Search - in chaos, struggling
        0.2,   # 5. Find - nadir or near it
        0.35,  # 6. Take - paying price, starting rise
        0.6,   # 7. Return - crossing back, rising
        0.85,  # 8. Change - transformed, high (often higher than start)
    ])

    def __init__(self, smooth_sigma: float = 3.0):
        """
        Initialize detector.

        Args:
            smooth_sigma: Gaussian smoothing for trajectory preprocessing
        """
        self.smooth_sigma = smooth_sigma
        self.stages = HARMON_STAGES

    def _preprocess_trajectory(self, values: np.ndarray) -> np.ndarray:
        """Smooth and normalize trajectory."""
        # Smooth
        smoothed = gaussian_filter1d(values, sigma=self.smooth_sigma)

        # Normalize to [0, 1]
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
            - global_max: index of global maximum
            - global_min: index of global minimum
            - inflection_points: where derivative changes sign
        """
        n = len(trajectory)

        # Find peaks and valleys
        peaks, _ = find_peaks(trajectory, distance=n//10)
        valleys, _ = find_peaks(-trajectory, distance=n//10)

        # Global extrema
        global_max = np.argmax(trajectory)
        global_min = np.argmin(trajectory)

        # Compute derivative for inflection points
        derivative = np.gradient(trajectory)
        second_derivative = np.gradient(derivative)

        # Inflection points where second derivative crosses zero
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
        }

    def _map_to_stages(self, trajectory: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Map trajectory to 8 Harmon stages.

        Returns:
            stage_positions: normalized position [0,1] for each stage
            stage_values: trajectory value at each stage position
        """
        n = len(trajectory)
        struct = self._find_structural_points(trajectory)

        # Default: evenly divide the trajectory
        default_positions = np.linspace(0, 1, 8)

        # Try to find better stage positions based on structure
        stage_positions = list(default_positions)

        # Stage 1 (Comfort): Start of narrative
        stage_positions[0] = 0.0

        # Stage 8 (Change): End of narrative
        stage_positions[7] = 1.0

        # Find the main valley (nadir) - this should be stages 4-5
        if len(struct["valleys"]) > 0:
            # Find deepest valley
            valley_values = trajectory[struct["valleys"]]
            main_valley_idx = struct["valleys"][np.argmin(valley_values)]
            nadir_pos = main_valley_idx / (n - 1)

            # Stages 4-5 cluster around nadir
            if 0.2 < nadir_pos < 0.8:  # Reasonable position for nadir
                stage_positions[3] = max(0.15, nadir_pos - 0.1)  # Stage 4
                stage_positions[4] = nadir_pos  # Stage 5
                stage_positions[5] = min(0.85, nadir_pos + 0.1)  # Stage 6

                # Stages 2-3: between start and nadir
                stage_positions[1] = stage_positions[3] * 0.3
                stage_positions[2] = stage_positions[3] * 0.7

                # Stages 7: between nadir and end
                stage_positions[6] = stage_positions[5] + (1 - stage_positions[5]) * 0.5

        # Get values at stage positions
        x_norm = np.linspace(0, 1, n)
        interp_func = interp1d(x_norm, trajectory, kind='linear')
        stage_values = [float(interp_func(pos)) for pos in stage_positions]

        return stage_positions, stage_values

    def _compute_conformance(
        self,
        stage_values: List[float],
        trajectory: np.ndarray
    ) -> Tuple[float, str, List[str]]:
        """
        Compute how well trajectory conforms to Harmon Circle.

        Returns:
            conformance_score: 0-1 measure of fit
            pattern_type: classification of pattern
            notes: observations about the match
        """
        notes = []
        stage_vals = np.array(stage_values)

        # 1. Correlation with canonical pattern
        correlation = np.corrcoef(stage_vals, self.CANONICAL_PATTERN)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        # 2. Check for descent-ascent structure (key characteristic)
        first_half_mean = np.mean(stage_vals[:4])
        nadir = np.min(stage_vals[3:6])
        second_half_mean = np.mean(stage_vals[4:])

        has_descent = stage_vals[0] > nadir + 0.1
        has_ascent = stage_vals[7] > nadir + 0.1
        has_valley = nadir < first_half_mean - 0.1

        descent_ascent_score = (
            (0.3 if has_descent else 0.0) +
            (0.3 if has_ascent else 0.0) +
            (0.2 if has_valley else 0.0)
        )

        # 3. Check threshold crossings (stages 2→3 and 6→7)
        crosses_down = stage_vals[2] < stage_vals[1]
        crosses_up = stage_vals[6] > stage_vals[5]
        threshold_score = (0.1 if crosses_down else 0.0) + (0.1 if crosses_up else 0.0)

        # 4. Transformation check: end different from start
        transformation = abs(stage_vals[7] - stage_vals[0])
        if stage_vals[7] > stage_vals[0]:
            notes.append("Positive transformation (ends higher)")
        elif stage_vals[7] < stage_vals[0] - 0.1:
            notes.append("Tragic arc (ends lower than start)")
        else:
            notes.append("Circular return (ends at similar level)")

        # Combined score
        conformance = (
            0.4 * max(0, correlation) +  # Correlation component
            descent_ascent_score +  # Structural component
            threshold_score  # Threshold crossing component
        )
        conformance = max(0.0, min(1.0, conformance))

        # Classify pattern type
        if conformance > 0.7 and has_descent and has_ascent:
            pattern_type = "full_circle"
            notes.append("Strong Harmon Circle conformance")
        elif has_descent and not has_ascent:
            pattern_type = "truncated_descent"
            notes.append("Descent without return (tragedy)")
        elif not has_descent and has_ascent:
            pattern_type = "ascent_only"
            notes.append("Ascent only (rags to riches)")
        elif correlation < -0.3:
            pattern_type = "inverted"
            notes.append("Inverted pattern (anti-circle)")
        elif conformance > 0.4:
            pattern_type = "partial_circle"
            notes.append("Partial circle structure")
        else:
            pattern_type = "non_conforming"
            notes.append("Does not follow circle structure")

        return conformance, pattern_type, notes

    def detect(
        self,
        trajectory: np.ndarray,
        trajectory_id: str = "unknown",
        title: str = "Unknown"
    ) -> CircleMatch:
        """
        Detect Harmon Circle structure in a trajectory.

        Args:
            trajectory: Array of sentiment/arousal values
            trajectory_id: Identifier for the trajectory
            title: Title of the work

        Returns:
            CircleMatch with detection results
        """
        # Preprocess
        processed = self._preprocess_trajectory(trajectory)

        # Map to stages
        stage_positions, stage_values = self._map_to_stages(processed)

        # Find key points
        struct = self._find_structural_points(processed)
        n = len(processed)

        descent_point = stage_positions[2]  # Go stage
        nadir_point = stage_positions[4]  # Find stage
        ascent_point = stage_positions[6]  # Return stage

        # Use actual structural points if available
        if len(struct["valleys"]) > 0:
            main_valley = struct["valleys"][np.argmin(processed[struct["valleys"]])]
            nadir_point = main_valley / (n - 1)

        # Compute conformance
        conformance, pattern_type, notes = self._compute_conformance(
            stage_values, processed
        )

        # Determine which stages were "detected" (have meaningful values)
        detected_stages = list(range(1, 9))  # All stages assigned by default

        return CircleMatch(
            trajectory_id=trajectory_id,
            title=title,
            conformance_score=conformance,
            stage_positions=stage_positions,
            stage_values=stage_values,
            detected_stages=detected_stages,
            descent_point=descent_point,
            nadir_point=nadir_point,
            ascent_point=ascent_point,
            pattern_type=pattern_type,
            notes=notes,
        )

    def detect_batch(
        self,
        trajectories: List[Dict]
    ) -> List[CircleMatch]:
        """
        Detect Harmon Circle in multiple trajectories.

        Args:
            trajectories: List of dicts with 'values', 'id', 'title' keys

        Returns:
            List of CircleMatch results
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
):
    """
    Analyze a corpus for Harmon Circle conformance.

    Args:
        trajectories_dir: Directory containing trajectory JSON files
        output_dir: Output directory for results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = HarmonCircleDetector()

    # Load trajectories
    traj_files = list(Path(trajectories_dir).glob("*_sentiment.json"))
    traj_files = [f for f in traj_files if f.name != "manifest.json"]

    console.print(f"[blue]Analyzing {len(traj_files)} trajectories for Harmon Circle...[/blue]")

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
        "high_conformance_count": sum(1 for r in results if r.conformance_score > 0.6),
        "results": [r.to_dict() for r in results],
    }

    with open(output_dir / "harmon_circle_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    # Print summary table
    table = Table(title="Harmon Circle Analysis Results")
    table.add_column("Pattern Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results)
        table.add_row(pattern, str(count), f"{pct:.1f}%")

    console.print(table)

    # Top conforming texts
    console.print("\n[bold]Top 5 Harmon Circle Conforming Texts:[/bold]")
    for match in results[:5]:
        console.print(f"  {match.conformance_score:.2f}: {match.title} ({match.pattern_type})")

    console.print(f"\n[green]✓ Results saved to {output_dir}[/green]")

    return results


def visualize_circle_match(
    match: CircleMatch,
    trajectory: np.ndarray,
    output_file: Optional[Path] = None
):
    """
    Create visualization of Harmon Circle match.

    Args:
        match: CircleMatch result
        trajectory: Original trajectory values
        output_file: Path to save figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch

    fig = plt.figure(figsize=(14, 6))

    # Left: Trajectory with stage markers
    ax1 = fig.add_subplot(1, 2, 1)

    # Preprocess trajectory
    smoothed = gaussian_filter1d(trajectory, sigma=3.0)
    normalized = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)
    x = np.linspace(0, 1, len(normalized))

    ax1.plot(x, normalized, 'b-', linewidth=2, label='Trajectory')

    # Mark stages
    colors = plt.cm.Set1(np.linspace(0, 1, 8))
    for i, (pos, val) in enumerate(zip(match.stage_positions, match.stage_values)):
        stage = HARMON_STAGES[i]
        ax1.axvline(x=pos, color=colors[i], alpha=0.3, linestyle='--')
        ax1.scatter([pos], [val], color=colors[i], s=100, zorder=5)
        ax1.annotate(f"{i+1}. {stage.short_name}", (pos, val),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Shade order/chaos regions
    ax1.axhspan(0.5, 1.0, alpha=0.1, color='blue', label='Order')
    ax1.axhspan(0.0, 0.5, alpha=0.1, color='red', label='Chaos')

    ax1.set_xlabel('Narrative Time')
    ax1.set_ylabel('Sentiment (normalized)')
    ax1.set_title(f'{match.title}\nConformance: {match.conformance_score:.2f} ({match.pattern_type})')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)

    # Right: Circle diagram
    ax2 = fig.add_subplot(1, 2, 2, aspect='equal')

    # Draw circle
    circle = plt.Circle((0.5, 0.5), 0.4, fill=False, color='gray', linewidth=2)
    ax2.add_patch(circle)

    # Draw horizontal divide
    ax2.plot([0.1, 0.9], [0.5, 0.5], 'k--', alpha=0.5)
    ax2.text(0.5, 0.92, 'ORDER', ha='center', fontsize=10, fontweight='bold')
    ax2.text(0.5, 0.08, 'CHAOS', ha='center', fontsize=10, fontweight='bold')

    # Place stages around circle
    angles = np.linspace(np.pi/2, -3*np.pi/2, 8, endpoint=False)
    for i, angle in enumerate(angles):
        x_pos = 0.5 + 0.4 * np.cos(angle)
        y_pos = 0.5 + 0.4 * np.sin(angle)

        # Size proportional to trajectory value at that stage
        size = 100 + 200 * match.stage_values[i]

        stage = HARMON_STAGES[i]
        color = 'blue' if stage.realm == 'order' else 'red'

        ax2.scatter([x_pos], [y_pos], s=size, c=color, alpha=0.6, zorder=5)

        # Label
        label_offset = 0.12
        lx = 0.5 + (0.4 + label_offset) * np.cos(angle)
        ly = 0.5 + (0.4 + label_offset) * np.sin(angle)
        ax2.text(lx, ly, f"{i+1}. {stage.short_name}",
                ha='center', va='center', fontsize=8)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Harmon Story Circle')
    ax2.axis('off')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        console.print(f"[green]Saved circle visualization to {output_file}[/green]")
    else:
        plt.show()

    plt.close()


@click.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True),
              help='Trajectory directory')
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(),
              help='Output directory')
@click.option('--visualize', '-v', is_flag=True, help='Generate visualizations')
def main(input_dir: str, output_dir: str, visualize: bool):
    """Detect Harmon Story Circle patterns in trajectories."""
    results = analyze_corpus(Path(input_dir), Path(output_dir))

    if visualize:
        viz_dir = Path(output_dir) / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Visualize top 5 conforming
        for match in results[:5]:
            # Load original trajectory
            traj_file = Path(input_dir) / f"{match.trajectory_id}_sentiment.json"
            if traj_file.exists():
                with open(traj_file) as f:
                    data = json.load(f)
                trajectory = np.array(data["values"])

                viz_file = viz_dir / f"{match.trajectory_id}_circle.png"
                visualize_circle_match(match, trajectory, viz_file)


if __name__ == "__main__":
    main()
