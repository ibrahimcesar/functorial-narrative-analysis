"""
Kishōtenketsu (起承転結) Detector - Information-Geometric Approach

Detects the 4-act East Asian narrative structure using information geometry
rather than sentiment-based heuristics.

The four acts:
    起 (Ki) - Introduction: Low information rate, establishing baseline
    承 (Shō) - Development: Gradual information accumulation, stable trajectory
    転 (Ten) - Twist/Turn: Information-theoretic reframing (KL spike, curvature anomaly)
    結 (Ketsu) - Reconciliation: Entropy compression, new understanding

Key insight: Kishōtenketsu's "twist" is NOT a dramatic conflict peak but an
INFORMATION-THEORETIC REFRAMING - the narrative suddenly asks the reader to
reconsider prior information from a new perspective.

Information-Geometric Signatures:
    Ki-Shō (起承):
        - Low curvature (smooth trajectory)
        - Stable surprisal variance
        - Gradual entropy accumulation

    Ten (転):
        - KL divergence spike (belief update)
        - Curvature anomaly (trajectory bends sharply)
        - NOT necessarily a sentiment extreme
        - Occurs typically 50-75% through narrative

    Ketsu (結):
        - Entropy compression (uncertainty reduction)
        - Return toward lower curvature
        - Information integration/synthesis

Contrast with Western Climax:
    - Western: tension builds → climax (extreme) → resolution
    - Kishōtenketsu: stability → reframing (perspective shift) → synthesis
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore, entropy
from scipy.special import rel_entr
import click
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class KishotenketsuStage:
    """Represents a stage in kishōtenketsu with information-geometric properties."""
    name_jp: str
    name_en: str
    description: str
    info_signature: str  # Information-geometric signature


KISHOTENKETSU_STAGES = [
    KishotenketsuStage(
        "起", "Ki (Introduction)",
        "Establish baseline information state",
        "low_curvature, baseline_entropy"
    ),
    KishotenketsuStage(
        "承", "Shō (Development)",
        "Gradual information accumulation",
        "stable_curvature, gradual_entropy_increase"
    ),
    KishotenketsuStage(
        "転", "Ten (Twist)",
        "Information-theoretic reframing",
        "kl_spike, curvature_anomaly"
    ),
    KishotenketsuStage(
        "結", "Ketsu (Reconciliation)",
        "Information synthesis and compression",
        "entropy_compression, curvature_descent"
    ),
]


@dataclass
class KishotenketsuMatch:
    """Result of information-geometric kishōtenketsu detection."""
    trajectory_id: str
    title: str
    conformance_score: float
    stage_boundaries: List[float]  # Normalized positions [0,1]
    stage_scores: Dict[str, float]  # Score for each stage
    ten_position: float  # Position of reframing (0-1)
    ten_strength: float  # KL divergence at Ten
    ten_type: str  # "kl_spike", "curvature_anomaly", "perspective_shift", "none"
    ki_sho_smoothness: float  # How smooth/stable Ki-Shō is
    ketsu_compression: float  # Entropy compression in Ketsu
    pattern_type: str  # Classification
    info_geo_features: Dict[str, float]  # Raw geometric features
    notes: List[str] = field(default_factory=list)

    @property
    def is_kishotenketsu(self) -> bool:
        """Whether this text conforms to kishōtenketsu structure."""
        return self.pattern_type in ("classic_kishotenketsu", "subtle_kishotenketsu", "modern_kishotenketsu")

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "title": self.title,
            "conformance_score": self.conformance_score,
            "stage_boundaries": self.stage_boundaries,
            "stage_scores": self.stage_scores,
            "ten_position": self.ten_position,
            "ten_strength": self.ten_strength,
            "ten_type": self.ten_type,
            "ki_sho_smoothness": self.ki_sho_smoothness,
            "ketsu_compression": self.ketsu_compression,
            "pattern_type": self.pattern_type,
            "info_geo_features": self.info_geo_features,
            "notes": self.notes,
        }


class KishotenketsuDetector:
    """
    Information-Geometric Kishōtenketsu Detector.

    Uses surprisal trajectories, KL divergence, and curvature analysis
    to identify the four-act structure characteristic of East Asian narrative.

    Key principles:
    1. Ki-Shō should show low curvature (smooth information flow)
    2. Ten is detected by KL divergence spike OR curvature anomaly
    3. Ketsu shows entropy compression (new understanding)
    4. Unlike Western climax, Ten need not be an emotional extreme
    """

    def __init__(
        self,
        smooth_sigma: float = 2.0,
        kl_threshold: float = 0.5,  # Threshold for KL spike detection
        curvature_threshold_percentile: float = 85,  # Percentile for curvature anomaly
        window_size: int = 50,  # Window for local statistics
    ):
        self.smooth_sigma = smooth_sigma
        self.kl_threshold = kl_threshold
        self.curvature_threshold_percentile = curvature_threshold_percentile
        self.window_size = window_size
        self.stages = KISHOTENKETSU_STAGES

    def _compute_local_entropy(self, values: np.ndarray, window: int = 20) -> np.ndarray:
        """Compute local entropy using sliding window histogram."""
        n = len(values)
        entropies = np.zeros(n)

        for i in range(n):
            start = max(0, i - window // 2)
            end = min(n, i + window // 2)
            local = values[start:end]

            if len(local) < 3:
                entropies[i] = 0
                continue

            # Compute histogram-based entropy
            hist, _ = np.histogram(local, bins=min(10, len(local) // 2 + 1), density=True)
            hist = hist[hist > 0]  # Remove zeros
            entropies[i] = entropy(hist)

        return entropies

    def _compute_kl_trajectory(self, values: np.ndarray, n_states: int = 20) -> np.ndarray:
        """
        Compute KL divergence trajectory between consecutive windows.

        High KL = large belief update = potential reframing point.
        """
        n = len(values)
        window = max(5, n // n_states)
        kl_values = []

        for i in range(0, n - window, window // 2):
            # Current window
            curr = values[i:i + window]
            # Next window
            next_start = min(i + window // 2, n - window)
            next_window = values[next_start:next_start + window]

            if len(curr) < 3 or len(next_window) < 3:
                kl_values.append(0)
                continue

            # Compute histograms
            bins = np.linspace(min(curr.min(), next_window.min()),
                              max(curr.max(), next_window.max()), 15)

            p, _ = np.histogram(curr, bins=bins, density=True)
            q, _ = np.histogram(next_window, bins=bins, density=True)

            # Add small epsilon to avoid division by zero
            p = p + 1e-10
            q = q + 1e-10
            p = p / p.sum()
            q = q / q.sum()

            # Symmetric KL divergence
            kl = 0.5 * (np.sum(rel_entr(p, q)) + np.sum(rel_entr(q, p)))
            kl_values.append(kl)

        # Interpolate to full length
        if len(kl_values) < 2:
            return np.zeros(n)

        kl_positions = np.linspace(0, 1, len(kl_values))
        full_positions = np.linspace(0, 1, n)
        return np.interp(full_positions, kl_positions, kl_values)

    def _compute_curvature(self, values: np.ndarray) -> np.ndarray:
        """Compute discrete curvature of trajectory."""
        if len(values) < 5:
            return np.zeros(len(values))

        # Smooth first
        smoothed = gaussian_filter1d(values, sigma=self.smooth_sigma)

        # First and second derivatives
        dx = np.gradient(smoothed)
        ddx = np.gradient(dx)

        # Curvature = |d²x| / (1 + (dx)²)^(3/2)
        curvature = np.abs(ddx) / (1 + dx**2)**1.5

        return curvature

    def _find_ten_point(
        self,
        values: np.ndarray,
        kl_trajectory: np.ndarray,
        curvature: np.ndarray
    ) -> Tuple[int, float, str]:
        """
        Find the Ten (転) point - the information-theoretic reframing.

        The Ten is characterized by:
        1. KL divergence spike (belief update) OR
        2. Curvature anomaly (trajectory bends) OR
        3. Both (strongest signal)

        Typically occurs 50-75% through narrative (not too early, not at end).

        Returns:
            ten_idx: Index of Ten point
            strength: Strength of the reframing signal
            ten_type: Type of Ten detected
        """
        n = len(values)

        # Focus on 40-80% of narrative (typical Ten region)
        search_start = int(0.4 * n)
        search_end = int(0.8 * n)

        if search_end <= search_start:
            search_start = int(0.3 * n)
            search_end = int(0.9 * n)

        # 1. Find KL spikes
        kl_region = kl_trajectory[search_start:search_end]
        kl_threshold = np.percentile(kl_trajectory, 75)

        kl_peaks = []
        if len(kl_region) > 3:
            try:
                peaks, props = find_peaks(
                    kl_region,
                    height=kl_threshold,
                    distance=max(1, len(kl_region) // 5)
                )
                kl_peaks = [(p + search_start, kl_trajectory[p + search_start]) for p in peaks]
            except ValueError:
                pass

        # 2. Find curvature anomalies
        curv_region = curvature[search_start:search_end]
        curv_threshold = np.percentile(curvature, self.curvature_threshold_percentile)

        curv_peaks = []
        if len(curv_region) > 3:
            try:
                peaks, props = find_peaks(
                    curv_region,
                    height=curv_threshold,
                    distance=max(1, len(curv_region) // 5)
                )
                curv_peaks = [(p + search_start, curvature[p + search_start]) for p in peaks]
            except ValueError:
                pass

        # 3. Score candidates
        candidates = []

        # KL spike candidates
        for idx, kl_val in kl_peaks:
            # Position score: prefer 50-70% position
            pos = idx / n
            pos_score = 1.0 - abs(pos - 0.6) * 2
            pos_score = max(0.2, pos_score)

            # Check if curvature is also elevated here
            local_curv = curvature[max(0, idx-5):min(n, idx+5)].mean()
            curv_bonus = 1.0 + min(1.0, local_curv / (curv_threshold + 1e-8))

            score = kl_val * pos_score * curv_bonus
            candidates.append((idx, score, kl_val, "kl_spike"))

        # Curvature anomaly candidates
        for idx, curv_val in curv_peaks:
            pos = idx / n
            pos_score = 1.0 - abs(pos - 0.6) * 2
            pos_score = max(0.2, pos_score)

            # Check if KL is also elevated
            local_kl = kl_trajectory[max(0, idx-5):min(n, idx+5)].mean()
            kl_bonus = 1.0 + min(1.0, local_kl / (kl_threshold + 1e-8))

            score = curv_val * pos_score * kl_bonus * 0.5  # Weight curvature less than KL
            candidates.append((idx, score, curv_val, "curvature_anomaly"))

        if not candidates:
            # Fallback: find maximum KL in search region
            if len(kl_region) > 0:
                max_idx = np.argmax(kl_region) + search_start
                max_kl = kl_trajectory[max_idx]
                if max_kl > self.kl_threshold * 0.5:
                    return max_idx, max_kl, "perspective_shift"

            # No clear Ten found
            return int(0.65 * n), 0.0, "none"

        # Select best candidate
        best = max(candidates, key=lambda x: x[1])
        return best[0], best[2], best[3]

    def _compute_ki_sho_smoothness(
        self,
        values: np.ndarray,
        curvature: np.ndarray,
        ten_idx: int
    ) -> float:
        """
        Compute smoothness of Ki-Shō section (before Ten).

        Kishōtenketsu Ki-Shō should have LOW curvature (smooth development).
        High smoothness = low curvature variance = kishōtenketsu-like.
        """
        # Ki-Shō is everything before Ten
        ki_sho_end = min(ten_idx, int(len(values) * 0.6))

        if ki_sho_end < 5:
            return 0.5

        ki_sho_curv = curvature[:ki_sho_end]

        # Smoothness = inverse of curvature magnitude and variance
        mean_curv = np.mean(ki_sho_curv)
        std_curv = np.std(ki_sho_curv)

        # Normalize against full trajectory
        full_mean = np.mean(curvature)

        # Lower curvature in Ki-Shō relative to full = higher smoothness
        if full_mean > 0:
            relative_smoothness = 1.0 - (mean_curv / (full_mean + 1e-8))
        else:
            relative_smoothness = 0.5

        # Low variance = more consistent smoothness
        cv = std_curv / (mean_curv + 1e-8)
        consistency = 1.0 / (1.0 + cv)

        smoothness = 0.6 * max(0, relative_smoothness) + 0.4 * consistency
        return np.clip(smoothness, 0, 1)

    def _compute_ketsu_compression(
        self,
        values: np.ndarray,
        local_entropy: np.ndarray,
        ten_idx: int
    ) -> float:
        """
        Compute entropy compression in Ketsu (after Ten).

        Kishōtenketsu Ketsu should show entropy DECREASE
        (uncertainty reduction, new understanding).
        """
        n = len(values)
        ketsu_start = ten_idx + int(0.1 * n)  # After Ten

        if ketsu_start >= n - 5:
            return 0.0

        # Compare entropy at Ten vs end
        ten_entropy = np.mean(local_entropy[max(0, ten_idx-5):min(n, ten_idx+5)])
        ketsu_entropy = np.mean(local_entropy[ketsu_start:])

        # Compression = decrease in entropy
        if ten_entropy > 0:
            compression = (ten_entropy - ketsu_entropy) / ten_entropy
        else:
            compression = 0.0

        return np.clip(compression, -1, 1)

    def _classify_pattern(
        self,
        ki_sho_smoothness: float,
        ten_strength: float,
        ten_type: str,
        ketsu_compression: float,
        ten_position: float,
        curvature_ki_sho: float = 0.0,
        curvature_ketsu: float = 0.0,
    ) -> Tuple[str, Dict[str, float], List[str]]:
        """
        Classify the narrative pattern based on information-geometric features.

        Key distinction between Kishōtenketsu and Western narrative:
        - Western: Building tension (rising curvature) → climax → quick resolution
        - Kishōtenketsu: Stable development → reframing → synthesis

        The critical markers are:
        1. Curvature RATIO: Ki-Shō curvature should be LOWER than Ketsu (reframing adds complexity)
        2. Ten POSITION: Ideal at 55-70% (later than Western midpoint climax)
        3. Ketsu COMPRESSION: Should show information integration (positive compression)
        4. Ten NATURE: KL spike (belief update) vs curvature spike (dramatic conflict)

        Returns:
            pattern_type: Classification
            stage_scores: Score for each stage
            notes: Observations
        """
        notes = []

        # Stage scores
        stage_scores = {
            "ki": 0.0,
            "sho": 0.0,
            "ten": 0.0,
            "ketsu": 0.0,
        }

        # === Ki score: Smooth, stable introduction ===
        stage_scores["ki"] = ki_sho_smoothness * 0.8
        if ki_sho_smoothness > 0.5:
            notes.append("Smooth Ki (introduction)")
        elif ki_sho_smoothness > 0.35:
            notes.append("Moderately smooth Ki")

        # === Shō score: Continued stability, gradual development ===
        stage_scores["sho"] = ki_sho_smoothness * 0.7
        if ki_sho_smoothness > 0.4:
            notes.append("Stable Shō (development)")

        # === Ten score: Reframing strength and position ===
        ten_score = 0.0
        if ten_type != "none":
            # Position bonus: 55-70% is ideal for kishōtenketsu
            # Western climax typically at 40-50%
            if 0.55 <= ten_position <= 0.72:
                pos_bonus = 1.0  # Ideal kishōtenketsu position
                notes.append(f"Ten at ideal position ({ten_position:.0%})")
            elif 0.50 <= ten_position <= 0.75:
                pos_bonus = 0.8
            elif 0.40 <= ten_position <= 0.55:
                pos_bonus = 0.5  # Early = more Western-like
                notes.append(f"Ten early ({ten_position:.0%}) - Western-like timing")
            else:
                pos_bonus = 0.3

            # Type bonus: KL spike = perspective shift (more kishōtenketsu)
            # Curvature spike = dramatic turn (more Western)
            if ten_type == "kl_spike":
                type_bonus = 1.0
                notes.append("Ten via belief update (KL spike)")
            elif ten_type == "perspective_shift":
                type_bonus = 0.8
                notes.append("Ten via perspective shift")
            else:  # curvature_anomaly
                type_bonus = 0.6
                notes.append("Ten via dramatic turn (curvature)")

            # Scale ten_strength (typically 5-15 range)
            strength_normalized = min(1.0, ten_strength / 12.0)
            ten_score = strength_normalized * pos_bonus * type_bonus

        else:
            notes.append("Weak or absent Ten")
            ten_score = 0.1

        stage_scores["ten"] = ten_score

        # === Ketsu score: Information synthesis/compression ===
        if ketsu_compression > 0.15:
            stage_scores["ketsu"] = min(1.0, ketsu_compression * 3)
            notes.append("Strong Ketsu (information synthesis)")
        elif ketsu_compression > 0.05:
            stage_scores["ketsu"] = ketsu_compression * 2
            notes.append("Partial Ketsu resolution")
        elif ketsu_compression > 0:
            stage_scores["ketsu"] = ketsu_compression
        else:
            stage_scores["ketsu"] = 0.05
            notes.append("No entropy compression in Ketsu")

        # === Curvature ratio analysis ===
        # Kishōtenketsu: Ki-Shō curvature should be relatively LOW
        # After Ten, curvature may increase (new understanding = new patterns)
        if curvature_ki_sho > 0 and curvature_ketsu > 0:
            curv_ratio = curvature_ki_sho / (curvature_ketsu + 1e-8)
            if curv_ratio < 0.9:
                # Ki-Shō smoother than Ketsu = kishōtenketsu pattern
                notes.append("Curvature increases after Ten (kishōtenketsu pattern)")
                stage_scores["ki"] *= 1.2
                stage_scores["sho"] *= 1.2
            elif curv_ratio > 1.2:
                # Ki-Shō rougher than Ketsu = Western build-up pattern
                notes.append("Curvature higher in Ki-Shō (Western build-up)")
                stage_scores["ki"] *= 0.7
                stage_scores["sho"] *= 0.7

        # Clamp scores
        for k in stage_scores:
            stage_scores[k] = np.clip(stage_scores[k], 0, 1)

        # === Overall conformance ===
        # Weight: Ki-Shō smoothness (30%), Ten quality (35%), Ketsu compression (35%)
        conformance = (
            stage_scores["ki"] * 0.15 +
            stage_scores["sho"] * 0.15 +
            stage_scores["ten"] * 0.35 +
            stage_scores["ketsu"] * 0.35
        )

        # === Pattern classification ===
        # Higher thresholds to be more selective

        # Classic kishōtenketsu: all elements present with strong compression
        if (conformance > 0.45 and
            ki_sho_smoothness > 0.35 and
            ten_type in ("kl_spike", "perspective_shift") and
            0.55 <= ten_position <= 0.75 and
            ketsu_compression > 0.08):
            pattern_type = "classic_kishotenketsu"
            notes.append("All kishōtenketsu elements present")

        # Modern kishōtenketsu: good structure, weaker compression
        elif (conformance > 0.40 and
              ten_type in ("kl_spike", "perspective_shift") and
              0.50 <= ten_position <= 0.75 and
              ketsu_compression > 0.03):
            pattern_type = "modern_kishotenketsu"

        # Subtle kishōtenketsu: Ten present at right position, some compression
        elif (ten_type != "none" and
              0.55 <= ten_position <= 0.78 and
              ketsu_compression > 0.01):
            pattern_type = "subtle_kishotenketsu"

        # Western conflict: early climax OR no compression OR curvature-based drama
        elif (ten_position < 0.50 or
              ketsu_compression <= 0 or
              (ten_type == "curvature_anomaly" and ketsu_compression < 0.02)):
            pattern_type = "western_conflict"
            notes.append("Pattern suggests Western dramatic structure")

        # Zuihitsu: smooth throughout, no clear structure
        elif ten_type == "none" and ki_sho_smoothness > 0.45:
            pattern_type = "zuihitsu"
            notes.append("Essay-like flow without clear reframing")

        # Hybrid: mixed characteristics
        else:
            pattern_type = "hybrid"
            notes.append("Mixed narrative structure")

        return pattern_type, stage_scores, notes

    def detect(
        self,
        trajectory: Union[np.ndarray, List[float]],
        trajectory_id: str = "unknown",
        title: str = "Unknown"
    ) -> KishotenketsuMatch:
        """
        Detect kishōtenketsu structure using information geometry.

        Args:
            trajectory: Surprisal/entropy values (from SurprisalExtractor)
            trajectory_id: Identifier
            title: Title of work

        Returns:
            KishotenketsuMatch with detection results
        """
        values = np.array(trajectory, dtype=float)
        n = len(values)

        if n < 10:
            return KishotenketsuMatch(
                trajectory_id=trajectory_id,
                title=title,
                conformance_score=0.0,
                stage_boundaries=[0.25, 0.5, 0.75, 1.0],
                stage_scores={"ki": 0, "sho": 0, "ten": 0, "ketsu": 0},
                ten_position=0.5,
                ten_strength=0.0,
                ten_type="none",
                ki_sho_smoothness=0.0,
                ketsu_compression=0.0,
                pattern_type="insufficient_data",
                info_geo_features={},
                notes=["Trajectory too short for analysis"],
            )

        # Compute information-geometric features
        curvature = self._compute_curvature(values)
        kl_trajectory = self._compute_kl_trajectory(values)
        local_entropy = self._compute_local_entropy(values)

        # Find Ten (reframing point)
        ten_idx, ten_strength, ten_type = self._find_ten_point(
            values, kl_trajectory, curvature
        )
        ten_position = ten_idx / n

        # Compute Ki-Shō smoothness
        ki_sho_smoothness = self._compute_ki_sho_smoothness(values, curvature, ten_idx)

        # Compute Ketsu compression
        ketsu_compression = self._compute_ketsu_compression(values, local_entropy, ten_idx)

        # Compute curvature in different regions for classification
        curvature_ki_sho = float(np.mean(curvature[:ten_idx])) if ten_idx > 0 else 0.0
        curvature_ketsu = float(np.mean(curvature[ten_idx:])) if ten_idx < n else 0.0

        # Classify pattern
        pattern_type, stage_scores, notes = self._classify_pattern(
            ki_sho_smoothness, ten_strength, ten_type, ketsu_compression, ten_position,
            curvature_ki_sho, curvature_ketsu
        )

        # Compute overall conformance
        conformance = (
            stage_scores["ki"] * 0.2 +
            stage_scores["sho"] * 0.2 +
            stage_scores["ten"] * 0.35 +
            stage_scores["ketsu"] * 0.25
        )

        # Compute stage boundaries based on Ten position
        if 0.4 < ten_position < 0.75:
            ki_end = ten_position * 0.4
            sho_end = ten_position
            ten_end = ten_position + (1 - ten_position) * 0.5
        else:
            ki_end = 0.25
            sho_end = 0.5
            ten_end = 0.75

        stage_boundaries = [ki_end, sho_end, ten_end, 1.0]

        # Collect info-geo features
        info_geo_features = {
            "mean_curvature": float(np.mean(curvature)),
            "max_curvature": float(np.max(curvature)),
            "mean_kl": float(np.mean(kl_trajectory)),
            "max_kl": float(np.max(kl_trajectory)),
            "entropy_start": float(np.mean(local_entropy[:n//4])),
            "entropy_end": float(np.mean(local_entropy[-n//4:])),
            "curvature_ki_sho": float(np.mean(curvature[:ten_idx])),
            "curvature_ketsu": float(np.mean(curvature[ten_idx:])),
        }

        return KishotenketsuMatch(
            trajectory_id=trajectory_id,
            title=title,
            conformance_score=conformance,
            stage_boundaries=stage_boundaries,
            stage_scores=stage_scores,
            ten_position=ten_position,
            ten_strength=ten_strength,
            ten_type=ten_type,
            ki_sho_smoothness=ki_sho_smoothness,
            ketsu_compression=ketsu_compression,
            pattern_type=pattern_type,
            info_geo_features=info_geo_features,
            notes=notes,
        )


def analyze_corpus(
    texts_dir: Path,
    output_dir: Path,
) -> List[KishotenketsuMatch]:
    """
    Analyze a corpus for kishōtenketsu using information geometry.

    Args:
        texts_dir: Directory containing text JSON files
        output_dir: Output directory for results
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from geometry.surprisal import SurprisalExtractor

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = KishotenketsuDetector()
    surprisal_ext = SurprisalExtractor(method='entropy', window_size=200)

    # Load texts
    text_files = list(Path(texts_dir).glob("*.json"))
    console.print(f"[blue]Analyzing {len(text_files)} texts for Kishōtenketsu...[/blue]")

    results = []
    pattern_counts = {}

    for text_file in text_files:
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            text = data.get('text', '')
            if len(text) < 3000:
                continue

            title = data.get('title', text_file.stem)

            # Extract surprisal trajectory
            trajectory = surprisal_ext.extract(text[:100000])

            # Detect kishōtenketsu
            match = detector.detect(trajectory.values, text_file.stem, title)
            results.append(match)

            pattern_counts[match.pattern_type] = pattern_counts.get(match.pattern_type, 0) + 1

        except Exception as e:
            console.print(f"[red]Error processing {text_file.name}: {e}[/red]")

    # Sort by conformance
    results.sort(key=lambda x: x.conformance_score, reverse=True)

    # Save results
    results_data = {
        "total_texts": len(results),
        "pattern_distribution": pattern_counts,
        "mean_conformance": float(np.mean([r.conformance_score for r in results])) if results else 0,
        "kishotenketsu_count": sum(1 for r in results if r.is_kishotenketsu),
        "results": [r.to_dict() for r in results],
    }

    with open(output_dir / "kishotenketsu_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    # Print summary
    table = Table(title="Kishōtenketsu Analysis (Information-Geometric)")
    table.add_column("Pattern Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results) if results else 0
        table.add_row(pattern, str(count), f"{pct:.1f}%")

    console.print(table)

    console.print(f"\n[bold]Top Kishōtenketsu Conforming Texts:[/bold]")
    for match in results[:5]:
        console.print(f"  {match.conformance_score:.2f}: {match.title} ({match.pattern_type})")
        console.print(f"         Ten: {match.ten_type} at {match.ten_position:.0%}, Smoothness: {match.ki_sho_smoothness:.2f}")

    console.print(f"\n[green]✓ Results saved to {output_dir}[/green]")

    return results


@click.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True))
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path())
def main(input_dir: str, output_dir: str):
    """Detect Kishōtenketsu patterns using information geometry."""
    analyze_corpus(Path(input_dir), Path(output_dir))


if __name__ == "__main__":
    main()
