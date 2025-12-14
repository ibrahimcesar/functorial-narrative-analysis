"""
Data-Driven Narrative Pattern Discovery

Instead of testing pre-existing models (which are epistemologically weak),
this module discovers natural narrative patterns directly from data using:

1. UNSUPERVISED CLUSTERING - Find natural groupings in trajectory space
2. MOTIF DISCOVERY - Identify recurring temporal patterns
3. INFORMATION-THEORETIC SIGNATURES - Characterize by entropy/surprisal features
4. CROSS-CULTURAL COMPARISON - What patterns are culture-specific vs universal?

This approach is bottom-up rather than top-down: let the data reveal structure
rather than projecting structure onto data.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


@dataclass
class DiscoveredPattern:
    """A narrative pattern discovered from data."""
    pattern_id: str
    name: str  # Auto-generated descriptive name
    n_members: int
    centroid: np.ndarray  # Representative trajectory
    features: Dict[str, float]  # Characteristic features
    member_ids: List[str]
    cultural_distribution: Dict[str, int]  # Japanese, Western, etc.
    description: str

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "n_members": self.n_members,
            "centroid": self.centroid.tolist(),
            "features": {k: float(v) for k, v in self.features.items()},
            "member_ids": self.member_ids,
            "cultural_distribution": self.cultural_distribution,
            "description": self.description,
        }


@dataclass
class DiscoveryReport:
    """Complete pattern discovery report."""
    n_trajectories: int
    n_patterns: int
    patterns: List[DiscoveredPattern]
    silhouette_score: float
    cultural_patterns: Dict[str, List[str]]  # Culture -> pattern IDs
    universal_patterns: List[str]  # Patterns appearing across cultures
    methodology: str

    def to_dict(self) -> dict:
        return {
            "n_trajectories": self.n_trajectories,
            "n_patterns": self.n_patterns,
            "patterns": [p.to_dict() for p in self.patterns],
            "silhouette_score": float(self.silhouette_score),
            "cultural_patterns": self.cultural_patterns,
            "universal_patterns": self.universal_patterns,
            "methodology": self.methodology,
        }


class PatternDiscoverer:
    """
    Discovers natural narrative patterns from trajectory data.

    Uses unsupervised learning to find clusters without imposing
    pre-existing theoretical structures.
    """

    def __init__(self, n_points: int = 100):
        """
        Args:
            n_points: Number of points to resample trajectories to
        """
        self.n_points = n_points

    def _resample_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Resample trajectory to fixed length."""
        x_old = np.linspace(0, 1, len(trajectory))
        x_new = np.linspace(0, 1, self.n_points)
        return np.interp(x_new, x_old, trajectory)

    def _normalize_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1] range."""
        min_val, max_val = trajectory.min(), trajectory.max()
        if max_val - min_val > 1e-8:
            return (trajectory - min_val) / (max_val - min_val)
        return np.full_like(trajectory, 0.5)

    def _extract_features(self, trajectory: np.ndarray) -> Dict[str, float]:
        """Extract interpretable features from trajectory."""
        t = trajectory

        # Basic statistics
        start = t[0]
        end = t[-1]
        mean = np.mean(t)
        std = np.std(t)

        # Trend
        slope, intercept = np.polyfit(np.linspace(0, 1, len(t)), t, 1)

        # Peaks and valleys
        peaks, _ = find_peaks(t, distance=len(t)//10, prominence=0.1)
        valleys, _ = find_peaks(-t, distance=len(t)//10, prominence=0.1)

        n_peaks = len(peaks)
        n_valleys = len(valleys)

        # Position of extrema
        max_pos = np.argmax(t) / len(t)
        min_pos = np.argmin(t) / len(t)

        # Entropy-like measure (variation)
        diff = np.diff(t)
        volatility = np.std(diff)

        # Asymmetry (skewness of trajectory)
        skewness = stats.skew(t)

        # Complexity (number of direction changes)
        sign_changes = np.sum(np.diff(np.sign(diff)) != 0)

        # Arc shape indicators
        is_rising = end > start + 0.2
        is_falling = end < start - 0.2
        has_climax_early = max_pos < 0.4
        has_climax_late = max_pos > 0.6
        has_climax_middle = 0.4 <= max_pos <= 0.6

        return {
            "start": float(start),
            "end": float(end),
            "net_change": float(end - start),
            "mean": float(mean),
            "std": float(std),
            "slope": float(slope),
            "n_peaks": n_peaks,
            "n_valleys": n_valleys,
            "max_position": float(max_pos),
            "min_position": float(min_pos),
            "volatility": float(volatility),
            "skewness": float(skewness),
            "complexity": sign_changes,
            "is_rising": int(is_rising),
            "is_falling": int(is_falling),
            "has_early_climax": int(has_climax_early),
            "has_late_climax": int(has_climax_late),
            "has_middle_climax": int(has_climax_middle),
        }

    def _generate_pattern_name(self, features: Dict[str, float], centroid: np.ndarray) -> str:
        """Generate descriptive name for discovered pattern."""
        parts = []

        # Direction
        if features["net_change"] > 0.3:
            parts.append("Rising")
        elif features["net_change"] < -0.3:
            parts.append("Falling")
        else:
            parts.append("Oscillating")

        # Complexity
        if features["n_peaks"] >= 2:
            parts.append("Multi-Peak")
        elif features["n_peaks"] == 1 and features["has_middle_climax"]:
            parts.append("Central-Climax")
        elif features["n_peaks"] == 1 and features["has_early_climax"]:
            parts.append("Early-Peak")
        elif features["n_peaks"] == 1 and features["has_late_climax"]:
            parts.append("Late-Peak")

        # Volatility
        if features["volatility"] > 0.1:
            parts.append("Turbulent")
        elif features["volatility"] < 0.03:
            parts.append("Smooth")

        # Complexity
        if features["complexity"] > 10:
            parts.append("Complex")

        return "-".join(parts) if parts else "Standard"

    def _generate_description(self, features: Dict[str, float], n_members: int) -> str:
        """Generate human-readable description of pattern."""
        desc_parts = []

        # Overall trend
        if features["net_change"] > 0.3:
            desc_parts.append("generally rises from beginning to end")
        elif features["net_change"] < -0.3:
            desc_parts.append("generally falls from beginning to end")
        else:
            desc_parts.append("maintains similar levels at start and end")

        # Peak structure
        if features["n_peaks"] == 0:
            desc_parts.append("with no significant peaks")
        elif features["n_peaks"] == 1:
            pos = features["max_position"]
            if pos < 0.3:
                desc_parts.append("with an early peak")
            elif pos > 0.7:
                desc_parts.append("with a late peak")
            else:
                desc_parts.append("with a central peak")
        else:
            desc_parts.append(f"with {features['n_peaks']} peaks")

        # Volatility
        if features["volatility"] > 0.1:
            desc_parts.append("showing high variability")
        elif features["volatility"] < 0.03:
            desc_parts.append("with smooth transitions")

        return f"Pattern found in {n_members} narratives: {', '.join(desc_parts)}."

    def discover_patterns(
        self,
        trajectories: List[np.ndarray],
        trajectory_ids: List[str],
        cultures: List[str],
        n_clusters: Optional[int] = None,
        method: str = "hierarchical"
    ) -> DiscoveryReport:
        """
        Discover natural patterns in narrative trajectories.

        Args:
            trajectories: List of trajectory arrays
            trajectory_ids: IDs for each trajectory
            cultures: Cultural origin of each trajectory
            n_clusters: Number of clusters (auto-determined if None)
            method: 'kmeans', 'hierarchical', or 'dbscan'

        Returns:
            DiscoveryReport with discovered patterns
        """
        print("\n" + "=" * 60)
        print("PATTERN DISCOVERY")
        print("Finding natural narrative structures in data")
        print("=" * 60)

        # Preprocess trajectories
        print("\n[1/5] Preprocessing trajectories...")
        processed = []
        valid_ids = []
        valid_cultures = []

        for i, (traj, tid, culture) in enumerate(zip(trajectories, trajectory_ids, cultures)):
            if len(traj) < 10:
                continue

            resampled = self._resample_trajectory(traj)
            normalized = self._normalize_trajectory(resampled)

            # Apply smoothing
            if len(normalized) > 10:
                normalized = savgol_filter(normalized, min(11, len(normalized)//2*2+1), 3)

            processed.append(normalized)
            valid_ids.append(tid)
            valid_cultures.append(culture)

        if len(processed) < 10:
            raise ValueError("Not enough valid trajectories for clustering")

        X = np.array(processed)
        print(f"  Processed {len(X)} trajectories")

        # Extract features for each trajectory
        print("\n[2/5] Extracting features...")
        all_features = [self._extract_features(t) for t in X]

        # Build feature matrix for clustering
        feature_names = list(all_features[0].keys())
        F = np.array([[f[fn] for fn in feature_names] for f in all_features])
        F_scaled = StandardScaler().fit_transform(F)

        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            print("\n[3/5] Determining optimal cluster count...")
            n_clusters = self._find_optimal_clusters(F_scaled, method)
            print(f"  Optimal clusters: {n_clusters}")
        else:
            print(f"\n[3/5] Using {n_clusters} clusters")

        # Perform clustering
        print("\n[4/5] Clustering trajectories...")

        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(F_scaled)
        elif method == "hierarchical":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(F_scaled)
        elif method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            labels = clusterer.fit_predict(F_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Compute silhouette score
        if n_clusters > 1 and n_clusters < len(X):
            sil_score = silhouette_score(F_scaled, labels)
        else:
            sil_score = 0.0

        print(f"  Silhouette score: {sil_score:.3f}")

        # Build patterns
        print("\n[5/5] Building pattern descriptions...")
        patterns = []

        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_trajectories = X[mask]
            cluster_ids = [valid_ids[i] for i, m in enumerate(mask) if m]
            cluster_cultures = [valid_cultures[i] for i, m in enumerate(mask) if m]
            cluster_features = [all_features[i] for i, m in enumerate(mask) if m]

            if len(cluster_trajectories) == 0:
                continue

            # Compute centroid
            centroid = np.mean(cluster_trajectories, axis=0)

            # Aggregate features
            avg_features = {}
            for fn in feature_names:
                avg_features[fn] = np.mean([f[fn] for f in cluster_features])

            # Cultural distribution
            culture_dist = {}
            for c in cluster_cultures:
                culture_dist[c] = culture_dist.get(c, 0) + 1

            # Generate name and description
            name = self._generate_pattern_name(avg_features, centroid)
            description = self._generate_description(avg_features, len(cluster_ids))

            pattern = DiscoveredPattern(
                pattern_id=f"P{cluster_id+1}",
                name=name,
                n_members=len(cluster_ids),
                centroid=centroid,
                features=avg_features,
                member_ids=cluster_ids,
                cultural_distribution=culture_dist,
                description=description,
            )
            patterns.append(pattern)

        # Analyze cultural patterns
        cultural_patterns = {"japanese": [], "western": [], "mixed": []}

        for pattern in patterns:
            jp_count = pattern.cultural_distribution.get("japanese", 0)
            we_count = pattern.cultural_distribution.get("western", 0)
            total = jp_count + we_count

            if total == 0:
                continue

            jp_ratio = jp_count / total

            if jp_ratio > 0.7:
                cultural_patterns["japanese"].append(pattern.pattern_id)
            elif jp_ratio < 0.3:
                cultural_patterns["western"].append(pattern.pattern_id)
            else:
                cultural_patterns["mixed"].append(pattern.pattern_id)

        # Universal patterns = mixed
        universal_patterns = cultural_patterns["mixed"]

        methodology = f"""
Pattern Discovery Methodology:
- Trajectories resampled to {self.n_points} points
- Normalized to [0, 1] range
- {len(feature_names)} features extracted per trajectory
- Clustering method: {method}
- Optimal clusters determined by silhouette analysis
- Cultural assignment: >70% = culture-specific, else mixed/universal
"""

        return DiscoveryReport(
            n_trajectories=len(X),
            n_patterns=len(patterns),
            patterns=patterns,
            silhouette_score=sil_score,
            cultural_patterns=cultural_patterns,
            universal_patterns=universal_patterns,
            methodology=methodology,
        )

    def _find_optimal_clusters(
        self,
        X: np.ndarray,
        method: str,
        max_clusters: int = 15
    ) -> int:
        """Find optimal number of clusters using silhouette score."""
        best_score = -1
        best_k = 2

        for k in range(2, min(max_clusters, len(X) // 5)):
            try:
                if method == "kmeans":
                    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
                else:
                    labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)

                score = silhouette_score(X, labels)

                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                pass

        return best_k


def run_pattern_discovery():
    """Run pattern discovery on corpus."""
    from src.geometry.surprisal import SurprisalExtractor

    print("=" * 60)
    print("DATA-DRIVEN PATTERN DISCOVERY")
    print("Finding natural narrative structures without pre-existing models")
    print("=" * 60)

    extractor = SurprisalExtractor(method="entropy")

    trajectories = []
    trajectory_ids = []
    cultures = []

    # Load Japanese texts
    print("\n[1/3] Loading Japanese trajectories...")
    jp_locations = [
        Path("data/raw/aozora_extended/texts"),
        Path("data/raw/aozora"),
    ]

    for jp_dir in jp_locations:
        if not jp_dir.exists():
            continue
        for f in list(jp_dir.glob("*.json"))[:50]:
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                content = data.get("content", data.get("text", ""))
                if len(content) > 5000:
                    traj = extractor.extract(content)
                    trajectories.append(np.array(traj.values))
                    trajectory_ids.append(data.get("id", f.stem))
                    cultures.append("japanese")
            except:
                pass

    print(f"  Loaded {sum(1 for c in cultures if c == 'japanese')} Japanese trajectories")

    # Load Western texts
    print("\n[2/3] Loading Western trajectories...")
    we_locations = [
        Path("data/raw/gutenberg"),
        Path("data/raw/large_corpus/books"),
        Path("data/raw/external_corpus/books"),
    ]

    for we_dir in we_locations:
        if not we_dir.exists():
            continue
        for f in list(we_dir.glob("*.json"))[:100]:
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                content = data.get("content", data.get("text", ""))
                if len(content) > 5000:
                    traj = extractor.extract(content)
                    trajectories.append(np.array(traj.values))
                    trajectory_ids.append(data.get("id", f.stem))
                    cultures.append("western")
            except:
                pass

    print(f"  Loaded {sum(1 for c in cultures if c == 'western')} Western trajectories")

    if len(trajectories) < 20:
        print("\nERROR: Not enough trajectories for pattern discovery")
        return None

    # Run discovery
    print("\n[3/3] Running pattern discovery...")
    discoverer = PatternDiscoverer(n_points=100)

    report = discoverer.discover_patterns(
        trajectories=trajectories,
        trajectory_ids=trajectory_ids,
        cultures=cultures,
        method="hierarchical"
    )

    # Print results
    print("\n" + "=" * 60)
    print("DISCOVERED PATTERNS")
    print("=" * 60)

    print(f"\nTotal patterns discovered: {report.n_patterns}")
    print(f"Silhouette score: {report.silhouette_score:.3f}")

    print("\nPatterns:")
    for p in report.patterns:
        print(f"\n  {p.pattern_id}: {p.name}")
        print(f"    Members: {p.n_members}")
        print(f"    Cultural: JP={p.cultural_distribution.get('japanese', 0)}, "
              f"WE={p.cultural_distribution.get('western', 0)}")
        print(f"    {p.description}")

    print("\nCultural Analysis:")
    print(f"  Japanese-specific patterns: {report.cultural_patterns.get('japanese', [])}")
    print(f"  Western-specific patterns: {report.cultural_patterns.get('western', [])}")
    print(f"  Universal patterns: {report.universal_patterns}")

    # Save results
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "discovered_patterns.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    return report


if __name__ == "__main__":
    run_pattern_discovery()
