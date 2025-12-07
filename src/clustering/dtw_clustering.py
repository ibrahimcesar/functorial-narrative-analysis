"""
DTW-based Clustering for Narrative Trajectories

Implements Dynamic Time Warping distance computation and hierarchical/k-medoids clustering to identify story shapes, following Reagan et al. (2016) methodology.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.ndimage import gaussian_filter1d
from sklearn.manifold import MDS
from fastdtw import fastdtw
from tqdm import tqdm
import click
from rich.console import Console

from src.functors.base import Trajectory

console = Console()


@dataclass
class ClusterResult:
    """Results from clustering analysis."""
    labels: np.ndarray
    n_clusters: int
    centroids: List[np.ndarray]
    centroid_indices: List[int]
    distance_matrix: np.ndarray
    trajectory_ids: List[str]
    silhouette_score: float = None

    def to_dict(self) -> dict:
        return {
            "labels": self.labels.tolist(),
            "n_clusters": self.n_clusters,
            "centroids": [c.tolist() for c in self.centroids],
            "centroid_indices": self.centroid_indices,
            "trajectory_ids": self.trajectory_ids,
            "silhouette_score": self.silhouette_score,
        }


def normalize_trajectory(values: np.ndarray) -> np.ndarray:
    """Normalize trajectory to [0, 1] range."""
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val - min_val > 0:
        return (values - min_val) / (max_val - min_val)
    return np.zeros_like(values)


def resample_trajectory(values: np.ndarray, n_points: int = 100) -> np.ndarray:
    """Resample trajectory to fixed length."""
    old_x = np.linspace(0, 1, len(values))
    new_x = np.linspace(0, 1, n_points)
    return np.interp(new_x, old_x, values)


def smooth_trajectory(values: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Apply Gaussian smoothing to trajectory."""
    return gaussian_filter1d(values, sigma=sigma)


def compute_dtw_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """
    Compute DTW distance between two trajectories.

    Args:
        traj1: First trajectory (1D array)
        traj2: Second trajectory (1D array)

    Returns:
        DTW distance
    """
    distance, _ = fastdtw(traj1.reshape(-1, 1), traj2.reshape(-1, 1))
    return distance


def compute_dtw_distance_matrix(
    trajectories: List[np.ndarray],
    normalize: bool = True,
    resample_length: int = 100,
    smooth_sigma: float = 3.0
) -> np.ndarray:
    """
    Compute pairwise DTW distance matrix.

    Args:
        trajectories: List of trajectory arrays
        normalize: Whether to normalize trajectories
        resample_length: Length to resample trajectories to
        smooth_sigma: Gaussian smoothing sigma

    Returns:
        Symmetric distance matrix (n x n)
    """
    n = len(trajectories)
    console.print(f"[blue]Computing DTW distance matrix for {n} trajectories...[/blue]")

    # Preprocess trajectories
    processed = []
    for traj in trajectories:
        t = traj.copy()
        if smooth_sigma > 0:
            t = smooth_trajectory(t, smooth_sigma)
        t = resample_trajectory(t, resample_length)
        if normalize:
            t = normalize_trajectory(t)
        processed.append(t)

    # Compute pairwise distances
    distances = np.zeros((n, n))
    total_pairs = n * (n - 1) // 2

    with tqdm(total=total_pairs, desc="Computing DTW distances") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                dist = compute_dtw_distance(processed[i], processed[j])
                distances[i, j] = dist
                distances[j, i] = dist
                pbar.update(1)

    return distances


class DTWClusterer:
    """
    Clusters narrative trajectories using DTW distance and various algorithms.
    """

    def __init__(
        self,
        n_clusters: int = 6,
        method: str = "kmedoids",
        normalize: bool = True,
        resample_length: int = 100,
        smooth_sigma: float = 3.0
    ):
        """
        Initialize clusterer.

        Args:
            n_clusters: Number of clusters (default 6 per Reagan et al.)
            method: Clustering method ("kmedoids", "hierarchical")
            normalize: Whether to normalize trajectories
            resample_length: Length to resample trajectories
            smooth_sigma: Gaussian smoothing sigma
        """
        self.n_clusters = n_clusters
        self.method = method
        self.normalize = normalize
        self.resample_length = resample_length
        self.smooth_sigma = smooth_sigma

        self.distance_matrix_ = None
        self.trajectories_ = None
        self.labels_ = None

    def fit(self, trajectories: List[Trajectory]) -> ClusterResult:
        """
        Fit clustering model to trajectories.

        Args:
            trajectories: List of Trajectory objects

        Returns:
            ClusterResult with cluster assignments
        """
        console.print(f"[bold blue]Clustering {len(trajectories)} trajectories[/bold blue]")
        console.print(f"[dim]Method: {self.method}, K={self.n_clusters}[/dim]")

        # Extract values
        traj_values = [t.values for t in trajectories]
        traj_ids = [t.metadata.get("source_id", str(i)) for i, t in enumerate(trajectories)]

        # Compute distance matrix
        self.distance_matrix_ = compute_dtw_distance_matrix(
            traj_values,
            normalize=self.normalize,
            resample_length=self.resample_length,
            smooth_sigma=self.smooth_sigma
        )

        # Store preprocessed trajectories
        self.trajectories_ = []
        for traj in traj_values:
            t = traj.copy()
            if self.smooth_sigma > 0:
                t = smooth_trajectory(t, self.smooth_sigma)
            t = resample_trajectory(t, self.resample_length)
            if self.normalize:
                t = normalize_trajectory(t)
            self.trajectories_.append(t)

        # Cluster
        if self.method == "kmedoids":
            labels, centroid_indices = self._cluster_kmedoids()
        elif self.method == "hierarchical":
            labels, centroid_indices = self._cluster_hierarchical()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.labels_ = labels

        # Compute centroids (medoids)
        centroids = [self.trajectories_[idx] for idx in centroid_indices]

        # Compute silhouette score
        silhouette = self._compute_silhouette()

        result = ClusterResult(
            labels=labels,
            n_clusters=self.n_clusters,
            centroids=centroids,
            centroid_indices=centroid_indices,
            distance_matrix=self.distance_matrix_,
            trajectory_ids=traj_ids,
            silhouette_score=silhouette
        )

        console.print(f"[green]✓ Clustering complete. Silhouette: {silhouette:.3f}[/green]")

        return result

    def _cluster_kmedoids(self) -> Tuple[np.ndarray, List[int]]:
        """Perform k-medoids clustering using PAM algorithm."""
        console.print("[yellow]Running k-medoids clustering (PAM)...[/yellow]")

        n = len(self.trajectories_)
        D = self.distance_matrix_

        # Initialize medoids using k-medoids++ style initialization
        np.random.seed(42)
        medoids = [np.random.randint(n)]

        for _ in range(1, self.n_clusters):
            # Compute min distance to existing medoids for each point
            min_dists = np.min(D[:, medoids], axis=1)
            # Select next medoid with probability proportional to distance squared
            probs = min_dists ** 2
            probs /= probs.sum()
            next_medoid = np.random.choice(n, p=probs)
            medoids.append(next_medoid)

        # PAM algorithm: iteratively swap medoids to minimize total cost
        max_iter = 100
        for iteration in range(max_iter):
            # Assign points to nearest medoid
            distances_to_medoids = D[:, medoids]
            labels = np.argmin(distances_to_medoids, axis=1)

            # Compute current cost
            current_cost = sum(D[i, medoids[labels[i]]] for i in range(n))

            # Try swapping each medoid with each non-medoid
            improved = False
            best_swap = None
            best_cost = current_cost

            for m_idx, medoid in enumerate(medoids):
                for candidate in range(n):
                    if candidate in medoids:
                        continue

                    # Try swapping
                    new_medoids = medoids.copy()
                    new_medoids[m_idx] = candidate

                    # Compute new cost
                    new_distances = D[:, new_medoids]
                    new_labels = np.argmin(new_distances, axis=1)
                    new_cost = sum(D[i, new_medoids[new_labels[i]]] for i in range(n))

                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_swap = (m_idx, candidate)
                        improved = True

            if improved and best_swap:
                medoids[best_swap[0]] = best_swap[1]
            else:
                break

        # Final assignment
        distances_to_medoids = D[:, medoids]
        labels = np.argmin(distances_to_medoids, axis=1)

        return labels, medoids

    def _cluster_hierarchical(self) -> Tuple[np.ndarray, List[int]]:
        """Perform hierarchical clustering."""
        console.print("[yellow]Running hierarchical clustering...[/yellow]")

        # Convert to condensed form
        condensed = squareform(self.distance_matrix_)

        # Linkage
        Z = linkage(condensed, method="ward")

        # Cut tree
        labels = fcluster(Z, t=self.n_clusters, criterion="maxclust") - 1

        # Find medoids (most central point in each cluster)
        centroid_indices = []
        for k in range(self.n_clusters):
            mask = labels == k
            if not np.any(mask):
                centroid_indices.append(0)
                continue

            cluster_indices = np.where(mask)[0]
            sub_matrix = self.distance_matrix_[np.ix_(cluster_indices, cluster_indices)]
            avg_distances = np.mean(sub_matrix, axis=1)
            medoid_local = np.argmin(avg_distances)
            centroid_indices.append(cluster_indices[medoid_local])

        return labels, centroid_indices

    def _compute_silhouette(self) -> float:
        """Compute silhouette score."""
        from sklearn.metrics import silhouette_score

        if len(set(self.labels_)) < 2:
            return 0.0

        return silhouette_score(self.distance_matrix_, self.labels_, metric="precomputed")


def load_trajectories(trajectory_dir: Path) -> List[Trajectory]:
    """Load trajectories from a directory."""
    trajectories = []

    for traj_file in sorted(trajectory_dir.glob("*_sentiment.json")):
        with open(traj_file, 'r') as f:
            data = json.load(f)
        trajectories.append(Trajectory.from_dict(data))

    return trajectories


def identify_shape_names(centroids: List[np.ndarray]) -> List[str]:
    """
    Attempt to identify Reagan et al. shape names based on centroid characteristics.

    The six shapes are:
    1. Rags to Riches: Rising
    2. Riches to Rags: Falling
    3. Man in a Hole: Fall then rise
    4. Icarus: Rise then fall
    5. Cinderella: Rise-fall-rise
    6. Oedipus: Fall-rise-fall
    """
    shape_names = []

    for i, centroid in enumerate(centroids):
        n = len(centroid)
        first_third = centroid[:n//3].mean()
        middle_third = centroid[n//3:2*n//3].mean()
        last_third = centroid[2*n//3:].mean()

        start = centroid[:10].mean()
        end = centroid[-10:].mean()
        mid = centroid[n//2-5:n//2+5].mean()

        overall_trend = end - start
        first_half_trend = mid - start
        second_half_trend = end - mid

        # Classify based on trends
        if overall_trend > 0.2 and abs(first_half_trend - second_half_trend) < 0.15:
            name = "Rags to Riches (Rising)"
        elif overall_trend < -0.2 and abs(first_half_trend - second_half_trend) < 0.15:
            name = "Riches to Rags (Falling)"
        elif first_half_trend < -0.1 and second_half_trend > 0.1:
            name = "Man in a Hole (Fall-Rise)"
        elif first_half_trend > 0.1 and second_half_trend < -0.1:
            name = "Icarus (Rise-Fall)"
        elif middle_third > first_third and middle_third < last_third:
            name = "Cinderella (Rise-Fall-Rise)"
        elif middle_third < first_third and middle_third > last_third:
            name = "Oedipus (Fall-Rise-Fall)"
        else:
            name = f"Shape {i+1}"

        shape_names.append(name)

    return shape_names


@click.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True),
              help='Input directory with sentiment trajectories')
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(),
              help='Output directory for clustering results')
@click.option('--n-clusters', '-k', default=6, help='Number of clusters')
@click.option('--method', '-m', default='kmedoids',
              type=click.Choice(['kmedoids', 'hierarchical']),
              help='Clustering method')
def main(input_dir: str, output_dir: str, n_clusters: int, method: str):
    """Cluster sentiment trajectories to identify story shapes."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trajectories
    console.print(f"[blue]Loading trajectories from {input_dir}...[/blue]")
    trajectories = load_trajectories(input_dir)
    console.print(f"[green]Loaded {len(trajectories)} trajectories[/green]")

    # Cluster
    clusterer = DTWClusterer(n_clusters=n_clusters, method=method)
    result = clusterer.fit(trajectories)

    # Identify shape names
    shape_names = identify_shape_names(result.centroids)

    # Save results
    results_data = {
        **result.to_dict(),
        "shape_names": shape_names,
        "cluster_sizes": [int(np.sum(result.labels == k)) for k in range(n_clusters)],
    }

    # Remove large distance matrix from saved results (if present)
    results_data.pop("distance_matrix", None)

    results_file = output_dir / "clustering_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    # Save distance matrix separately
    np.save(output_dir / "distance_matrix.npy", result.distance_matrix)

    # Save centroid data
    centroids_data = {
        "centroids": [c.tolist() for c in result.centroids],
        "shape_names": shape_names
    }
    with open(output_dir / "centroids.json", 'w') as f:
        json.dump(centroids_data, f, indent=2)

    console.print(f"[bold green]✓ Results saved to {output_dir}[/bold green]")

    # Print summary
    console.print("\n[bold]Cluster Summary:[/bold]")
    for k in range(n_clusters):
        count = np.sum(result.labels == k)
        console.print(f"  {shape_names[k]}: {count} texts")


if __name__ == "__main__":
    main()
