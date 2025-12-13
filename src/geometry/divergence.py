"""
KL Divergence and Belief Trajectory Analysis.

Implements information-theoretic measures for narrative analysis:
    - KL Divergence: Measures "cost" of belief update
    - Belief Trajectories: Track reader expectations over narrative time
    - Foreshadowing Detection: Identify manifold preparation

Reference:
    Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory.
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
from scipy.special import rel_entr
from scipy.stats import entropy


@dataclass
class BeliefState:
    """
    Represents reader's belief state at a point in the narrative.

    In information geometry, this is a point on the statistical manifold.
    """
    # Probability distribution over possible outcomes/states
    distribution: np.ndarray
    # Labels for each outcome
    labels: Optional[List[str]] = None
    # Narrative position (0-1)
    position: float = 0.0
    # Confidence/certainty
    entropy: float = 0.0

    def __post_init__(self):
        # Normalize distribution
        total = np.sum(self.distribution)
        if total > 0:
            self.distribution = self.distribution / total
        # Compute entropy
        self.entropy = float(entropy(self.distribution))


@dataclass
class BeliefTrajectory:
    """
    Trajectory of belief states through a narrative.

    This represents the reader's journey through the statistical manifold.
    """
    states: List[BeliefState]
    kl_divergences: List[float] = field(default_factory=list)
    cumulative_divergence: float = 0.0

    def __post_init__(self):
        if len(self.states) > 1 and not self.kl_divergences:
            self._compute_divergences()

    def _compute_divergences(self):
        """Compute KL divergence between consecutive states."""
        self.kl_divergences = []

        for i in range(1, len(self.states)):
            p = self.states[i].distribution
            q = self.states[i - 1].distribution

            # KL(P || Q) - how much does new state diverge from old?
            kl = self._kl_divergence(p, q)
            self.kl_divergences.append(kl)

        self.cumulative_divergence = sum(self.kl_divergences)

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Compute KL divergence D_KL(P || Q).

        Args:
            p: New distribution
            q: Old/reference distribution
            epsilon: Small value to avoid log(0)

        Returns:
            KL divergence in bits
        """
        # Add epsilon to avoid division by zero
        p = p + epsilon
        q = q + epsilon

        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        # KL divergence
        kl = np.sum(rel_entr(p, q))
        return float(kl / math.log(2))  # Convert to bits

    def get_entropy_trajectory(self) -> np.ndarray:
        """Get entropy at each state."""
        return np.array([s.entropy for s in self.states])

    def get_positions(self) -> np.ndarray:
        """Get positions of each state."""
        return np.array([s.position for s in self.states])

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "n_states": len(self.states),
            "kl_divergences": self.kl_divergences,
            "cumulative_divergence": self.cumulative_divergence,
            "positions": self.get_positions().tolist(),
            "entropies": self.get_entropy_trajectory().tolist(),
        }


class KLDivergenceAnalyzer:
    """
    Analyze narratives using KL divergence framework.

    This implements the information-geometric view where:
        - Narrative states are probability distributions
        - Transitions have "cost" measured by KL divergence
        - Good stories optimize the rate of belief updating

    The Goldilocks principle: engagement peaks at moderate KL rates.
    Too low = boring, too high = confusing.
    """

    def __init__(
        self,
        n_outcomes: int = 10,
        window_size: int = 100,
        smoothing: float = 0.1,
    ):
        """
        Initialize analyzer.

        Args:
            n_outcomes: Number of outcome categories to track
            window_size: Tokens per analysis window
            smoothing: Laplace smoothing parameter
        """
        self.n_outcomes = n_outcomes
        self.window_size = window_size
        self.smoothing = smoothing

    def _estimate_distribution(
        self,
        text: str,
        vocabulary: Optional[set] = None,
    ) -> np.ndarray:
        """
        Estimate probability distribution from text.

        This is a simplified model - in practice, you'd want to use
        actual LM predictions for outcome probabilities.

        Here we use vocabulary distribution as a proxy for
        "what kinds of things are happening."
        """
        # Simple tokenization
        import re
        tokens = re.findall(r"\b\w+\b", text.lower())

        if not tokens:
            return np.ones(self.n_outcomes) / self.n_outcomes

        # Count tokens
        counts = Counter(tokens)

        if vocabulary is None:
            # Use most common tokens
            vocabulary = set([t for t, _ in counts.most_common(self.n_outcomes * 10)])

        # Create distribution over vocabulary clusters
        # (In practice, you'd cluster tokens semantically)
        distribution = np.zeros(self.n_outcomes)

        vocab_list = list(vocabulary)
        for i, token in enumerate(vocab_list[:self.n_outcomes * 10]):
            cluster = i % self.n_outcomes
            distribution[cluster] += counts.get(token, 0)

        # Add smoothing
        distribution = distribution + self.smoothing

        # Normalize
        distribution = distribution / np.sum(distribution)

        return distribution

    def compute_trajectory(
        self,
        text: str,
        n_states: int = 20,
    ) -> BeliefTrajectory:
        """
        Compute belief trajectory through a text.

        Args:
            text: Full narrative text
            n_states: Number of states to sample

        Returns:
            BeliefTrajectory with KL divergences
        """
        # Split text into segments
        words = text.split()
        segment_size = len(words) // n_states

        if segment_size < 10:
            segment_size = 10
            n_states = len(words) // segment_size

        states = []
        vocabulary = None

        for i in range(n_states):
            start = i * segment_size
            end = min((i + 1) * segment_size, len(words))
            segment = " ".join(words[start:end])

            # Estimate distribution
            dist = self._estimate_distribution(segment, vocabulary)

            # Update vocabulary for consistency
            if vocabulary is None:
                import re
                tokens = re.findall(r"\b\w+\b", text.lower())
                vocabulary = set(tokens)

            state = BeliefState(
                distribution=dist,
                position=i / (n_states - 1) if n_states > 1 else 0.5,
            )
            states.append(state)

        return BeliefTrajectory(states=states)

    def analyze_surprise(
        self,
        trajectory: BeliefTrajectory,
    ) -> Dict[str, float]:
        """
        Analyze surprise patterns in trajectory.

        Returns metrics about the information-theoretic properties:
            - mean_kl: Average KL divergence per transition
            - max_kl: Maximum KL divergence (biggest surprise)
            - kl_variance: Variance in KL (consistency of pacing)
            - goldilocks_score: How close to optimal rate
        """
        if not trajectory.kl_divergences:
            return {
                "mean_kl": 0.0,
                "max_kl": 0.0,
                "kl_variance": 0.0,
                "goldilocks_score": 0.5,
            }

        kl_values = np.array(trajectory.kl_divergences)

        mean_kl = float(np.mean(kl_values))
        max_kl = float(np.max(kl_values))
        kl_variance = float(np.var(kl_values))

        # Goldilocks score: peak engagement at moderate KL
        # Using inverted-U model: optimal around mean_kl ≈ 1 bit
        optimal_kl = 1.0
        goldilocks_score = math.exp(-((mean_kl - optimal_kl) ** 2) / 2)

        return {
            "mean_kl": mean_kl,
            "max_kl": max_kl,
            "kl_variance": kl_variance,
            "goldilocks_score": goldilocks_score,
            "cumulative_kl": trajectory.cumulative_divergence,
        }

    def detect_twists(
        self,
        trajectory: BeliefTrajectory,
        threshold_sigma: float = 2.0,
    ) -> List[Dict]:
        """
        Detect plot twists as KL divergence spikes.

        A twist is a point where KL divergence significantly
        exceeds the local mean - the reader's beliefs underwent
        an unusually large update.

        Args:
            trajectory: Belief trajectory
            threshold_sigma: Number of standard deviations for spike detection

        Returns:
            List of twist events with position and magnitude
        """
        if len(trajectory.kl_divergences) < 3:
            return []

        kl_values = np.array(trajectory.kl_divergences)
        mean_kl = np.mean(kl_values)
        std_kl = np.std(kl_values)

        threshold = mean_kl + threshold_sigma * std_kl

        twists = []
        for i, kl in enumerate(kl_values):
            if kl > threshold:
                twists.append({
                    "position": trajectory.states[i + 1].position,
                    "magnitude": float(kl),
                    "z_score": float((kl - mean_kl) / std_kl) if std_kl > 0 else 0,
                })

        return twists

    def detect_foreshadowing(
        self,
        trajectory: BeliefTrajectory,
        twist_position: float,
        window: float = 0.2,
    ) -> float:
        """
        Detect foreshadowing before a twist.

        Foreshadowing "prepares the manifold" - it reduces the KL
        divergence at the twist by gradually shifting the probability
        distribution toward the eventual reveal.

        Args:
            trajectory: Belief trajectory
            twist_position: Normalized position of the twist
            window: How far back to look for foreshadowing

        Returns:
            Foreshadowing score (0-1, higher = more foreshadowing)
        """
        # Find states before the twist
        pre_twist_states = [
            s for s in trajectory.states
            if twist_position - window <= s.position < twist_position
        ]

        if len(pre_twist_states) < 2:
            return 0.0

        # Compute entropy trend before twist
        # Foreshadowing should gradually reduce entropy in the
        # relevant dimensions (preparing for the reveal)
        entropies = [s.entropy for s in pre_twist_states]

        if len(entropies) < 2:
            return 0.0

        # Linear regression to detect downward trend
        x = np.arange(len(entropies))
        slope = np.polyfit(x, entropies, 1)[0]

        # Negative slope = foreshadowing (entropy decreasing)
        # Normalize to 0-1 score
        foreshadowing_score = max(0.0, min(1.0, -slope * 10))

        return float(foreshadowing_score)


def compute_narrative_kl_functor(
    text: str,
    n_points: int = 100,
) -> np.ndarray:
    """
    Compute KL divergence trajectory for a narrative.

    This is the categorical functor F_kl: Narr → Trajectory
    that maps narrative states to KL divergence values.

    Args:
        text: Narrative text
        n_points: Number of points in output trajectory

    Returns:
        Array of KL divergence values
    """
    analyzer = KLDivergenceAnalyzer()
    trajectory = analyzer.compute_trajectory(text, n_states=n_points)

    if not trajectory.kl_divergences:
        return np.zeros(n_points)

    # Pad KL values (one fewer than states)
    kl_values = [0.0] + trajectory.kl_divergences
    return np.array(kl_values)
