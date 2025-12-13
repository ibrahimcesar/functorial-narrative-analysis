"""
Surprisal Extraction for Narrative Analysis.

Computes token-level and aggregated surprisal trajectories using
language models as "model readers."

Surprisal = -log P(token | context)

This measures local unexpectedness at each point in the narrative,
forming the basis for information-geometric analysis.

For practical use without GPU, we provide both:
    1. LM-based surprisal (using transformers, requires GPU for speed)
    2. N-gram based surprisal (fast, works on CPU)
    3. Entropy-based proxy (using vocabulary distribution)
"""

import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


@dataclass
class SurprisalTrajectory:
    """
    Surprisal trajectory for a narrative.

    Attributes:
        values: Surprisal values at each point
        positions: Normalized positions (0-1) in the narrative
        tokens: Optional list of tokens corresponding to values
        aggregation_level: 'token', 'sentence', 'paragraph', 'window'
        metadata: Additional information
    """
    values: np.ndarray
    positions: np.ndarray
    tokens: Optional[List[str]] = None
    aggregation_level: str = "window"
    metadata: Dict = field(default_factory=dict)

    def smooth(self, sigma: float = 2.0) -> "SurprisalTrajectory":
        """Apply Gaussian smoothing."""
        smoothed = gaussian_filter1d(self.values, sigma=sigma)
        return SurprisalTrajectory(
            values=smoothed,
            positions=self.positions,
            tokens=self.tokens,
            aggregation_level=self.aggregation_level,
            metadata={**self.metadata, "smoothed": sigma}
        )

    def resample(self, n_points: int = 100) -> "SurprisalTrajectory":
        """Resample to fixed number of points."""
        new_positions = np.linspace(0, 1, n_points)
        new_values = np.interp(new_positions, self.positions, self.values)
        return SurprisalTrajectory(
            values=new_values,
            positions=new_positions,
            tokens=None,  # Tokens don't map after resampling
            aggregation_level="resampled",
            metadata={**self.metadata, "resampled": n_points}
        )

    def normalize(self, method: str = "zscore") -> "SurprisalTrajectory":
        """Normalize surprisal values."""
        if method == "zscore":
            mean = np.mean(self.values)
            std = np.std(self.values)
            if std > 0:
                normalized = (self.values - mean) / std
            else:
                normalized = np.zeros_like(self.values)
        elif method == "minmax":
            min_val = np.min(self.values)
            max_val = np.max(self.values)
            if max_val - min_val > 0:
                normalized = (self.values - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(self.values)
        else:
            normalized = self.values

        return SurprisalTrajectory(
            values=normalized,
            positions=self.positions,
            tokens=self.tokens,
            aggregation_level=self.aggregation_level,
            metadata={**self.metadata, "normalized": method}
        )

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "values": self.values.tolist(),
            "positions": self.positions.tolist(),
            "aggregation_level": self.aggregation_level,
            "metadata": self.metadata,
        }


class SurprisalExtractor:
    """
    Extract surprisal trajectories from text.

    Supports multiple backends:
        - 'ngram': Fast n-gram based estimation
        - 'entropy': Entropy-based proxy measure
        - 'transformers': Use HuggingFace transformers (requires GPU)

    Usage:
        extractor = SurprisalExtractor(method='ngram')
        trajectory = extractor.extract(text)
    """

    def __init__(
        self,
        method: str = "entropy",
        n_gram_order: int = 3,
        window_size: int = 100,
        model_name: Optional[str] = None,
    ):
        """
        Initialize extractor.

        Args:
            method: 'ngram', 'entropy', or 'transformers'
            n_gram_order: Order for n-gram model
            window_size: Tokens per window for aggregation
            model_name: HuggingFace model name for transformers method
        """
        self.method = method
        self.n_gram_order = n_gram_order
        self.window_size = window_size
        self.model_name = model_name or "gpt2"

        self._ngram_model = None
        self._transformer_model = None
        self._tokenizer = None

    def _tokenize_simple(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        import re
        # Basic tokenization: split on whitespace, keep punctuation
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text.lower())
        return tokens

    def _build_ngram_model(self, tokens: List[str]) -> Dict[Tuple, Counter]:
        """Build n-gram language model from tokens."""
        model = {}
        n = self.n_gram_order

        # Add start tokens
        padded = ["<s>"] * (n - 1) + tokens + ["</s>"]

        for i in range(len(padded) - n + 1):
            context = tuple(padded[i:i + n - 1])
            next_token = padded[i + n - 1]

            if context not in model:
                model[context] = Counter()
            model[context][next_token] += 1

        return model

    def _ngram_probability(
        self,
        token: str,
        context: Tuple[str, ...],
        model: Dict[Tuple, Counter],
        vocab_size: int,
        smoothing: float = 0.1,
    ) -> float:
        """Get smoothed probability from n-gram model."""
        if context in model:
            count = model[context][token]
            total = sum(model[context].values())
            # Add-k smoothing
            prob = (count + smoothing) / (total + smoothing * vocab_size)
        else:
            # Backoff to uniform
            prob = 1.0 / vocab_size

        return prob

    def _extract_ngram_surprisal(self, text: str) -> SurprisalTrajectory:
        """Extract surprisal using n-gram model."""
        tokens = self._tokenize_simple(text)

        if len(tokens) < self.n_gram_order:
            return SurprisalTrajectory(
                values=np.array([0.0]),
                positions=np.array([0.5]),
            )

        # Build model from text itself (or could use pre-built corpus model)
        model = self._build_ngram_model(tokens)
        vocab = set(tokens)
        vocab_size = len(vocab)

        n = self.n_gram_order
        padded = ["<s>"] * (n - 1) + tokens

        surprisals = []
        for i in range(n - 1, len(padded)):
            context = tuple(padded[i - n + 1:i])
            token = padded[i]

            prob = self._ngram_probability(token, context, model, vocab_size)
            surprisal = -math.log2(prob) if prob > 0 else 20.0  # Cap at 20 bits
            surprisals.append(surprisal)

        # Aggregate to windows
        values, positions = self._aggregate_to_windows(surprisals)

        return SurprisalTrajectory(
            values=values,
            positions=positions,
            aggregation_level="window",
            metadata={"method": "ngram", "n": self.n_gram_order}
        )

    def _extract_entropy_surprisal(self, text: str) -> SurprisalTrajectory:
        """
        Extract surprisal proxy using local entropy.

        This measures how predictable each window is based on
        vocabulary distribution - not true surprisal but captures
        similar information about narrative complexity.
        """
        tokens = self._tokenize_simple(text)

        if len(tokens) < self.window_size:
            return SurprisalTrajectory(
                values=np.array([0.0]),
                positions=np.array([0.5]),
            )

        # Compute entropy for each window
        entropies = []
        step = self.window_size // 2

        for i in range(0, len(tokens) - self.window_size + 1, step):
            window = tokens[i:i + self.window_size]

            # Compute entropy of token distribution in window
            counts = Counter(window)
            total = sum(counts.values())

            entropy = 0.0
            for count in counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)

            entropies.append(entropy)

        values = np.array(entropies)
        positions = np.linspace(0, 1, len(entropies))

        return SurprisalTrajectory(
            values=values,
            positions=positions,
            aggregation_level="window",
            metadata={"method": "entropy", "window_size": self.window_size}
        )

    def _extract_transformer_surprisal(self, text: str) -> SurprisalTrajectory:
        """Extract surprisal using transformer language model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch required for transformer method. "
                "Install with: pip install transformers torch"
            )

        # Load model lazily
        if self._transformer_model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._transformer_model = AutoModelForCausalLM.from_pretrained(
                self.model_name
            )
            self._transformer_model.eval()

        # Tokenize
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs["input_ids"]

        # Get log probabilities
        with torch.no_grad():
            outputs = self._transformer_model(input_ids)
            logits = outputs.logits

        # Compute surprisal for each token
        log_probs = torch.log_softmax(logits, dim=-1)

        surprisals = []
        for i in range(1, input_ids.shape[1]):
            token_id = input_ids[0, i].item()
            log_prob = log_probs[0, i - 1, token_id].item()
            surprisal = -log_prob / math.log(2)  # Convert to bits
            surprisals.append(surprisal)

        # Aggregate to windows
        values, positions = self._aggregate_to_windows(surprisals)

        return SurprisalTrajectory(
            values=values,
            positions=positions,
            aggregation_level="window",
            metadata={"method": "transformers", "model": self.model_name}
        )

    def _aggregate_to_windows(
        self,
        token_values: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate token-level values to windows."""
        if len(token_values) <= self.window_size:
            return np.array([np.mean(token_values)]), np.array([0.5])

        windows = []
        step = self.window_size // 2

        for i in range(0, len(token_values) - self.window_size + 1, step):
            window = token_values[i:i + self.window_size]
            windows.append(np.mean(window))

        values = np.array(windows)
        positions = np.linspace(0, 1, len(windows))

        return values, positions

    def extract(self, text: str) -> SurprisalTrajectory:
        """
        Extract surprisal trajectory from text.

        Args:
            text: Input narrative text

        Returns:
            SurprisalTrajectory with surprisal values
        """
        if self.method == "ngram":
            return self._extract_ngram_surprisal(text)
        elif self.method == "entropy":
            return self._extract_entropy_surprisal(text)
        elif self.method == "transformers":
            return self._extract_transformer_surprisal(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def extract_relative(
        self,
        text: str,
        baseline_texts: List[str],
    ) -> SurprisalTrajectory:
        """
        Extract surprisal relative to a baseline corpus.

        This measures how a specific story deviates from genre norms.

        Args:
            text: Target narrative
            baseline_texts: List of genre-typical texts

        Returns:
            Trajectory of relative surprisal (deviation from baseline)
        """
        # Get target surprisal
        target = self.extract(text)

        # Compute baseline statistics
        baseline_trajectories = [self.extract(t) for t in baseline_texts]

        # Resample all to same length
        n_points = 100
        target_resampled = target.resample(n_points)
        baseline_resampled = [t.resample(n_points) for t in baseline_trajectories]

        # Compute mean and std at each position
        baseline_values = np.array([t.values for t in baseline_resampled])
        baseline_mean = np.mean(baseline_values, axis=0)
        baseline_std = np.std(baseline_values, axis=0)
        baseline_std = np.where(baseline_std > 0, baseline_std, 1.0)

        # Compute z-score relative to baseline
        relative_values = (target_resampled.values - baseline_mean) / baseline_std

        return SurprisalTrajectory(
            values=relative_values,
            positions=target_resampled.positions,
            aggregation_level="relative",
            metadata={
                "method": self.method,
                "baseline_size": len(baseline_texts),
                "relative": True,
            }
        )
