"""
Base class for observation functors.

All functors map sequences of text windows to numerical trajectories,
enabling shape analysis across different measurement dimensions.
"""

from abc import ABC, abstractmethod
from typing import List, Union
from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    """
    Represents a narrative trajectory in some measurement space.

    Attributes:
        values: Array of measurements at each time point
        time_points: Normalized time points (0-1)
        functor_name: Name of the functor that produced this trajectory
        metadata: Additional information about the trajectory
    """
    values: np.ndarray
    time_points: np.ndarray
    functor_name: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.time_points is None:
            self.time_points = np.linspace(0, 1, len(self.values))

    def normalize(self) -> "Trajectory":
        """Normalize values to [0, 1] range."""
        min_val = np.min(self.values)
        max_val = np.max(self.values)
        if max_val - min_val > 0:
            normalized = (self.values - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(self.values)
        return Trajectory(
            values=normalized,
            time_points=self.time_points,
            functor_name=self.functor_name,
            metadata={**self.metadata, "normalized": True}
        )

    def smooth(self, window_size: int = 5) -> "Trajectory":
        """Apply moving average smoothing."""
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(self.values, kernel, mode='same')
        return Trajectory(
            values=smoothed,
            time_points=self.time_points,
            functor_name=self.functor_name,
            metadata={**self.metadata, "smoothed": window_size}
        )

    def resample(self, n_points: int = 100) -> "Trajectory":
        """Resample trajectory to fixed number of points."""
        new_time = np.linspace(0, 1, n_points)
        new_values = np.interp(new_time, self.time_points, self.values)
        return Trajectory(
            values=new_values,
            time_points=new_time,
            functor_name=self.functor_name,
            metadata={**self.metadata, "resampled": n_points}
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "values": self.values.tolist(),
            "time_points": self.time_points.tolist(),
            "functor_name": self.functor_name,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        """Create from dictionary."""
        return cls(
            values=np.array(data["values"]),
            time_points=np.array(data["time_points"]),
            functor_name=data["functor_name"],
            metadata=data.get("metadata", {})
        )


class BaseFunctor(ABC):
    """
    Abstract base class for observation functors.

    A functor F: Narr → Trajectory maps narrative states (represented as
    text windows) to numerical trajectories that can be clustered and compared.
    """

    name: str = "base"

    @abstractmethod
    def __call__(self, windows: List[str]) -> Trajectory:
        """
        Apply the functor to a sequence of text windows.

        Args:
            windows: List of text windows representing narrative states

        Returns:
            Trajectory object with measurements for each window
        """
        pass

    def process_text(self, text: str, window_size: int = 1000, overlap: int = 500) -> Trajectory:
        """
        Process a full text by windowing and applying the functor.

        Args:
            text: Full narrative text
            window_size: Size of each window in tokens (approximate)
            overlap: Overlap between windows in tokens

        Returns:
            Trajectory for the text
        """
        windows = self._create_windows(text, window_size, overlap)
        return self(windows)

    def _create_windows(self, text: str, window_size: int, overlap: int) -> List[str]:
        """
        Split text into overlapping windows.

        Args:
            text: Full text to window
            window_size: Approximate tokens per window
            overlap: Overlap between windows

        Returns:
            List of text windows
        """
        words = text.split()
        step = window_size - overlap
        windows = []

        for i in range(0, len(words), step):
            window = ' '.join(words[i:i + window_size])
            if len(window.split()) >= window_size // 2:  # Minimum half window
                windows.append(window)

        return windows if windows else [text]
