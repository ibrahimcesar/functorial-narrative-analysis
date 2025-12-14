"""
F_voice: Narrative Voice Functor

Maps narrative states to point-of-view and narrative distance measurements.
Captures:
- First person vs third person narration
- Narrative distance (close/intimate vs distant/omniscient)
- POV shifts and instabilities
- Free indirect discourse detection

This functor measures "who is telling the story" and "how close we are to characters"
at any moment in the narrative.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

import numpy as np
from tqdm import tqdm
import click
from rich.console import Console

from .base import BaseFunctor, Trajectory

console = Console()


# First person indicators
FIRST_PERSON_SINGULAR = {'i', 'me', 'my', 'mine', 'myself'}
FIRST_PERSON_PLURAL = {'we', 'us', 'our', 'ours', 'ourselves'}

# Third person indicators
THIRD_PERSON_SINGULAR = {'he', 'she', 'him', 'her', 'his', 'hers', 'himself', 'herself'}
THIRD_PERSON_PLURAL = {'they', 'them', 'their', 'theirs', 'themselves'}

# Second person (rare in narrative, but sometimes used)
SECOND_PERSON = {'you', 'your', 'yours', 'yourself', 'yourselves'}

# Narrative distance markers (close/intimate)
CLOSE_NARRATIVE_MARKERS = [
    # Internal thought indicators
    'thought', 'wondered', 'felt', 'realized', 'knew', 'believed',
    'hoped', 'feared', 'wished', 'imagined', 'remembered', 'forgot',
    'sensed', 'noticed', 'seemed', 'appeared',

    # Direct perception
    'saw', 'heard', 'smelled', 'tasted', 'touched',

    # Emotional states
    'happy', 'sad', 'angry', 'afraid', 'anxious', 'nervous',
    'excited', 'worried', 'confused', 'surprised', 'shocked',

    # Free indirect discourse markers
    'of course', 'naturally', 'obviously', 'surely', 'certainly',
    'perhaps', 'maybe', 'probably', 'somehow', 'something',
]

# Narrative distance markers (distant/omniscient)
DISTANT_NARRATIVE_MARKERS = [
    # Objective description
    'there was', 'there were', 'it was', 'the room', 'the house',
    'the street', 'the city', 'the sky', 'the sun', 'the moon',

    # Historical/omniscient framing
    'years later', 'long ago', 'at that time', 'in those days',
    'meanwhile', 'elsewhere', 'unknown to', 'little did',

    # External observation
    'could be seen', 'appeared to be', 'seemed to be',
    'one might', 'an observer', 'from outside',
]

# Japanese first person pronouns
JAPANESE_FIRST_PERSON = ['私', '僕', '俺', 'わたし', 'ぼく', 'おれ', 'あたし', 'わし']

# Japanese third person reference patterns
JAPANESE_THIRD_MARKERS = ['彼', '彼女', 'かれ', 'かのじょ']


class NarrativeVoiceFunctor(BaseFunctor):
    """
    Narrative voice functor measuring POV and narrative distance.

    Maps text windows to scores where:
    - Values encode narrative distance: 0 = close/intimate, 1 = distant/omniscient
    - Metadata tracks POV type (first/third person)

    The primary axis is narrative distance, with POV as a categorical feature.
    """

    name = "narrative_voice"

    def __init__(self, method: str = "lexicon"):
        """
        Initialize narrative voice functor.

        Args:
            method: "lexicon" for dictionary-based analysis
        """
        self.method = method

        # Compile pronoun sets
        self.first_singular = FIRST_PERSON_SINGULAR
        self.first_plural = FIRST_PERSON_PLURAL
        self.third_singular = THIRD_PERSON_SINGULAR
        self.third_plural = THIRD_PERSON_PLURAL
        self.second_person = SECOND_PERSON

        self.close_markers = CLOSE_NARRATIVE_MARKERS
        self.distant_markers = DISTANT_NARRATIVE_MARKERS

    def _count_pronouns(self, text: str) -> Dict[str, int]:
        """Count pronoun occurrences by category."""
        words = re.findall(r'\b\w+\b', text.lower())
        word_set = Counter(words)

        counts = {
            "first_singular": sum(word_set.get(p, 0) for p in self.first_singular),
            "first_plural": sum(word_set.get(p, 0) for p in self.first_plural),
            "third_singular": sum(word_set.get(p, 0) for p in self.third_singular),
            "third_plural": sum(word_set.get(p, 0) for p in self.third_plural),
            "second_person": sum(word_set.get(p, 0) for p in self.second_person),
        }

        counts["first_person"] = counts["first_singular"] + counts["first_plural"]
        counts["third_person"] = counts["third_singular"] + counts["third_plural"]
        counts["total_pronouns"] = sum(counts[k] for k in
            ["first_singular", "first_plural", "third_singular", "third_plural", "second_person"])

        return counts

    def _determine_pov(self, pronoun_counts: Dict[str, int]) -> str:
        """Determine point of view from pronoun distribution."""
        first = pronoun_counts["first_person"]
        third = pronoun_counts["third_person"]
        second = pronoun_counts["second_person"]

        total = first + third + second
        if total < 5:
            return "indeterminate"

        first_ratio = first / total
        third_ratio = third / total
        second_ratio = second / total

        if first_ratio > 0.6:
            return "first_person"
        elif third_ratio > 0.6:
            return "third_person"
        elif second_ratio > 0.4:
            return "second_person"
        elif first_ratio > 0.3 and third_ratio > 0.3:
            return "mixed"
        else:
            return "third_person"  # Default

    def _compute_narrative_distance(self, text: str) -> float:
        """
        Compute narrative distance score.

        Lower = closer/more intimate narration
        Higher = more distant/omniscient narration
        """
        text_lower = text.lower()

        # Count close vs distant markers
        close_count = 0
        distant_count = 0

        for marker in self.close_markers:
            close_count += text_lower.count(marker)

        for marker in self.distant_markers:
            distant_count += text_lower.count(marker)

        # Free indirect discourse detection (exclamations in narration)
        fid_count = len(re.findall(r'[^"\']\s*!\s*[^"\']', text))
        close_count += fid_count * 0.5

        # Direct thought (italics or inner monologue patterns)
        thought_pattern = len(re.findall(r'thought|wondered|felt that', text_lower))
        close_count += thought_pattern * 0.5

        # Calculate distance score
        total = close_count + distant_count
        if total == 0:
            return 0.5  # Neutral

        distance = distant_count / (total + 1)

        # Normalize to [0, 1]
        return max(0.0, min(1.0, distance))

    def _score_window(self, text: str) -> Tuple[float, Dict]:
        """
        Compute narrative voice score for a text window.

        Returns:
            Tuple of (distance_score, metadata)
        """
        pronoun_counts = self._count_pronouns(text)
        pov = self._determine_pov(pronoun_counts)
        distance = self._compute_narrative_distance(text)

        # Adjust distance based on POV
        # First person tends to be closer
        if pov == "first_person":
            distance = distance * 0.8  # Bias toward closeness
        elif pov == "third_person":
            distance = 0.2 + distance * 0.8  # Slight bias toward distance

        metadata = {
            "pov": pov,
            "pronoun_counts": pronoun_counts,
            "raw_distance": float(distance),
        }

        return distance, metadata

    def __call__(self, windows: List[str]) -> Trajectory:
        """
        Apply narrative voice functor to text windows.

        Args:
            windows: List of text windows

        Returns:
            Trajectory with narrative distance scores
        """
        scores = []
        pov_sequence = []
        window_metadata = []

        for window in windows:
            score, meta = self._score_window(window)
            scores.append(score)
            pov_sequence.append(meta["pov"])
            window_metadata.append(meta)

        values = np.array(scores)
        time_points = np.linspace(0, 1, len(values))

        # Determine dominant POV
        pov_counts = Counter(pov_sequence)
        dominant_pov = pov_counts.most_common(1)[0][0] if pov_counts else "indeterminate"

        # Count POV shifts
        pov_shifts = sum(1 for i in range(1, len(pov_sequence))
                        if pov_sequence[i] != pov_sequence[i-1])

        return Trajectory(
            values=values,
            time_points=time_points,
            functor_name=self.name,
            metadata={
                "n_windows": len(windows),
                "dominant_pov": dominant_pov,
                "pov_distribution": dict(pov_counts),
                "pov_shifts": pov_shifts,
                "mean_distance": float(np.mean(values)),
                "distance_variance": float(np.var(values)),
            }
        )


class JapaneseNarrativeVoiceFunctor(BaseFunctor):
    """
    Narrative voice functor for Japanese text.

    Japanese narrative voice detection considers:
    - First person pronouns (私, 僕, 俺, etc.)
    - Sentence-ending particles indicating speaker perspective
    - Honorific/humble speech patterns
    """

    name = "narrative_voice_ja"

    def __init__(self):
        self.first_person = JAPANESE_FIRST_PERSON
        self.third_markers = JAPANESE_THIRD_MARKERS

        # Close narrative markers (Japanese)
        self.close_markers = [
            '思った', '感じた', '考えた', '気づいた', '分かった',
            'と思う', 'だろう', 'かもしれない', 'のだ', 'のだった',
        ]

        # Distant narrative markers (Japanese)
        self.distant_markers = [
            'であった', 'である', 'のである', 'という',
            'その時', 'かつて', 'ある日', '昔',
        ]

    def _determine_pov(self, text: str) -> str:
        """Determine POV from Japanese text."""
        first_count = sum(text.count(p) for p in self.first_person)
        third_count = sum(text.count(p) for p in self.third_markers)

        total = first_count + third_count
        if total < 3:
            return "indeterminate"

        if first_count > third_count * 2:
            return "first_person"
        elif third_count > first_count * 2:
            return "third_person"
        else:
            return "mixed"

    def _compute_distance(self, text: str) -> float:
        """Compute narrative distance for Japanese text."""
        close_count = sum(text.count(m) for m in self.close_markers)
        distant_count = sum(text.count(m) for m in self.distant_markers)

        total = close_count + distant_count
        if total == 0:
            return 0.5

        return max(0.0, min(1.0, distant_count / (total + 1)))

    def _score_window(self, text: str) -> Tuple[float, Dict]:
        """Score a Japanese text window."""
        pov = self._determine_pov(text)
        distance = self._compute_distance(text)

        if pov == "first_person":
            distance = distance * 0.8

        return distance, {"pov": pov}

    def __call__(self, windows: List[str]) -> Trajectory:
        scores = []
        pov_sequence = []

        for window in windows:
            score, meta = self._score_window(window)
            scores.append(score)
            pov_sequence.append(meta["pov"])

        pov_counts = Counter(pov_sequence)
        dominant_pov = pov_counts.most_common(1)[0][0] if pov_counts else "indeterminate"

        return Trajectory(
            values=np.array(scores),
            time_points=np.linspace(0, 1, len(scores)),
            functor_name=self.name,
            metadata={
                "language": "ja",
                "n_windows": len(scores),
                "dominant_pov": dominant_pov,
                "pov_distribution": dict(pov_counts),
            }
        )


class POVShiftDetector:
    """
    Detect significant POV shifts in narrative.

    Identifies:
    - Frame narratives (story within story)
    - Epistolary sections (letters, diary entries)
    - Stream of consciousness shifts
    - Omniscient intrusions in close third
    """

    def __init__(self):
        self.patterns = {
            "stable_first": "Consistent first person throughout",
            "stable_third": "Consistent third person throughout",
            "frame_narrative": "POV changes suggest embedded stories",
            "epistolary": "Document/letter insertions",
            "shifting": "Frequent POV changes",
            "evolving": "Gradual shift in POV/distance",
        }

    def detect(self, trajectory: Trajectory) -> Dict:
        """
        Detect POV shift patterns.

        Args:
            trajectory: Narrative voice trajectory

        Returns:
            Dict with pattern analysis
        """
        meta = trajectory.metadata
        dominant_pov = meta.get("dominant_pov", "indeterminate")
        pov_shifts = meta.get("pov_shifts", 0)
        n_windows = meta.get("n_windows", 1)

        # Calculate shift rate
        shift_rate = pov_shifts / max(1, n_windows - 1)

        # Analyze distance trajectory
        values = trajectory.values
        distance_var = np.var(values) if len(values) > 1 else 0

        # Classify pattern
        if pov_shifts == 0:
            if dominant_pov == "first_person":
                pattern = "stable_first"
            else:
                pattern = "stable_third"
            confidence = 0.9
        elif shift_rate > 0.3:
            pattern = "shifting"
            confidence = shift_rate
        elif shift_rate > 0.1 and distance_var > 0.1:
            pattern = "frame_narrative"
            confidence = 0.7
        else:
            # Check for gradual evolution
            if len(values) > 4:
                first_quarter = np.mean(values[:len(values)//4])
                last_quarter = np.mean(values[-len(values)//4:])
                if abs(last_quarter - first_quarter) > 0.2:
                    pattern = "evolving"
                    confidence = abs(last_quarter - first_quarter)
                else:
                    pattern = "stable_third" if dominant_pov == "third_person" else "stable_first"
                    confidence = 0.7
            else:
                pattern = "stable_third" if dominant_pov == "third_person" else "stable_first"
                confidence = 0.6

        return {
            "pattern": pattern,
            "confidence": float(confidence),
            "dominant_pov": dominant_pov,
            "shift_rate": float(shift_rate),
            "distance_variance": float(distance_var),
        }


def process_corpus(
    input_dir: Path,
    output_dir: Path,
    language: str = "en",
    window_size: int = 1000,
    overlap: int = 500
):
    """
    Process a corpus through the narrative voice functor.

    Args:
        input_dir: Directory containing text JSON files
        output_dir: Output directory for trajectories
        language: "en" or "ja"
        window_size: Window size (words for EN, chars for JA)
        overlap: Window overlap
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select functor
    if language == "ja":
        functor = JapaneseNarrativeVoiceFunctor()
    else:
        functor = NarrativeVoiceFunctor()

    shift_detector = POVShiftDetector()

    # Find text files
    text_files = list(Path(input_dir).glob("*.json"))
    text_files = [f for f in text_files if f.name not in ["manifest.json", "metadata.json"]]
    console.print(f"[blue]Processing {len(text_files)} texts for narrative voice...[/blue]")

    results = []

    for text_file in tqdm(text_files, desc="Extracting narrative voice"):
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            text = data.get("text", "")
            if not text:
                continue

            # Create windows
            if language == "ja":
                text_clean = re.sub(r'\s+', '', text)
                step = window_size - overlap
                windows = []
                for i in range(0, len(text_clean), step):
                    window = text_clean[i:i + window_size]
                    if len(window) >= window_size // 2:
                        windows.append(window)
                windows = windows if windows else [text]
            else:
                words = text.split()
                step = window_size - overlap
                windows = []
                for i in range(0, len(words), step):
                    window = ' '.join(words[i:i + window_size])
                    if len(window.split()) >= window_size // 2:
                        windows.append(window)
                windows = windows if windows else [text]

            # Apply functor
            trajectory = functor(windows)

            # Detect POV patterns
            shift_info = shift_detector.detect(trajectory)

            trajectory.metadata.update({
                "source_id": data.get("id", text_file.stem),
                "title": data.get("title", "Unknown"),
                "language": language,
                "pov_pattern": shift_info["pattern"],
            })

            # Save trajectory
            out_file = output_dir / f"{text_file.stem}_voice.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory.to_dict(), f, ensure_ascii=False, indent=2)

            results.append({
                "id": data.get("id", text_file.stem),
                "title": data.get("title"),
                "dominant_pov": trajectory.metadata.get("dominant_pov", "unknown"),
                "mean_distance": float(np.mean(trajectory.values)),
                "pov_shifts": trajectory.metadata.get("pov_shifts", 0),
                "pov_pattern": shift_info["pattern"],
                "trajectory_file": str(out_file.name),
            })

        except Exception as e:
            console.print(f"[red]Error processing {text_file}: {e}[/red]")
            continue

    # Save manifest
    manifest = {
        "functor": functor.name,
        "language": language,
        "window_size": window_size,
        "overlap": overlap,
        "count": len(results),
        "trajectories": results,
    }

    with open(output_dir / "manifest.json", 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    console.print(f"[bold green]✓ Processed {len(results)} texts for narrative voice[/bold green]")


@click.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True))
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path())
@click.option('--language', '-l', default='en', type=click.Choice(['en', 'ja']))
@click.option('--window-size', '-w', default=1000)
@click.option('--overlap', default=500)
def main(input_dir: str, output_dir: str, language: str, window_size: int, overlap: int):
    """Extract narrative voice trajectories from text corpus."""
    process_corpus(Path(input_dir), Path(output_dir), language, window_size, overlap)


if __name__ == "__main__":
    main()
