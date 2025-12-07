"""
F_arousal: Tension/Excitement Functor

Maps narrative states to positions on a calm-excited axis.
Measures narrative tension, action intensity, and emotional arousal.

Based on the Valence-Arousal-Dominance (VAD) model of emotion,
this functor captures the activation/energy dimension separate from
positive/negative sentiment.
"""

import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm
import click
from rich.console import Console

from .base import BaseFunctor, Trajectory

console = Console()


# High arousal words (excitement, tension, action)
HIGH_AROUSAL_WORDS = [
    # Action/Movement
    'run', 'chase', 'flee', 'fight', 'attack', 'escape', 'rush', 'crash',
    'explode', 'burst', 'scream', 'shout', 'yell', 'roar', 'thunder',
    'race', 'sprint', 'leap', 'jump', 'strike', 'slam', 'smash', 'dash',

    # Intensity markers
    'sudden', 'suddenly', 'instantly', 'immediately', 'frantically',
    'desperately', 'urgently', 'violently', 'fiercely', 'wildly',

    # Emotional intensity
    'terrified', 'horrified', 'furious', 'enraged', 'ecstatic', 'thrilled',
    'shocked', 'stunned', 'astonished', 'amazed', 'panicked', 'frantic',
    'excited', 'exhilarated', 'electrified', 'passionate', 'intense',

    # Danger/Threat
    'danger', 'deadly', 'fatal', 'lethal', 'peril', 'threat', 'crisis',
    'emergency', 'catastrophe', 'disaster', 'terror', 'horror', 'fear',

    # Physical intensity
    'heart pounding', 'pulse racing', 'blood rushing', 'adrenaline',
    'trembling', 'shaking', 'gasping', 'breathless', 'sweating',

    # Exclamations (patterns)
    '!', '?!', '!!',
]

# Low arousal words (calm, peaceful, static)
LOW_AROUSAL_WORDS = [
    # Calm states
    'calm', 'peaceful', 'tranquil', 'serene', 'quiet', 'still', 'silent',
    'gentle', 'soft', 'slow', 'slowly', 'gradually', 'leisurely',

    # Rest/Stasis
    'rest', 'sleep', 'slumber', 'drowsy', 'doze', 'nap', 'dream',
    'relax', 'relaxed', 'ease', 'comfortable', 'settled', 'steady',

    # Contemplation
    'thought', 'pondered', 'considered', 'reflected', 'mused', 'wondered',
    'contemplated', 'meditated', 'remembered', 'recalled',

    # Passive states
    'sat', 'stood', 'lay', 'waited', 'watched', 'listened', 'observed',

    # Time markers (slow pace)
    'eventually', 'finally', 'at last', 'in time', 'after a while',

    # Nature (calm)
    'breeze', 'murmur', 'whisper', 'hush', 'lull', 'soothe',
]

# Punctuation patterns affecting arousal
EXCLAMATION_PATTERN = re.compile(r'[!]+')
QUESTION_EXCLAIM_PATTERN = re.compile(r'[?!]+')
ELLIPSIS_PATTERN = re.compile(r'\.{3,}')


class ArousalFunctor(BaseFunctor):
    """
    Arousal functor measuring narrative tension and excitement.

    Maps text windows to arousal scores in [0, 1] range where:
    - 0 = very calm, peaceful, low-energy
    - 1 = very excited, tense, high-energy
    """

    name = "arousal"

    def __init__(self, method: str = "lexicon"):
        """
        Initialize arousal functor.

        Args:
            method: "lexicon" for dictionary-based, "hybrid" for combined approach
        """
        self.method = method
        self.high_arousal = set(w.lower() for w in HIGH_AROUSAL_WORDS if not w.startswith('!'))
        self.low_arousal = set(w.lower() for w in LOW_AROUSAL_WORDS)

    def _count_exclamations(self, text: str) -> int:
        """Count exclamation-related punctuation."""
        exclaim = len(EXCLAMATION_PATTERN.findall(text))
        question_exclaim = len(QUESTION_EXCLAIM_PATTERN.findall(text))
        return exclaim + question_exclaim

    def _lexicon_score(self, text: str) -> float:
        """
        Compute arousal using lexicon lookup.

        Args:
            text: Text window

        Returns:
            Arousal score in [0, 1]
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        high_count = 0
        low_count = 0

        # Count lexicon matches
        for word in words:
            if word in self.high_arousal:
                high_count += 1
            elif word in self.low_arousal:
                low_count += 1

        # Check for multi-word patterns
        for phrase in ['heart pounding', 'pulse racing', 'blood rushing']:
            if phrase in text_lower:
                high_count += 2

        # Punctuation-based arousal
        exclamations = self._count_exclamations(text)
        high_count += exclamations * 0.5

        # Sentence length variance (short choppy sentences = high arousal)
        sentences = re.split(r'[.!?]+', text)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if len(sentence_lengths) > 1:
            length_variance = np.var(sentence_lengths)
            mean_length = np.mean(sentence_lengths)
            # Short mean sentence length with high variance = action
            if mean_length < 10 and length_variance > 20:
                high_count += 1

        # Calculate score
        total = high_count + low_count
        if total == 0:
            return 0.5  # Neutral

        # Normalize to [0, 1]
        raw_score = (high_count - low_count) / (total + 1)
        # Map from [-1, 1] to [0, 1]
        score = (raw_score + 1) / 2

        return max(0.0, min(1.0, score))

    def _score_window(self, text: str) -> float:
        """
        Compute arousal for a text window.

        Args:
            text: Window text

        Returns:
            Arousal score in [0, 1]
        """
        return self._lexicon_score(text)

    def __call__(self, windows: List[str]) -> Trajectory:
        """
        Apply arousal functor to text windows.

        Args:
            windows: List of text windows

        Returns:
            Trajectory with arousal scores
        """
        scores = [self._score_window(window) for window in windows]

        values = np.array(scores)
        time_points = np.linspace(0, 1, len(values))

        return Trajectory(
            values=values,
            time_points=time_points,
            functor_name=self.name,
            metadata={
                "method": self.method,
                "n_windows": len(windows),
            }
        )


# Japanese arousal words
HIGH_AROUSAL_JA = [
    '走る', '逃げる', '戦う', '叫ぶ', '怒る', '驚く', '恐れる', '震える',
    '激しい', '急に', '突然', '必死', '危険', '緊急', '恐怖', '興奮',
    '爆発', '衝突', '叫び', '絶叫', '悲鳴', '怒り', '激怒', '狂う',
]

LOW_AROUSAL_JA = [
    '静か', '穏やか', '平和', '安らか', '眠る', '休む', '落ち着く',
    'ゆっくり', '徐々に', '静寂', '沈黙', '思う', '考える', '待つ',
]


class JapaneseArousalFunctor(BaseFunctor):
    """Arousal functor for Japanese text."""

    name = "arousal_ja"

    def __init__(self):
        self.high_arousal = set(HIGH_AROUSAL_JA)
        self.low_arousal = set(LOW_AROUSAL_JA)

    def _score_window(self, text: str) -> float:
        """Compute arousal for Japanese text window."""
        high_count = sum(1 for word in self.high_arousal if word in text)
        low_count = sum(1 for word in self.low_arousal if word in text)

        # Exclamation marks
        high_count += text.count('!') * 0.5
        high_count += text.count('！') * 0.5

        total = high_count + low_count
        if total == 0:
            return 0.5

        raw_score = (high_count - low_count) / (total + 1)
        return max(0.0, min(1.0, (raw_score + 1) / 2))

    def __call__(self, windows: List[str]) -> Trajectory:
        scores = [self._score_window(window) for window in windows]

        return Trajectory(
            values=np.array(scores),
            time_points=np.linspace(0, 1, len(scores)),
            functor_name=self.name,
            metadata={"language": "ja", "n_windows": len(windows)}
        )


def process_corpus(
    input_dir: Path,
    output_dir: Path,
    language: str = "en",
    window_size: int = 1000,
    overlap: int = 500
):
    """
    Process a corpus through the arousal functor.

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
        functor = JapaneseArousalFunctor()
    else:
        functor = ArousalFunctor()

    # Find text files
    text_files = list(Path(input_dir).glob("*.json"))
    text_files = [f for f in text_files if f.name != "manifest.json"]
    console.print(f"[blue]Processing {len(text_files)} texts for arousal...[/blue]")

    results = []

    for text_file in tqdm(text_files, desc="Extracting arousal"):
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            text = data.get("text", "")
            if not text:
                continue

            # Create windows
            if language == "ja":
                # Japanese: window by characters
                text_clean = re.sub(r'\s+', '', text)
                step = window_size - overlap
                windows = []
                for i in range(0, len(text_clean), step):
                    window = text_clean[i:i + window_size]
                    if len(window) >= window_size // 2:
                        windows.append(window)
                windows = windows if windows else [text]
            else:
                # English: window by words
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

            # Add metadata
            trajectory.metadata.update({
                "source_id": data.get("id", text_file.stem),
                "title": data.get("title", "Unknown"),
                "language": language,
            })

            # Save trajectory
            out_file = output_dir / f"{text_file.stem}_arousal.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory.to_dict(), f, ensure_ascii=False, indent=2)

            results.append({
                "id": data.get("id", text_file.stem),
                "title": data.get("title"),
                "mean_arousal": float(np.mean(trajectory.values)),
                "std_arousal": float(np.std(trajectory.values)),
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

    console.print(f"[bold green]✓ Processed {len(results)} texts for arousal[/bold green]")


@click.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True))
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path())
@click.option('--language', '-l', default='en', type=click.Choice(['en', 'ja']))
@click.option('--window-size', '-w', default=1000)
@click.option('--overlap', default=500)
def main(input_dir: str, output_dir: str, language: str, window_size: int, overlap: int):
    """Extract arousal trajectories from text corpus."""
    process_corpus(Path(input_dir), Path(output_dir), language, window_size, overlap)


if __name__ == "__main__":
    main()
