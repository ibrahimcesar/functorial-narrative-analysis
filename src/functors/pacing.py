"""
F_pacing: Narrative Pacing Functor

Maps narrative states to pacing/rhythm measurements based on:
- Scene length variation
- Dialogue density (dialogue vs. narration ratio)
- Sentence rhythm (short punchy vs. long flowing)
- Paragraph breaks frequency

High pacing = fast, action-oriented, dialogue-heavy
Low pacing = slow, descriptive, narration-heavy

This functor captures the temporal "feel" of narrative progression.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
import click
from rich.console import Console

from .base import BaseFunctor, Trajectory

console = Console()


class PacingFunctor(BaseFunctor):
    """
    Pacing functor measuring narrative rhythm and tempo.

    Maps text windows to pacing scores in [0, 1] where:
    - 0 = very slow pacing (long descriptive passages, minimal dialogue)
    - 1 = very fast pacing (short sentences, heavy dialogue, action)
    """

    name = "pacing"

    def __init__(
        self,
        dialogue_weight: float = 0.35,
        sentence_weight: float = 0.30,
        paragraph_weight: float = 0.20,
        action_weight: float = 0.15,
    ):
        """
        Initialize pacing functor.

        Args:
            dialogue_weight: Weight for dialogue density
            sentence_weight: Weight for sentence length variation
            paragraph_weight: Weight for paragraph frequency
            action_weight: Weight for action verb density
        """
        self.dialogue_weight = dialogue_weight
        self.sentence_weight = sentence_weight
        self.paragraph_weight = paragraph_weight
        self.action_weight = action_weight

        # Action verbs that indicate fast pacing
        self.action_verbs = {
            'run', 'ran', 'jump', 'jumped', 'grab', 'grabbed', 'rush', 'rushed',
            'dash', 'dashed', 'sprint', 'sprinted', 'slam', 'slammed', 'crash',
            'crashed', 'throw', 'threw', 'catch', 'caught', 'hit', 'strike',
            'struck', 'kick', 'kicked', 'punch', 'punched', 'push', 'pushed',
            'pull', 'pulled', 'shout', 'shouted', 'scream', 'screamed', 'yell',
            'yelled', 'snap', 'snapped', 'spin', 'spun', 'dodge', 'dodged',
            'leap', 'leaped', 'burst', 'explode', 'exploded', 'chase', 'chased',
        }

    def _detect_dialogue(self, text: str) -> Tuple[int, int]:
        """
        Detect dialogue in text.

        Returns:
            Tuple of (dialogue_chars, total_chars)
        """
        # Match text within quotes
        dialogue_patterns = [
            r'"[^"]*"',  # Double quotes
            r"'[^']*'",  # Single quotes (careful with contractions)
            r'"[^"]*"',  # Smart quotes
            r'「[^」]*」',  # Japanese quotes
            r'『[^』]*』',  # Japanese double quotes
        ]

        dialogue_chars = 0
        for pattern in dialogue_patterns:
            matches = re.findall(pattern, text)
            dialogue_chars += sum(len(m) for m in matches)

        return dialogue_chars, len(text)

    def _compute_dialogue_density(self, text: str) -> float:
        """
        Compute dialogue density score.

        Higher = more dialogue (faster pacing)
        """
        dialogue_chars, total_chars = self._detect_dialogue(text)

        if total_chars == 0:
            return 0.5

        ratio = dialogue_chars / total_chars

        # Normalize: 0% dialogue = 0, 50%+ dialogue = 1
        return min(1.0, ratio * 2)

    def _compute_sentence_rhythm(self, text: str) -> float:
        """
        Compute sentence rhythm score based on length patterns.

        Short sentences = fast pacing (higher score)
        Long sentences = slow pacing (lower score)
        High variance = dynamic pacing (moderate boost)
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.5

        # Word counts per sentence
        lengths = [len(s.split()) for s in sentences]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)

        # Short average = fast (inverse relationship)
        # Typical range: 5-30 words per sentence
        if mean_length < 8:
            length_score = 1.0
        elif mean_length > 25:
            length_score = 0.0
        else:
            length_score = 1 - (mean_length - 8) / 17

        # High variance adds dynamism (slight boost)
        cv = std_length / mean_length if mean_length > 0 else 0
        variance_boost = min(0.15, cv * 0.1)

        return min(1.0, length_score + variance_boost)

    def _compute_paragraph_frequency(self, text: str) -> float:
        """
        Compute paragraph break frequency.

        More paragraph breaks = faster visual pacing
        """
        # Estimate paragraphs by double newlines or significant whitespace
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if len(paragraphs) <= 1:
            # No paragraph breaks detected, estimate from length
            words = len(text.split())
            # Assume one paragraph per ~150 words is "normal"
            estimated_paragraphs = max(1, words / 150)
            actual_density = len(paragraphs) / estimated_paragraphs
            return min(1.0, actual_density)

        # Words per paragraph
        words = len(text.split())
        words_per_para = words / len(paragraphs)

        # Short paragraphs = fast pacing
        # Typical range: 30-200 words per paragraph
        if words_per_para < 50:
            return 1.0
        elif words_per_para > 150:
            return 0.2
        else:
            return 1 - (words_per_para - 50) / 100

    def _compute_action_density(self, text: str) -> float:
        """
        Compute action verb density.

        More action verbs = faster pacing
        """
        words = re.findall(r'\b\w+\b', text.lower())

        if len(words) == 0:
            return 0.5

        action_count = sum(1 for w in words if w in self.action_verbs)

        # Normalize: 0% action = 0, 3%+ action verbs = 1
        ratio = action_count / len(words)
        return min(1.0, ratio * 33)  # ~3% threshold

    def _score_window(self, text: str) -> float:
        """
        Compute pacing score for a text window.

        Args:
            text: Window text

        Returns:
            Pacing score in [0, 1]
        """
        dialogue = self._compute_dialogue_density(text)
        sentence = self._compute_sentence_rhythm(text)
        paragraph = self._compute_paragraph_frequency(text)
        action = self._compute_action_density(text)

        combined = (
            self.dialogue_weight * dialogue +
            self.sentence_weight * sentence +
            self.paragraph_weight * paragraph +
            self.action_weight * action
        )

        return max(0.0, min(1.0, combined))

    def __call__(self, windows: List[str]) -> Trajectory:
        """
        Apply pacing functor to text windows.

        Args:
            windows: List of text windows

        Returns:
            Trajectory with pacing scores
        """
        scores = [self._score_window(window) for window in windows]

        values = np.array(scores)
        time_points = np.linspace(0, 1, len(values))

        return Trajectory(
            values=values,
            time_points=time_points,
            functor_name=self.name,
            metadata={
                "n_windows": len(windows),
                "mean_pacing": float(np.mean(values)),
                "pacing_variance": float(np.var(values)),
            }
        )


class JapanesePacingFunctor(BaseFunctor):
    """
    Pacing functor for Japanese text.

    Japanese-specific considerations:
    - Different quote markers (「」『』)
    - Sentence endings (。！？)
    - Character-based length measurement
    """

    name = "pacing_ja"

    def __init__(self):
        # Japanese action verbs
        self.action_verbs = [
            '走る', '跳ぶ', '叫ぶ', '掴む', '投げる', '打つ', '蹴る',
            '逃げる', '追う', '戦う', '飛ぶ', '突く', '切る', '殴る',
        ]

    def _compute_dialogue_density(self, text: str) -> float:
        """Compute dialogue density for Japanese."""
        dialogue_patterns = [r'「[^」]*」', r'『[^』]*』', r'"[^"]*"']

        dialogue_chars = 0
        for pattern in dialogue_patterns:
            matches = re.findall(pattern, text)
            dialogue_chars += sum(len(m) for m in matches)

        total = len(text.replace(' ', '').replace('\n', ''))
        if total == 0:
            return 0.5

        return min(1.0, (dialogue_chars / total) * 2)

    def _compute_sentence_rhythm(self, text: str) -> float:
        """Compute sentence rhythm for Japanese."""
        sentences = re.split(r'[。！？]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.5

        lengths = [len(s) for s in sentences]  # Character count
        mean_length = np.mean(lengths)

        # Japanese sentences: 10-50 chars typical
        if mean_length < 15:
            return 1.0
        elif mean_length > 40:
            return 0.2
        else:
            return 1 - (mean_length - 15) / 25

    def _compute_action_density(self, text: str) -> float:
        """Compute action verb density for Japanese."""
        count = sum(1 for verb in self.action_verbs if verb in text)
        # Normalize by text length (chars)
        ratio = count / max(1, len(text) / 100)
        return min(1.0, ratio * 0.5)

    def _score_window(self, text: str) -> float:
        """Compute pacing score for Japanese text."""
        dialogue = self._compute_dialogue_density(text)
        sentence = self._compute_sentence_rhythm(text)
        action = self._compute_action_density(text)

        return 0.4 * dialogue + 0.4 * sentence + 0.2 * action

    def __call__(self, windows: List[str]) -> Trajectory:
        scores = [self._score_window(window) for window in windows]

        return Trajectory(
            values=np.array(scores),
            time_points=np.linspace(0, 1, len(scores)),
            functor_name=self.name,
            metadata={"language": "ja", "n_windows": len(windows)}
        )


def detect_scene_breaks(text: str) -> List[int]:
    """
    Detect potential scene breaks in text.

    Scene breaks indicated by:
    - Multiple blank lines
    - Centered asterisks or dashes
    - Chapter markers
    """
    breaks = []

    # Find double+ newlines
    for match in re.finditer(r'\n\s*\n\s*\n', text):
        breaks.append(match.start())

    # Find centered breaks (*** or ---)
    for match in re.finditer(r'\n\s*[\*\-]{3,}\s*\n', text):
        breaks.append(match.start())

    # Chapter markers
    for match in re.finditer(r'\n\s*(Chapter|CHAPTER|Part|PART)\s+\w+', text):
        breaks.append(match.start())

    return sorted(set(breaks))


def analyze_scene_lengths(text: str) -> Dict:
    """
    Analyze scene length distribution.

    Returns statistics about scene lengths.
    """
    breaks = detect_scene_breaks(text)

    if not breaks:
        return {
            "n_scenes": 1,
            "mean_length": len(text.split()),
            "std_length": 0,
            "min_length": len(text.split()),
            "max_length": len(text.split()),
        }

    # Add start and end
    positions = [0] + breaks + [len(text)]

    scene_lengths = []
    for i in range(len(positions) - 1):
        scene = text[positions[i]:positions[i+1]]
        scene_lengths.append(len(scene.split()))

    return {
        "n_scenes": len(scene_lengths),
        "mean_length": float(np.mean(scene_lengths)),
        "std_length": float(np.std(scene_lengths)),
        "min_length": min(scene_lengths),
        "max_length": max(scene_lengths),
    }


@click.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True))
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path())
@click.option('--language', '-l', default='en', type=click.Choice(['en', 'ja']))
@click.option('--window-size', '-w', default=1000)
@click.option('--overlap', default=500)
def main(input_dir: str, output_dir: str, language: str, window_size: int, overlap: int):
    """Extract pacing trajectories from text corpus."""
    from .entropy import process_corpus as _process  # Reuse pattern

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if language == "ja":
        functor = JapanesePacingFunctor()
    else:
        functor = PacingFunctor()

    text_files = list(Path(input_dir).glob("*.json"))
    text_files = [f for f in text_files if f.name != "manifest.json"]
    console.print(f"[blue]Processing {len(text_files)} texts for pacing...[/blue]")

    results = []

    for text_file in tqdm(text_files, desc="Extracting pacing"):
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

            trajectory = functor(windows)

            trajectory.metadata.update({
                "source_id": data.get("id", text_file.stem),
                "title": data.get("title", "Unknown"),
                "language": language,
            })

            out_file = output_dir / f"{text_file.stem}_pacing.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory.to_dict(), f, ensure_ascii=False, indent=2)

            results.append({
                "id": data.get("id", text_file.stem),
                "title": data.get("title"),
                "mean_pacing": float(np.mean(trajectory.values)),
                "pacing_variance": float(np.var(trajectory.values)),
                "trajectory_file": str(out_file.name),
            })

        except Exception as e:
            console.print(f"[red]Error processing {text_file}: {e}[/red]")
            continue

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

    console.print(f"[bold green]✓ Processed {len(results)} texts for pacing[/bold green]")


if __name__ == "__main__":
    main()
