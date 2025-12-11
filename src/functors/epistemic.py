"""
F_epistemic: Epistemic State Functor

Maps narrative states to positions on an information/knowledge axis,
measuring:
- Certainty vs uncertainty in narrative voice
- Revelation and discovery patterns
- Mystery and suspense dynamics

This functor captures the "knowledge shape" of a narrative -
how information is revealed to the reader and how characters'
epistemic states evolve.

Key patterns:
- Mystery: High uncertainty → resolution (decreasing)
- Discovery: Progressive revelation (increasing certainty)
- Suspense: Oscillating uncertainty
- Exposition: High certainty throughout
"""

import json
import re
from pathlib import Path
from typing import List, Set

import numpy as np
from tqdm import tqdm
import click
from rich.console import Console

from .base import BaseFunctor, Trajectory

console = Console()


# Epistemic markers for uncertainty
UNCERTAINTY_MARKERS = {
    # Modal verbs of uncertainty
    'might', 'may', 'could', 'would', 'should', 'perhaps', 'maybe',
    'possibly', 'probably', 'likely', 'unlikely', 'apparently',

    # Hedging expressions
    'seem', 'seems', 'seemed', 'appear', 'appears', 'appeared',
    'suggest', 'suggests', 'suggested', 'suppose', 'supposed',
    'guess', 'guessed', 'wonder', 'wondered', 'wondering',

    # Uncertainty nouns
    'doubt', 'question', 'mystery', 'enigma', 'puzzle', 'riddle',
    'uncertainty', 'confusion', 'bewilderment', 'perplexity',

    # Question indicators
    'who', 'what', 'where', 'when', 'why', 'how', 'whether',

    # Negation of knowledge
    'unknown', 'unclear', 'uncertain', 'unsure', 'ignorant',
    "didn't know", "don't know", "couldn't tell", "can't tell",

    # Speculation
    'if', 'unless', 'suppose', 'assuming', 'presumably',
    'hypothetically', 'theoretically', 'conceivably',
}

# Epistemic markers for certainty
CERTAINTY_MARKERS = {
    # Modal verbs of certainty
    'must', 'will', 'shall', 'certainly', 'definitely', 'surely',
    'undoubtedly', 'inevitably', 'absolutely', 'clearly',

    # Knowledge verbs
    'know', 'knew', 'known', 'understand', 'understood', 'realize',
    'realized', 'discover', 'discovered', 'find', 'found', 'learn',
    'learned', 'reveal', 'revealed', 'recognize', 'recognized',

    # Certainty expressions
    'obvious', 'evident', 'apparent', 'plain', 'clear', 'certain',
    'sure', 'positive', 'confident', 'convinced', 'decided',

    # Factual indicators
    'fact', 'truth', 'actually', 'indeed', 'truly', 'really',
    'of course', 'naturally', 'obviously', 'evidently',

    # Conclusion markers
    'therefore', 'thus', 'hence', 'consequently', 'accordingly',
    'conclude', 'concluded', 'determine', 'determined',
}

# Discovery/revelation markers
REVELATION_MARKERS = {
    'discover', 'discovered', 'reveal', 'revealed', 'uncover',
    'uncovered', 'expose', 'exposed', 'find out', 'found out',
    'realize', 'realized', 'learn', 'learned', 'understand',
    'understood', 'see', 'saw', 'notice', 'noticed', 'recognize',
    'recognized', 'comprehend', 'grasp', 'grasped', 'dawn',
    'suddenly', 'finally', 'at last', 'truth', 'secret',
}


class EpistemicFunctor(BaseFunctor):
    """
    Epistemic functor measuring narrative certainty/uncertainty.

    Maps text windows to scores in [0, 1] where:
    - 0 = high uncertainty, mystery, unanswered questions
    - 1 = high certainty, resolution, revealed information
    """

    name = "epistemic"

    def __init__(self, method: str = "lexicon"):
        """
        Initialize epistemic functor.

        Args:
            method: "lexicon" for dictionary-based, "hybrid" for combined
        """
        self.method = method
        self.uncertainty = UNCERTAINTY_MARKERS
        self.certainty = CERTAINTY_MARKERS
        self.revelation = REVELATION_MARKERS

    def _count_questions(self, text: str) -> int:
        """Count question marks as uncertainty indicators."""
        return text.count('?')

    def _count_markers(self, text: str, markers: Set[str]) -> int:
        """Count occurrences of markers in text."""
        text_lower = text.lower()
        count = 0
        for marker in markers:
            # Handle multi-word markers
            if ' ' in marker:
                count += text_lower.count(marker)
            else:
                # Word boundary matching
                pattern = r'\b' + re.escape(marker) + r'\b'
                count += len(re.findall(pattern, text_lower))
        return count

    def _score_window(self, text: str) -> float:
        """
        Compute epistemic score for a text window.

        Higher score = more certainty/knowledge
        Lower score = more uncertainty/mystery
        """
        uncertainty_count = self._count_markers(text, self.uncertainty)
        certainty_count = self._count_markers(text, self.certainty)
        revelation_count = self._count_markers(text, self.revelation)
        question_count = self._count_questions(text)

        # Questions add to uncertainty
        uncertainty_count += question_count * 0.5

        # Revelations add to certainty (knowledge being revealed)
        certainty_count += revelation_count * 0.75

        total = uncertainty_count + certainty_count
        if total == 0:
            return 0.5  # Neutral

        # Score as proportion of certainty
        score = certainty_count / (total + 1)

        # Normalize to [0, 1]
        return max(0.0, min(1.0, score))

    def __call__(self, windows: List[str]) -> Trajectory:
        """
        Apply epistemic functor to text windows.

        Args:
            windows: List of text windows

        Returns:
            Trajectory with epistemic scores
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


# Japanese epistemic markers
UNCERTAINTY_JA = [
    'かもしれない', 'かも', 'だろう', 'でしょう', 'らしい', 'ようだ',
    '思う', '考える', '疑う', '不思議', '謎', '疑問', '知らない',
    '分からない', 'わからない', 'はず', '多分', 'おそらく', 'たぶん',
    'もしかして', 'もしか', '不明', '不確か', '曖昧',
]

CERTAINTY_JA = [
    '確か', '確実', '明らか', '当然', 'もちろん', '無論', '必ず',
    '絶対', '知る', '知った', '分かる', '分かった', 'わかる', 'わかった',
    '理解', '発見', '気づく', '気づいた', '悟る', '悟った', '判明',
    '真実', '事実', '本当', '実際', 'はっきり', '明確',
]

REVELATION_JA = [
    '発見', '発覚', '明らかに', '判明', '気づく', '気づいた',
    '悟る', '悟った', '知る', '知った', '真相', '秘密', '暴く',
    'ついに', 'やっと', 'とうとう', '実は', '実際',
]


class JapaneseEpistemicFunctor(BaseFunctor):
    """Epistemic functor for Japanese text."""

    name = "epistemic_ja"

    def __init__(self):
        self.uncertainty = UNCERTAINTY_JA
        self.certainty = CERTAINTY_JA
        self.revelation = REVELATION_JA

    def _score_window(self, text: str) -> float:
        """Compute epistemic score for Japanese text window."""
        uncertainty_count = sum(1 for marker in self.uncertainty if marker in text)
        certainty_count = sum(1 for marker in self.certainty if marker in text)
        revelation_count = sum(1 for marker in self.revelation if marker in text)

        # Question marks
        uncertainty_count += (text.count('?') + text.count('？')) * 0.5

        # Revelations add to certainty
        certainty_count += revelation_count * 0.75

        total = uncertainty_count + certainty_count
        if total == 0:
            return 0.5

        return max(0.0, min(1.0, certainty_count / (total + 1)))

    def __call__(self, windows: List[str]) -> Trajectory:
        scores = [self._score_window(window) for window in windows]

        return Trajectory(
            values=np.array(scores),
            time_points=np.linspace(0, 1, len(scores)),
            functor_name=self.name,
            metadata={"language": "ja", "n_windows": len(windows)}
        )


class EpistemicPatternDetector:
    """
    Detects epistemic narrative patterns.

    Patterns:
    - Mystery: Starts uncertain, ends certain (upward slope)
    - Discovery: Progressive revelation (gradual increase)
    - Suspense: Oscillating uncertainty (high variance)
    - Exposition: High certainty throughout (flat high)
    - Ambiguity: Low certainty throughout (flat low)
    """

    def __init__(self):
        self.patterns = {
            "mystery_resolution": "Start uncertain → end certain",
            "progressive_discovery": "Gradual increase in certainty",
            "suspense": "Oscillating uncertainty",
            "exposition": "High certainty throughout",
            "ambiguity": "Low certainty throughout",
            "reverse_mystery": "Start certain → end uncertain",
        }

    def detect(self, trajectory: Trajectory) -> dict:
        """
        Detect epistemic pattern in trajectory.

        Returns:
            dict with pattern classification and confidence
        """
        values = trajectory.values
        n = len(values)

        if n < 4:
            return {"pattern": "unknown", "confidence": 0.0}

        # Compute features
        mean_val = np.mean(values)
        std_val = np.std(values)
        start_val = np.mean(values[:n//4])
        end_val = np.mean(values[-n//4:])
        slope = end_val - start_val

        # Classify pattern
        if mean_val > 0.65 and std_val < 0.15:
            pattern = "exposition"
            confidence = mean_val
        elif mean_val < 0.35 and std_val < 0.15:
            pattern = "ambiguity"
            confidence = 1 - mean_val
        elif slope > 0.2 and std_val < 0.2:
            pattern = "progressive_discovery"
            confidence = slope
        elif slope > 0.25:
            pattern = "mystery_resolution"
            confidence = slope
        elif slope < -0.25:
            pattern = "reverse_mystery"
            confidence = abs(slope)
        elif std_val > 0.2:
            pattern = "suspense"
            confidence = std_val
        else:
            pattern = "mixed"
            confidence = 0.5

        return {
            "pattern": pattern,
            "confidence": float(confidence),
            "mean_certainty": float(mean_val),
            "certainty_variance": float(std_val),
            "start_to_end_slope": float(slope),
        }


def process_corpus(
    input_dir: Path,
    output_dir: Path,
    language: str = "en",
    window_size: int = 1000,
    overlap: int = 500
):
    """Process a corpus through the epistemic functor."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if language == "ja":
        functor = JapaneseEpistemicFunctor()
    else:
        functor = EpistemicFunctor()

    pattern_detector = EpistemicPatternDetector()

    text_files = list(Path(input_dir).glob("*.json"))
    text_files = [f for f in text_files if f.name != "manifest.json"]
    console.print(f"[blue]Processing {len(text_files)} texts for epistemic analysis...[/blue]")

    results = []

    for text_file in tqdm(text_files, desc="Extracting epistemic trajectories"):
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

            # Detect pattern
            pattern_info = pattern_detector.detect(trajectory)

            trajectory.metadata.update({
                "source_id": data.get("id", text_file.stem),
                "title": data.get("title", "Unknown"),
                "language": language,
                "epistemic_pattern": pattern_info["pattern"],
                "pattern_confidence": pattern_info["confidence"],
            })

            out_file = output_dir / f"{text_file.stem}_epistemic.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory.to_dict(), f, ensure_ascii=False, indent=2)

            results.append({
                "id": data.get("id", text_file.stem),
                "title": data.get("title"),
                "mean_certainty": float(np.mean(trajectory.values)),
                "std_certainty": float(np.std(trajectory.values)),
                "epistemic_pattern": pattern_info["pattern"],
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

    console.print(f"[bold green]✓ Processed {len(results)} texts for epistemic analysis[/bold green]")


@click.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True))
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path())
@click.option('--language', '-l', default='en', type=click.Choice(['en', 'ja']))
@click.option('--window-size', '-w', default=1000)
@click.option('--overlap', default=500)
def main(input_dir: str, output_dir: str, language: str, window_size: int, overlap: int):
    """Extract epistemic trajectories from text corpus."""
    process_corpus(Path(input_dir), Path(output_dir), language, window_size, overlap)


if __name__ == "__main__":
    main()
