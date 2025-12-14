"""
F_character: Character Presence Functor

Maps narrative states to character presence/focus measurements.
Tracks named entities (characters) throughout the narrative to understand:
- Protagonist focus patterns
- Character ensemble dynamics
- Introduction/exit patterns
- Character interaction density

This functor captures "who is on stage" at any given moment in the narrative.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm
import click
from rich.console import Console

from .base import BaseFunctor, Trajectory

console = Console()


# Common titles and honorifics to help identify character names
ENGLISH_TITLES = {
    'mr', 'mrs', 'ms', 'miss', 'dr', 'doctor', 'prof', 'professor',
    'sir', 'lord', 'lady', 'captain', 'colonel', 'general', 'major',
    'king', 'queen', 'prince', 'princess', 'duke', 'duchess',
    'count', 'countess', 'baron', 'baroness', 'father', 'mother',
    'brother', 'sister', 'uncle', 'aunt', 'reverend', 'pastor',
}

# Common non-character capitalized words to filter out
ENGLISH_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his',
    'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
    'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'january', 'february', 'march', 'april', 'may', 'june', 'july',
    'august', 'september', 'october', 'november', 'december',
    'chapter', 'part', 'book', 'volume', 'section',
}

# Japanese name patterns
JAPANESE_NAME_SUFFIXES = ['さん', 'くん', 'ちゃん', '様', '殿', '氏', '君', '嬢']


class CharacterExtractor:
    """
    Extract character names from text using pattern-based NER.

    Uses heuristics rather than ML models for portability:
    - Capitalized word sequences (English)
    - Title + Name patterns
    - Dialogue attribution ("said X", "X replied")
    - Recurring proper nouns
    """

    def __init__(self, language: str = "en"):
        self.language = language
        self.titles = ENGLISH_TITLES
        self.stopwords = ENGLISH_STOPWORDS

    def extract_english_names(self, text: str) -> List[str]:
        """Extract potential character names from English text."""
        names = []

        # Pattern 1: Title + Capitalized Name
        title_pattern = r'\b(' + '|'.join(self.titles) + r')\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        for match in re.finditer(title_pattern, text, re.IGNORECASE):
            name = match.group(2)
            names.append(name)

        # Pattern 2: Dialogue attribution
        # "said Alice", "Alice said", "replied Holmes"
        dialogue_patterns = [
            r'(?:said|replied|asked|answered|exclaimed|shouted|whispered|muttered|cried)\s+([A-Z][a-z]+)',
            r'([A-Z][a-z]+)\s+(?:said|replied|asked|answered|exclaimed|shouted|whispered|muttered|cried)',
        ]
        for pattern in dialogue_patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1)
                if name.lower() not in self.stopwords:
                    names.append(name)

        # Pattern 3: Capitalized words that appear multiple times
        # (likely to be character names in narrative)
        cap_words = re.findall(r'\b([A-Z][a-z]{2,})\b', text)
        word_counts = Counter(cap_words)
        for word, count in word_counts.items():
            if count >= 2 and word.lower() not in self.stopwords:
                names.extend([word] * count)

        return names

    def extract_japanese_names(self, text: str) -> List[str]:
        """Extract potential character names from Japanese text."""
        names = []

        # Pattern 1: Name + suffix (さん, くん, etc.)
        for suffix in JAPANESE_NAME_SUFFIXES:
            # Match katakana or kanji before suffix
            pattern = r'([ァ-ヶー]{2,}|[一-龯]{1,4})' + re.escape(suffix)
            for match in re.finditer(pattern, text):
                names.append(match.group(1))

        # Pattern 2: Dialogue attribution patterns
        # 「...」と太郎は言った
        dialogue_pattern = r'」と([ァ-ヶー]{2,}|[一-龯]{1,4})(?:は|が|の)'
        for match in re.finditer(dialogue_pattern, text):
            names.append(match.group(1))

        # Pattern 3: Katakana sequences (often names in novels)
        katakana_names = re.findall(r'[ァ-ヶー]{3,}', text)
        names.extend(katakana_names)

        return names

    def extract(self, text: str) -> List[str]:
        """Extract character names from text."""
        if self.language == "ja":
            return self.extract_japanese_names(text)
        else:
            return self.extract_english_names(text)


class CharacterPresenceFunctor(BaseFunctor):
    """
    Character presence functor measuring narrative focus on characters.

    Maps text windows to character presence scores in [0, 1] where:
    - 0 = minimal character focus (description, setting)
    - 1 = high character focus (dialogue-heavy, character-centric)

    Also tracks:
    - Character diversity (how many characters appear)
    - Protagonist dominance (focus on main character vs ensemble)
    - New character introductions
    """

    name = "character_presence"

    def __init__(self, language: str = "en", track_individuals: bool = True):
        """
        Initialize character presence functor.

        Args:
            language: "en" or "ja"
            track_individuals: Whether to track individual character mentions
        """
        self.language = language
        self.track_individuals = track_individuals
        self.extractor = CharacterExtractor(language)
        self._global_characters: Counter = Counter()

    def _score_window(self, text: str) -> Tuple[float, Dict]:
        """
        Compute character presence score for a text window.

        Returns:
            Tuple of (score, metadata)
        """
        # Extract character mentions
        names = self.extractor.extract(text)
        name_counts = Counter(names)

        # Update global tracking
        self._global_characters.update(name_counts)

        # Calculate metrics
        total_mentions = len(names)
        unique_characters = len(name_counts)

        # Normalize by text length
        if self.language == "ja":
            text_length = len(text.replace(' ', '').replace('\n', ''))
        else:
            text_length = len(text.split())

        if text_length == 0:
            return 0.5, {"mentions": 0, "unique": 0}

        # Character density: mentions per 100 words/characters
        density = (total_mentions / text_length) * 100

        # Normalize to [0, 1]
        # Typical range: 0-20 mentions per 100 words
        presence_score = min(1.0, density / 15)

        # Diversity bonus: more characters = richer scene
        if unique_characters > 3:
            diversity_bonus = min(0.1, (unique_characters - 3) * 0.02)
            presence_score = min(1.0, presence_score + diversity_bonus)

        metadata = {
            "total_mentions": total_mentions,
            "unique_characters": unique_characters,
            "top_characters": dict(name_counts.most_common(5)),
            "density": density,
        }

        return presence_score, metadata

    def __call__(self, windows: List[str]) -> Trajectory:
        """
        Apply character presence functor to text windows.

        Args:
            windows: List of text windows

        Returns:
            Trajectory with character presence scores
        """
        # Reset global tracking for new text
        self._global_characters = Counter()

        scores = []
        window_metadata = []

        for window in windows:
            score, meta = self._score_window(window)
            scores.append(score)
            window_metadata.append(meta)

        values = np.array(scores)
        time_points = np.linspace(0, 1, len(values))

        # Compute global character statistics
        top_characters = dict(self._global_characters.most_common(10))
        total_unique = len(self._global_characters)

        # Protagonist dominance: how much does the top character dominate?
        if self._global_characters:
            top_count = self._global_characters.most_common(1)[0][1]
            total_count = sum(self._global_characters.values())
            protagonist_dominance = top_count / total_count if total_count > 0 else 0
        else:
            protagonist_dominance = 0

        return Trajectory(
            values=values,
            time_points=time_points,
            functor_name=self.name,
            metadata={
                "language": self.language,
                "n_windows": len(windows),
                "total_unique_characters": total_unique,
                "top_characters": top_characters,
                "protagonist_dominance": protagonist_dominance,
                "mean_presence": float(np.mean(values)),
                "presence_variance": float(np.var(values)),
            }
        )


class JapaneseCharacterPresenceFunctor(CharacterPresenceFunctor):
    """Character presence functor optimized for Japanese text."""

    name = "character_presence_ja"

    def __init__(self, track_individuals: bool = True):
        super().__init__(language="ja", track_individuals=track_individuals)


class CharacterArcAnalyzer:
    """
    Analyze character appearance patterns across the narrative.

    Detects patterns like:
    - Protagonist arc (main character throughout)
    - Ensemble (multiple characters, balanced)
    - Rotating focus (different characters in different sections)
    - Late introduction (major characters appear late)
    - Early exit (characters disappear)
    """

    def __init__(self):
        self.patterns = {
            "protagonist": "Single dominant character throughout",
            "ensemble": "Multiple balanced characters",
            "rotating_focus": "Focus shifts between characters",
            "convergent": "Characters come together over time",
            "divergent": "Characters separate over time",
        }

    def analyze(self, trajectory: Trajectory, window_metadata: List[Dict] = None) -> Dict:
        """
        Analyze character arc patterns from trajectory.

        Args:
            trajectory: Character presence trajectory
            window_metadata: Optional per-window character data

        Returns:
            Dict with arc analysis
        """
        values = trajectory.values
        n = len(values)

        if n < 4:
            return {"pattern": "unknown", "confidence": 0.0}

        # Analyze presence variance
        presence_var = np.var(values)
        mean_presence = np.mean(values)

        # Check trajectory metadata for character info
        meta = trajectory.metadata
        protagonist_dominance = meta.get("protagonist_dominance", 0)
        total_unique = meta.get("total_unique_characters", 0)

        # Classify pattern
        if protagonist_dominance > 0.5:
            pattern = "protagonist"
            confidence = protagonist_dominance
        elif total_unique > 5 and protagonist_dominance < 0.3:
            pattern = "ensemble"
            confidence = 1 - protagonist_dominance
        elif presence_var > 0.05:
            # High variance suggests rotating focus
            pattern = "rotating_focus"
            confidence = min(1.0, presence_var * 5)
        else:
            # Check for convergent/divergent
            first_half = np.mean(values[:n//2])
            second_half = np.mean(values[n//2:])

            if second_half > first_half + 0.1:
                pattern = "convergent"
                confidence = second_half - first_half
            elif first_half > second_half + 0.1:
                pattern = "divergent"
                confidence = first_half - second_half
            else:
                pattern = "stable"
                confidence = 1 - abs(first_half - second_half)

        return {
            "pattern": pattern,
            "confidence": float(confidence),
            "protagonist_dominance": protagonist_dominance,
            "total_unique_characters": total_unique,
            "mean_presence": float(mean_presence),
            "presence_variance": float(presence_var),
        }


def process_corpus(
    input_dir: Path,
    output_dir: Path,
    language: str = "en",
    window_size: int = 1000,
    overlap: int = 500
):
    """
    Process a corpus through the character presence functor.

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
        functor = JapaneseCharacterPresenceFunctor()
    else:
        functor = CharacterPresenceFunctor()

    arc_analyzer = CharacterArcAnalyzer()

    # Find text files
    text_files = list(Path(input_dir).glob("*.json"))
    text_files = [f for f in text_files if f.name not in ["manifest.json", "metadata.json"]]
    console.print(f"[blue]Processing {len(text_files)} texts for character presence...[/blue]")

    results = []

    for text_file in tqdm(text_files, desc="Extracting character presence"):
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

            # Analyze character arc
            arc_info = arc_analyzer.analyze(trajectory)

            trajectory.metadata.update({
                "source_id": data.get("id", text_file.stem),
                "title": data.get("title", "Unknown"),
                "language": language,
                "character_arc_pattern": arc_info["pattern"],
            })

            # Save trajectory
            out_file = output_dir / f"{text_file.stem}_character.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory.to_dict(), f, ensure_ascii=False, indent=2)

            results.append({
                "id": data.get("id", text_file.stem),
                "title": data.get("title"),
                "mean_presence": float(np.mean(trajectory.values)),
                "unique_characters": trajectory.metadata.get("total_unique_characters", 0),
                "protagonist_dominance": trajectory.metadata.get("protagonist_dominance", 0),
                "arc_pattern": arc_info["pattern"],
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

    console.print(f"[bold green]✓ Processed {len(results)} texts for character presence[/bold green]")


@click.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True))
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path())
@click.option('--language', '-l', default='en', type=click.Choice(['en', 'ja']))
@click.option('--window-size', '-w', default=1000)
@click.option('--overlap', default=500)
def main(input_dir: str, output_dir: str, language: str, window_size: int, overlap: int):
    """Extract character presence trajectories from text corpus."""
    process_corpus(Path(input_dir), Path(output_dir), language, window_size, overlap)


if __name__ == "__main__":
    main()
