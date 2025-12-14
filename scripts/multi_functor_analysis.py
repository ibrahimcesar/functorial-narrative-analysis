#!/usr/bin/env python3
"""
Multi-Functor Narrative Analysis

Addresses Limitation 7.2: Run analysis using multiple functors beyond sentiment:
- Sentiment (emotional valence)
- Entropy (complexity/predictability)
- Arousal (tension/excitement)
- Epistemic (certainty/uncertainty)
- Pacing (scene length, dialogue density)
- Character Presence (named entity tracking)
- Narrative Voice (POV detection)

This enables richer characterization of narrative structure and cross-validation
of ICC classifications.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
import click
from rich.console import Console
from rich.table import Table
from scipy.stats import pearsonr

# Import functors
from src.functors.sentiment import SentimentFunctor
from src.functors.entropy import EntropyFunctor, JapaneseEntropyFunctor
from src.functors.arousal import ArousalFunctor, JapaneseArousalFunctor
from src.functors.epistemic import EpistemicFunctor, JapaneseEpistemicFunctor
from src.functors.pacing import PacingFunctor, JapanesePacingFunctor
from src.functors.character_presence import CharacterPresenceFunctor, JapaneseCharacterPresenceFunctor
from src.functors.narrative_voice import NarrativeVoiceFunctor, JapaneseNarrativeVoiceFunctor
from src.detectors.icc import ICCDetector

console = Console()


@dataclass
class MultiFunctorResult:
    """Results from multi-functor analysis."""
    title: str
    source_id: str

    # ICC classification (from sentiment)
    icc_class: str
    icc_confidence: float

    # Functor trajectories summary
    sentiment_mean: float
    sentiment_std: float
    entropy_mean: float
    entropy_std: float
    arousal_mean: float
    arousal_std: float
    epistemic_mean: float
    epistemic_std: float
    pacing_mean: float
    pacing_std: float
    character_presence_mean: float
    character_presence_std: float
    narrative_distance_mean: float
    narrative_distance_std: float

    # Cross-functor correlations
    sentiment_entropy_corr: float
    sentiment_arousal_corr: float
    sentiment_epistemic_corr: float
    arousal_entropy_corr: float
    pacing_arousal_corr: float
    sentiment_pacing_corr: float

    # Derived metrics
    complexity_score: float  # Combined entropy + arousal variance
    narrative_tension_profile: str  # High/Medium/Low
    epistemic_pattern: str  # From epistemic functor
    dominant_pov: str  # From narrative voice functor
    protagonist_dominance: float  # From character presence functor


def create_windows(text: str, window_size: int = 1000, overlap: int = 500,
                   language: str = "en") -> List[str]:
    """Create overlapping windows from text."""
    if language == "ja":
        text_clean = re.sub(r'\s+', '', text)
        step = window_size - overlap
        windows = []
        for i in range(0, len(text_clean), step):
            window = text_clean[i:i + window_size]
            if len(window) >= window_size // 2:
                windows.append(window)
        return windows if windows else [text]
    else:
        words = text.split()
        step = window_size - overlap
        windows = []
        for i in range(0, len(words), step):
            window = ' '.join(words[i:i + window_size])
            if len(window.split()) >= window_size // 2:
                windows.append(window)
        return windows if windows else [text]


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute correlation, handling edge cases."""
    if len(x) < 3 or len(y) < 3:
        return 0.0
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    try:
        corr, _ = pearsonr(x, y)
        return float(corr) if not np.isnan(corr) else 0.0
    except:
        return 0.0


def analyze_text_multi_functor(
    text: str,
    title: str = "Unknown",
    source_id: str = "unknown",
    language: str = "en",
    window_size: int = 1000,
    overlap: int = 500
) -> MultiFunctorResult:
    """
    Analyze a single text with multiple functors.

    Args:
        text: Full text content
        title: Text title
        source_id: Source identifier
        language: "en" or "ja"
        window_size: Window size for analysis
        overlap: Window overlap

    Returns:
        MultiFunctorResult with comprehensive analysis
    """
    # Create windows
    windows = create_windows(text, window_size, overlap, language)

    # Initialize functors based on language
    if language == "ja":
        sentiment_functor = SentimentFunctor(method="bert")
        entropy_functor = JapaneseEntropyFunctor()
        arousal_functor = JapaneseArousalFunctor()
        epistemic_functor = JapaneseEpistemicFunctor()
        pacing_functor = JapanesePacingFunctor()
        character_functor = JapaneseCharacterPresenceFunctor()
        voice_functor = JapaneseNarrativeVoiceFunctor()
    else:
        sentiment_functor = SentimentFunctor(method="vader")
        entropy_functor = EntropyFunctor(method="combined")
        arousal_functor = ArousalFunctor()
        epistemic_functor = EpistemicFunctor()
        pacing_functor = PacingFunctor()
        character_functor = CharacterPresenceFunctor()
        voice_functor = NarrativeVoiceFunctor()

    # Apply functors
    sentiment_traj = sentiment_functor(windows)
    entropy_traj = entropy_functor(windows)
    arousal_traj = arousal_functor(windows)
    epistemic_traj = epistemic_functor(windows)
    pacing_traj = pacing_functor(windows)
    character_traj = character_functor(windows)
    voice_traj = voice_functor(windows)

    # ICC classification from sentiment
    detector = ICCDetector()
    icc_result = detector.detect(sentiment_traj.values)

    # Epistemic pattern detection
    from src.functors.epistemic import EpistemicPatternDetector
    pattern_detector = EpistemicPatternDetector()
    epistemic_pattern = pattern_detector.detect(epistemic_traj)

    # Compute correlations
    sentiment_entropy_corr = safe_correlation(sentiment_traj.values, entropy_traj.values)
    sentiment_arousal_corr = safe_correlation(sentiment_traj.values, arousal_traj.values)
    sentiment_epistemic_corr = safe_correlation(sentiment_traj.values, epistemic_traj.values)
    arousal_entropy_corr = safe_correlation(arousal_traj.values, entropy_traj.values)
    pacing_arousal_corr = safe_correlation(pacing_traj.values, arousal_traj.values)
    sentiment_pacing_corr = safe_correlation(sentiment_traj.values, pacing_traj.values)

    # Complexity score: combines entropy and arousal variance
    complexity_score = float(np.mean(entropy_traj.values) * 0.5 +
                            np.std(arousal_traj.values) * 0.5)

    # Tension profile based on arousal
    mean_arousal = np.mean(arousal_traj.values)
    if mean_arousal > 0.6:
        tension_profile = "High"
    elif mean_arousal > 0.4:
        tension_profile = "Medium"
    else:
        tension_profile = "Low"

    # Extract metadata from new functors
    dominant_pov = voice_traj.metadata.get("dominant_pov", "unknown")
    protagonist_dominance = character_traj.metadata.get("protagonist_dominance", 0.0)

    return MultiFunctorResult(
        title=title,
        source_id=source_id,
        icc_class=icc_result.icc_class,
        icc_confidence=icc_result.confidence,
        sentiment_mean=float(np.mean(sentiment_traj.values)),
        sentiment_std=float(np.std(sentiment_traj.values)),
        entropy_mean=float(np.mean(entropy_traj.values)),
        entropy_std=float(np.std(entropy_traj.values)),
        arousal_mean=float(np.mean(arousal_traj.values)),
        arousal_std=float(np.std(arousal_traj.values)),
        epistemic_mean=float(np.mean(epistemic_traj.values)),
        epistemic_std=float(np.std(epistemic_traj.values)),
        pacing_mean=float(np.mean(pacing_traj.values)),
        pacing_std=float(np.std(pacing_traj.values)),
        character_presence_mean=float(np.mean(character_traj.values)),
        character_presence_std=float(np.std(character_traj.values)),
        narrative_distance_mean=float(np.mean(voice_traj.values)),
        narrative_distance_std=float(np.std(voice_traj.values)),
        sentiment_entropy_corr=sentiment_entropy_corr,
        sentiment_arousal_corr=sentiment_arousal_corr,
        sentiment_epistemic_corr=sentiment_epistemic_corr,
        arousal_entropy_corr=arousal_entropy_corr,
        pacing_arousal_corr=pacing_arousal_corr,
        sentiment_pacing_corr=sentiment_pacing_corr,
        complexity_score=complexity_score,
        narrative_tension_profile=tension_profile,
        epistemic_pattern=epistemic_pattern["pattern"],
        dominant_pov=dominant_pov,
        protagonist_dominance=float(protagonist_dominance)
    )


def process_corpus_multi_functor(
    input_dir: Path,
    output_dir: Path,
    language: str = "en",
    window_size: int = 1000,
    overlap: int = 500
) -> List[MultiFunctorResult]:
    """
    Process entire corpus with multi-functor analysis.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_files = list(Path(input_dir).glob("*.json"))
    text_files = [f for f in text_files if f.name not in ["manifest.json", "metadata.json"]]

    console.print(f"[blue]Processing {len(text_files)} texts with multi-functor analysis...[/blue]")

    results = []

    from tqdm import tqdm
    for text_file in tqdm(text_files, desc="Multi-functor analysis"):
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            text = data.get("text", "")
            if not text or len(text) < 500:
                continue

            result = analyze_text_multi_functor(
                text=text,
                title=data.get("title", "Unknown"),
                source_id=data.get("id", text_file.stem),
                language=language,
                window_size=window_size,
                overlap=overlap
            )

            results.append(result)

            # Save individual result
            out_file = output_dir / f"{text_file.stem}_multi.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, ensure_ascii=False, indent=2)

        except Exception as e:
            console.print(f"[red]Error processing {text_file}: {e}[/red]")
            continue

    # Save summary
    summary = {
        "count": len(results),
        "language": language,
        "results": [asdict(r) for r in results]
    }

    with open(output_dir / "multi_functor_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    console.print(f"[bold green]✓ Processed {len(results)} texts[/bold green]")

    return results


def display_results(results: List[MultiFunctorResult]):
    """Display results in a rich table."""
    table = Table(title="Multi-Functor Analysis Results")

    table.add_column("Title", style="cyan", max_width=25)
    table.add_column("ICC", style="magenta")
    table.add_column("Sent", justify="right")
    table.add_column("Pace", justify="right")
    table.add_column("Char", justify="right")
    table.add_column("POV", style="green")
    table.add_column("Tension", style="yellow")

    for r in results[:20]:  # Show first 20
        table.add_row(
            r.title[:23] + "..." if len(r.title) > 25 else r.title,
            r.icc_class,
            f"{r.sentiment_mean:.2f}",
            f"{r.pacing_mean:.2f}",
            f"{r.character_presence_mean:.2f}",
            r.dominant_pov[:10],
            r.narrative_tension_profile
        )

    console.print(table)

    # Extended table with more metrics
    table2 = Table(title="Additional Metrics")
    table2.add_column("Title", style="cyan", max_width=25)
    table2.add_column("Entropy", justify="right")
    table2.add_column("Arousal", justify="right")
    table2.add_column("Epistemic", justify="right")
    table2.add_column("Protag Dom", justify="right")
    table2.add_column("Pattern", style="blue")

    for r in results[:20]:
        table2.add_row(
            r.title[:23] + "..." if len(r.title) > 25 else r.title,
            f"{r.entropy_mean:.2f}",
            f"{r.arousal_mean:.2f}",
            f"{r.epistemic_mean:.2f}",
            f"{r.protagonist_dominance:.2f}",
            r.epistemic_pattern[:12]
        )

    console.print(table2)

    # Correlation summary
    console.print("\n[bold]Cross-Functor Correlations (averages):[/bold]")
    avg_se = np.mean([r.sentiment_entropy_corr for r in results])
    avg_sa = np.mean([r.sentiment_arousal_corr for r in results])
    avg_sep = np.mean([r.sentiment_epistemic_corr for r in results])
    avg_ae = np.mean([r.arousal_entropy_corr for r in results])
    avg_pa = np.mean([r.pacing_arousal_corr for r in results])
    avg_sp = np.mean([r.sentiment_pacing_corr for r in results])

    console.print(f"  Sentiment ↔ Entropy:   {avg_se:+.3f}")
    console.print(f"  Sentiment ↔ Arousal:   {avg_sa:+.3f}")
    console.print(f"  Sentiment ↔ Epistemic: {avg_sep:+.3f}")
    console.print(f"  Sentiment ↔ Pacing:    {avg_sp:+.3f}")
    console.print(f"  Arousal ↔ Entropy:     {avg_ae:+.3f}")
    console.print(f"  Pacing ↔ Arousal:      {avg_pa:+.3f}")


@click.command()
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=True), help="Input file or directory")
@click.option('--output', '-o', 'output_dir', default='output/multi_functor',
              type=click.Path(), help="Output directory")
@click.option('--language', '-l', default='en', type=click.Choice(['en', 'ja']),
              help="Text language")
@click.option('--window-size', '-w', default=1000, help="Window size")
@click.option('--overlap', default=500, help="Window overlap")
def main(input_path: str, output_dir: str, language: str, window_size: int, overlap: int):
    """
    Run multi-functor narrative analysis.

    Analyzes texts using seven functors for comprehensive narrative characterization:
    - Sentiment (emotional valence)
    - Entropy (lexical complexity)
    - Arousal (tension/excitement)
    - Epistemic (certainty/uncertainty)
    - Pacing (scene length, dialogue density)
    - Character presence (named entity tracking)
    - Narrative voice (POV detection)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if input_path.is_file():
        # Single file analysis
        with open(input_path, 'r', encoding='utf-8') as f:
            if input_path.suffix == '.json':
                data = json.load(f)
                text = data.get("text", "")
                title = data.get("title", input_path.stem)
            else:
                text = f.read()
                title = input_path.stem

        result = analyze_text_multi_functor(
            text=text,
            title=title,
            source_id=input_path.stem,
            language=language,
            window_size=window_size,
            overlap=overlap
        )

        display_results([result])

        # Save result
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{input_path.stem}_multi.json", 'w') as f:
            json.dump(asdict(result), f, indent=2)

    else:
        # Directory processing
        results = process_corpus_multi_functor(
            input_path, output_dir, language, window_size, overlap
        )
        display_results(results)


if __name__ == "__main__":
    main()
