"""
Preprocessing Pipeline for Narrative Analysis

Transforms raw texts into analysis-ready format:
1. Sentence segmentation
2. Sliding window creation
3. Normalization to narrative time axis
"""

import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict

import click
from tqdm import tqdm
from rich.console import Console

console = Console()


@dataclass
class ProcessedText:
    """A preprocessed text ready for functor analysis."""
    id: str
    title: str
    author: str
    source: str
    windows: List[str]
    window_size: int
    overlap: int
    total_words: int
    n_windows: int
    metadata: dict = None

    def to_dict(self) -> dict:
        return asdict(self)


def segment_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Uses simple rule-based segmentation. For production,
    consider using spaCy or nltk.sent_tokenize.

    Args:
        text: Raw text

    Returns:
        List of sentences
    """
    import re

    # Clean text
    text = re.sub(r'\s+', ' ', text)

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Filter empty
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def create_windows(
    text: str,
    window_size: int = 1000,
    overlap: int = 500
) -> List[str]:
    """
    Create overlapping windows from text.

    Args:
        text: Input text
        window_size: Words per window
        overlap: Words overlap between windows

    Returns:
        List of text windows
    """
    words = text.split()
    step = window_size - overlap
    windows = []

    for i in range(0, len(words), step):
        window = ' '.join(words[i:i + window_size])
        # Only include windows that are at least half the target size
        if len(window.split()) >= window_size // 2:
            windows.append(window)

    # Ensure at least one window
    if not windows and words:
        windows = [text]

    return windows


def process_text_file(
    input_path: Path,
    window_size: int = 1000,
    overlap: int = 500
) -> Optional[ProcessedText]:
    """
    Process a single text file.

    Args:
        input_path: Path to JSON file with text
        window_size: Words per window
        overlap: Words overlap

    Returns:
        ProcessedText object or None if processing fails
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        text = data.get("text", "")
        if not text:
            return None

        # Create windows
        windows = create_windows(text, window_size, overlap)

        return ProcessedText(
            id=data.get("id", input_path.stem),
            title=data.get("title", "Unknown"),
            author=data.get("author", "Unknown"),
            source=data.get("source", "unknown"),
            windows=windows,
            window_size=window_size,
            overlap=overlap,
            total_words=len(text.split()),
            n_windows=len(windows),
            metadata={
                "year": data.get("year"),
                "subjects": data.get("subjects", []),
            }
        )

    except Exception as e:
        console.print(f"[red]Error processing {input_path}: {e}[/red]")
        return None


def preprocess_corpus(
    input_dir: Path,
    output_dir: Path,
    window_size: int = 1000,
    overlap: int = 500
):
    """
    Preprocess an entire corpus.

    Args:
        input_dir: Directory with raw text JSON files
        output_dir: Output directory for processed texts
        window_size: Words per window
        overlap: Words overlap
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    text_files = list(input_dir.glob("*.json"))
    # Exclude manifest
    text_files = [f for f in text_files if f.name != "manifest.json"]

    console.print(f"[blue]Preprocessing {len(text_files)} texts...[/blue]")
    console.print(f"[dim]Window size: {window_size}, Overlap: {overlap}[/dim]")

    results = []

    for text_file in tqdm(text_files, desc="Preprocessing"):
        processed = process_text_file(text_file, window_size, overlap)

        if processed:
            # Save processed text
            out_file = output_dir / f"{processed.id}.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(processed.to_dict(), f, ensure_ascii=False, indent=2)

            results.append({
                "id": processed.id,
                "title": processed.title,
                "n_windows": processed.n_windows,
                "total_words": processed.total_words,
            })

    # Save manifest
    manifest = {
        "preprocessing": {
            "window_size": window_size,
            "overlap": overlap,
        },
        "count": len(results),
        "texts": results,
    }

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    console.print(f"[bold green]✓ Preprocessed {len(results)} texts[/bold green]")
    console.print(f"[green]Saved to {output_dir}[/green]")


@click.group()
def cli():
    """Preprocessing commands for narrative analysis."""
    pass


@cli.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True))
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path())
def segment(input_dir: str, output_dir: str):
    """Segment texts into sentences."""
    console.print("[yellow]Sentence segmentation (pass-through for now)[/yellow]")
    # For now, just copy files - full segmentation can be added later
    import shutil
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for f in Path(input_dir).glob("*.json"):
        shutil.copy(f, Path(output_dir) / f.name)
    console.print("[green]✓ Segmentation complete[/green]")


@cli.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True))
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path())
@click.option('--window-size', '-w', default=1000, help='Words per window')
@click.option('--overlap', '-v', default=500, help='Overlap between windows')
def window(input_dir: str, output_dir: str, window_size: int, overlap: int):
    """Create sliding windows from texts."""
    preprocess_corpus(
        Path(input_dir),
        Path(output_dir),
        window_size=window_size,
        overlap=overlap
    )


@cli.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True))
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path())
def normalize(input_dir: str, output_dir: str):
    """Normalize texts to narrative time axis."""
    # For windowed texts, normalization is implicit (windows are sequential)
    console.print("[yellow]Normalization (pass-through - windows are already normalized)[/yellow]")
    import shutil
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for f in Path(input_dir).glob("*.json"):
        shutil.copy(f, Path(output_dir) / f.name)
    console.print("[green]✓ Normalization complete[/green]")


def main():
    cli()


if __name__ == "__main__":
    main()
