"""
Functorial Narrative Analysis CLI

Command-line interface for running the full analysis pipeline.
"""

import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="fna")
def cli():
    """
    Functorial Narrative Analysis (FNA)

    A category-theoretic framework for cross-cultural narrative structure analysis.

    \b
    Quick Start:
        fna analyze text.txt              # Analyze a single text
        fna batch data/corpus/ -o results # Batch process a corpus
        fna compare en_results ja_results # Compare two corpora

    \b
    Full Pipeline:
        fna pipeline --corpus gutenberg --output results/
    """
    pass


@cli.command()
@click.argument('text_file', type=click.Path(exists=True))
@click.option('--functor', '-f', multiple=True,
              type=click.Choice(['sentiment', 'arousal', 'entropy', 'thematic', 'epistemic', 'all']),
              default=['all'], help='Functors to apply')
@click.option('--window-size', '-w', default=1000, help='Window size in words')
@click.option('--overlap', default=500, help='Window overlap')
@click.option('--output', '-o', type=click.Path(), help='Output file (JSON)')
@click.option('--plot', '-p', is_flag=True, help='Generate trajectory plot')
@click.option('--language', '-l', default='en', type=click.Choice(['en', 'ja']),
              help='Text language')
def analyze(text_file: str, functor: tuple, window_size: int, overlap: int,
            output: Optional[str], plot: bool, language: str):
    """
    Analyze a single text file.

    Extracts functor trajectories and detects narrative structures.

    \b
    Examples:
        fna analyze novel.txt
        fna analyze novel.txt -f sentiment -f arousal
        fna analyze novel.txt -o results.json --plot
    """
    import numpy as np
    from .functors import (
        SentimentFunctor, ArousalFunctor, EntropyFunctor,
        ThematicFunctor, EpistemicFunctor,
        JapaneseArousalFunctor, JapaneseEntropyFunctor
    )
    from .detectors import HarmonCircleDetector, KishotenketsuDetector

    console.print(f"[bold blue]Analyzing:[/bold blue] {text_file}")

    # Read text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    word_count = len(text.split())
    console.print(f"  Words: {word_count:,}")

    # Select functors
    functors_to_use = []
    if 'all' in functor:
        functor = ['sentiment', 'arousal', 'entropy', 'thematic', 'epistemic']

    for f_name in functor:
        if f_name == 'sentiment':
            functors_to_use.append(('sentiment', SentimentFunctor(method='vader')))
        elif f_name == 'arousal':
            if language == 'ja':
                functors_to_use.append(('arousal', JapaneseArousalFunctor()))
            else:
                functors_to_use.append(('arousal', ArousalFunctor()))
        elif f_name == 'entropy':
            if language == 'ja':
                functors_to_use.append(('entropy', JapaneseEntropyFunctor()))
            else:
                functors_to_use.append(('entropy', EntropyFunctor()))
        elif f_name == 'thematic':
            functors_to_use.append(('thematic', ThematicFunctor()))
        elif f_name == 'epistemic':
            functors_to_use.append(('epistemic', EpistemicFunctor()))

    # Extract trajectories
    results = {"file": text_file, "word_count": word_count, "trajectories": {}}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting trajectories...", total=len(functors_to_use))

        for name, func in functors_to_use:
            trajectory = func.process_text(text, window_size, overlap)
            results["trajectories"][name] = {
                "values": trajectory.values.tolist(),
                "mean": float(np.mean(trajectory.values)),
                "std": float(np.std(trajectory.values)),
                "min": float(np.min(trajectory.values)),
                "max": float(np.max(trajectory.values)),
            }
            progress.advance(task)

    # Detect structures
    console.print("\n[bold]Detecting narrative structures...[/bold]")

    if 'sentiment' in results["trajectories"]:
        sentiment_values = np.array(results["trajectories"]["sentiment"]["values"])

        # Harmon Circle
        harmon = HarmonCircleDetector()
        harmon_match = harmon.detect(sentiment_values, Path(text_file).stem, Path(text_file).stem)
        results["harmon_circle"] = {
            "conformance": harmon_match.conformance_score,
            "pattern_type": harmon_match.pattern_type,
            "notes": harmon_match.notes,
        }

        # Kishōtenketsu
        kisho = KishotenketsuDetector()
        kisho_match = kisho.detect(sentiment_values, Path(text_file).stem, Path(text_file).stem)
        results["kishotenketsu"] = {
            "conformance": kisho_match.conformance_score,
            "pattern_type": kisho_match.pattern_type,
            "has_twist": kisho_match.has_twist,
            "notes": kisho_match.notes,
        }

    # Display results
    console.print("\n[bold green]Results:[/bold green]")

    table = Table(title="Functor Trajectories")
    table.add_column("Functor", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for name, data in results["trajectories"].items():
        table.add_row(
            name,
            f"{data['mean']:.3f}",
            f"{data['std']:.3f}",
            f"{data['min']:.3f}",
            f"{data['max']:.3f}",
        )

    console.print(table)

    if "harmon_circle" in results:
        console.print(f"\n[bold]Harmon Circle:[/bold] {results['harmon_circle']['conformance']:.2f} "
                     f"({results['harmon_circle']['pattern_type']})")

    if "kishotenketsu" in results:
        console.print(f"[bold]Kishōtenketsu:[/bold] {results['kishotenketsu']['conformance']:.2f} "
                     f"({results['kishotenketsu']['pattern_type']})")

    # Save output
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]Results saved to {output}[/green]")

    # Generate plot
    if plot:
        import matplotlib.pyplot as plt

        n_functors = len(results["trajectories"])
        fig, axes = plt.subplots(n_functors, 1, figsize=(12, 3 * n_functors), sharex=True)

        if n_functors == 1:
            axes = [axes]

        for ax, (name, data) in zip(axes, results["trajectories"].items()):
            values = data["values"]
            x = np.linspace(0, 1, len(values))
            ax.plot(x, values, linewidth=1.5)
            ax.fill_between(x, values, alpha=0.3)
            ax.set_ylabel(name.capitalize())
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Narrative Time")
        fig.suptitle(Path(text_file).stem, fontsize=14)
        plt.tight_layout()

        plot_file = output.replace('.json', '.png') if output else f"{Path(text_file).stem}_trajectory.png"
        plt.savefig(plot_file, dpi=150)
        console.print(f"[green]Plot saved to {plot_file}[/green]")
        plt.close()


@cli.command()
@click.argument('corpus_dir', type=click.Path(exists=True))
@click.option('--output', '-o', required=True, type=click.Path(), help='Output directory')
@click.option('--functor', '-f', multiple=True,
              type=click.Choice(['sentiment', 'arousal', 'entropy', 'all']),
              default=['all'], help='Functors to apply')
@click.option('--language', '-l', default='en', type=click.Choice(['en', 'ja']))
@click.option('--window-size', '-w', default=1000)
def batch(corpus_dir: str, output: str, functor: tuple, language: str, window_size: int):
    """
    Batch process a corpus directory.

    \b
    Examples:
        fna batch data/gutenberg/ -o results/gutenberg/
        fna batch data/aozora/ -o results/aozora/ -l ja
    """
    from .functors.sentiment import process_corpus as sentiment_corpus
    from .functors.arousal import process_corpus as arousal_corpus
    from .functors.entropy import process_corpus as entropy_corpus

    corpus_path = Path(corpus_dir)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    if 'all' in functor:
        functor = ['sentiment', 'arousal', 'entropy']

    for f_name in functor:
        console.print(f"\n[bold blue]Processing {f_name}...[/bold blue]")

        out_dir = output_path / f_name
        out_dir.mkdir(exist_ok=True)

        if f_name == 'sentiment':
            sentiment_corpus(corpus_path, out_dir, window_size=window_size)
        elif f_name == 'arousal':
            arousal_corpus(corpus_path, out_dir, language=language, window_size=window_size)
        elif f_name == 'entropy':
            entropy_corpus(corpus_path, out_dir, language=language, window_size=window_size)

    console.print(f"\n[bold green]Batch processing complete![/bold green]")
    console.print(f"Results saved to: {output_path}")


@cli.command()
@click.argument('corpus1', type=click.Path(exists=True))
@click.argument('corpus2', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for comparison')
def compare(corpus1: str, corpus2: str, output: Optional[str]):
    """
    Compare two corpora statistically.

    \b
    Examples:
        fna compare results/english/ results/japanese/
    """
    import numpy as np
    from scipy import stats

    console.print(f"[bold blue]Comparing corpora:[/bold blue]")
    console.print(f"  Corpus 1: {corpus1}")
    console.print(f"  Corpus 2: {corpus2}")

    results = {"corpus1": corpus1, "corpus2": corpus2, "comparisons": {}}

    # Find common functors
    c1_path = Path(corpus1)
    c2_path = Path(corpus2)

    for functor_dir in c1_path.iterdir():
        if functor_dir.is_dir() and (c2_path / functor_dir.name).exists():
            functor_name = functor_dir.name

            # Load manifests
            m1_file = functor_dir / "manifest.json"
            m2_file = c2_path / functor_name / "manifest.json"

            if not m1_file.exists() or not m2_file.exists():
                continue

            with open(m1_file) as f:
                m1 = json.load(f)
            with open(m2_file) as f:
                m2 = json.load(f)

            # Extract means
            key = f"mean_{functor_name}"
            vals1 = [t.get(key, t.get("mean_sentiment", 0)) for t in m1.get("trajectories", [])]
            vals2 = [t.get(key, t.get("mean_sentiment", 0)) for t in m2.get("trajectories", [])]

            if not vals1 or not vals2:
                continue

            # Statistical test
            t_stat, p_value = stats.ttest_ind(vals1, vals2)

            results["comparisons"][functor_name] = {
                "corpus1_mean": float(np.mean(vals1)),
                "corpus1_std": float(np.std(vals1)),
                "corpus1_n": len(vals1),
                "corpus2_mean": float(np.mean(vals2)),
                "corpus2_std": float(np.std(vals2)),
                "corpus2_n": len(vals2),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
            }

    # Display results
    table = Table(title="Cross-Corpus Comparison")
    table.add_column("Functor", style="cyan")
    table.add_column("Corpus 1", justify="right")
    table.add_column("Corpus 2", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Sig.", justify="center")

    for name, data in results["comparisons"].items():
        sig = "[green]***[/green]" if data["p_value"] < 0.001 else \
              "[green]**[/green]" if data["p_value"] < 0.01 else \
              "[green]*[/green]" if data["p_value"] < 0.05 else ""

        table.add_row(
            name,
            f"{data['corpus1_mean']:.3f} (±{data['corpus1_std']:.3f})",
            f"{data['corpus2_mean']:.3f} (±{data['corpus2_std']:.3f})",
            f"{data['p_value']:.4f}",
            sig,
        )

    console.print(table)
    console.print("\n[dim]Significance: * p<0.05, ** p<0.01, *** p<0.001[/dim]")

    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]Results saved to {output}[/green]")


@cli.command()
@click.option('--corpus', '-c', required=True,
              type=click.Choice(['gutenberg', 'aozora', 'all']),
              help='Corpus to process')
@click.option('--output', '-o', required=True, type=click.Path(), help='Output directory')
@click.option('--sample-size', '-n', default=50, help='Number of texts to sample')
def pipeline(corpus: str, output: str, sample_size: int):
    """
    Run the full analysis pipeline.

    \b
    Examples:
        fna pipeline --corpus gutenberg --output results/
        fna pipeline --corpus all --output results/ -n 100
    """
    import subprocess

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]Running full analysis pipeline[/bold blue]")

    steps = []

    if corpus in ('gutenberg', 'all'):
        steps.append(('Collecting Gutenberg corpus', f'make corpus-gutenberg SAMPLE_SIZE={sample_size}'))

    if corpus in ('aozora', 'all'):
        steps.append(('Collecting Aozora corpus', f'make corpus-aozora SAMPLE_SIZE={sample_size}'))

    steps.extend([
        ('Preprocessing', 'make preprocess'),
        ('Extracting trajectories', 'make extract-trajectories'),
        ('Detecting structures', 'make detect-all'),
        ('Clustering', 'make cluster'),
        ('Analyzing', 'make analyze-all'),
        ('Generating visualizations', 'make visualize'),
        ('Creating report', 'make report'),
    ])

    for desc, cmd in steps:
        console.print(f"\n[bold]{desc}...[/bold]")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"[red]Error: {result.stderr}[/red]")
        else:
            console.print(f"[green]Done[/green]")

    console.print("\n[bold green]Pipeline complete![/bold green]")


@cli.command()
def info():
    """
    Display information about available functors and detectors.
    """
    console.print("[bold blue]Functorial Narrative Analysis[/bold blue]")
    console.print("A category-theoretic framework for cross-cultural narrative structure analysis.\n")

    console.print("[bold]Available Functors:[/bold]")
    functors = [
        ("F_sentiment", "Emotional valence (happiness-sadness axis)", "[-1, 1]"),
        ("F_arousal", "Tension/excitement (calm-excited axis)", "[0, 1]"),
        ("F_entropy", "Complexity/predictability", "[0, 1]"),
        ("F_thematic", "Semantic drift (thematic coherence)", "[0, 1]"),
        ("F_epistemic", "Certainty/uncertainty markers", "[0, 1]"),
    ]

    table = Table()
    table.add_column("Functor", style="cyan")
    table.add_column("Description")
    table.add_column("Range", justify="right")

    for name, desc, range_ in functors:
        table.add_row(name, desc, range_)

    console.print(table)

    console.print("\n[bold]Structural Detectors:[/bold]")
    detectors = [
        ("Harmon Circle", "8-stage Western narrative structure"),
        ("Kishōtenketsu", "4-act East Asian narrative structure"),
    ]

    for name, desc in detectors:
        console.print(f"  [cyan]{name}[/cyan]: {desc}")

    console.print("\n[bold]Supported Languages:[/bold]")
    console.print("  - English (en)")
    console.print("  - Japanese (ja)")

    console.print("\n[dim]Run 'fna --help' for usage information.[/dim]")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
