#!/usr/bin/env python3
"""
Writer CLI - A friendly command-line tool for authors to analyze their manuscripts.

This tool provides narrative structure analysis using the ICC (Information Complexity Classes)
model to help writers understand and improve their storytelling.

Usage:
    # Analyze your manuscript
    python scripts/writer_cli.py analyze manuscript.txt

    # Quick overview
    python scripts/writer_cli.py analyze manuscript.txt --quick

    # Generate terrain visualization
    python scripts/writer_cli.py terrain manuscript.txt -o my_terrain.png

    # Compare two manuscripts
    python scripts/writer_cli.py compare draft_v1.txt draft_v2.txt

    # Get writing suggestions based on ICC class
    python scripts/writer_cli.py suggest manuscript.txt --target ICC-4
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

console = Console()


# =============================================================================
# ICC Class Descriptions for Writers
# =============================================================================

ICC_WRITER_GUIDE = {
    "ICC-0": {
        "name": "Wandering Mist",
        "tagline": "The Uncharted Path",
        "description": "Your narrative defies conventional patterns - it may be experimental, "
                      "multi-threaded, or genuinely innovative.",
        "strengths": [
            "Unique and unpredictable structure",
            "Freedom from formulaic storytelling",
            "Potential for literary innovation",
        ],
        "considerations": [
            "Readers may find it harder to follow",
            "Consider adding subtle guiding threads",
            "Ensure the complexity serves the story",
        ],
        "examples": ["Ulysses", "Cloud Atlas", "2666", "Anna Karenina"],
    },
    "ICC-1": {
        "name": "Quiet Ascent",
        "tagline": "The Contemplative Rise",
        "description": "Your story builds gradually toward growth or revelation, "
                      "with a calm, steady progression.",
        "strengths": [
            "Meditative, immersive quality",
            "Room for character depth",
            "Satisfying sense of progress",
        ],
        "considerations": [
            "May feel slow to action-oriented readers",
            "Ensure the quiet moments are meaningful",
            "The ending should feel earned",
        ],
        "examples": ["Kokoro", "Snow Country", "Literary fiction"],
    },
    "ICC-2": {
        "name": "Gentle Descent",
        "tagline": "The Elegiac Fall",
        "description": "Your narrative moves gracefully toward loss or ending, "
                      "creating a melancholic but beautiful arc.",
        "strengths": [
            "Emotional depth and resonance",
            "Beautiful, controlled tragedy",
            "Lasting emotional impact",
        ],
        "considerations": [
            "Prepare readers for the emotional tone",
            "Balance sadness with moments of beauty",
            "The descent should feel inevitable",
        ],
        "examples": ["The Tale of Genji", "The Makioka Sisters"],
    },
    "ICC-3": {
        "name": "Eternal Return",
        "tagline": "The Cyclical Journey",
        "description": "Your story oscillates through multiple peaks while returning to "
                      "equilibrium - the classic adventure pattern.",
        "strengths": [
            "Engaging, episodic structure",
            "Multiple climactic moments",
            "Satisfying sense of completion",
        ],
        "considerations": [
            "Ensure each cycle adds something new",
            "Avoid repetitive feeling",
            "The return should show growth",
        ],
        "examples": ["The Odyssey", "Don Quixote", "Adventure serials"],
    },
    "ICC-4": {
        "name": "Triumphant Climb",
        "tagline": "The Dramatic Ascent",
        "description": "Your story rises through setbacks toward ultimate triumph - "
                      "the classic rags-to-riches pattern.",
        "strengths": [
            "Highly engaging and commercial",
            "Strong emotional payoff",
            "Clear reader satisfaction",
        ],
        "considerations": [
            "Avoid predictability",
            "Make setbacks feel genuine",
            "Ensure triumph is earned",
        ],
        "examples": ["Rocky", "Pride and Prejudice", "The Count of Monte Cristo"],
    },
    "ICC-5": {
        "name": "Tragic Fall",
        "tagline": "The Dramatic Descent",
        "description": "Your narrative descends through reversals toward doom - "
                      "the operatic tragedy pattern.",
        "strengths": [
            "Powerful emotional impact",
            "Rich dramatic tension",
            "Memorable, haunting quality",
        ],
        "considerations": [
            "Balance darkness with moments of hope",
            "Make the fall feel meaningful",
            "Consider what readers take away",
        ],
        "examples": ["Macbeth", "Breaking Bad", "The Great Gatsby"],
    },
}


def load_text(file_path: str) -> tuple[str, str]:
    """Load text from a file."""
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        sys.exit(1)

    text = path.read_text(encoding='utf-8')
    title = path.stem.replace('_', ' ').replace('-', ' ').title()
    return text, title


def analyze_manuscript(text: str, title: str) -> Dict[str, Any]:
    """Analyze a manuscript and return results."""
    from src.functors.sentiment import SentimentFunctor
    from src.detectors.icc import ICCDetector

    # Extract sentiment trajectory
    functor = SentimentFunctor()
    trajectory = functor.process_text(text)
    normalized = trajectory.normalize()

    # Detect ICC class
    detector = ICCDetector()
    result = detector.detect(
        normalized.values,
        trajectory_id=title.lower().replace(' ', '_'),
        title=title
    )

    return {
        "title": title,
        "word_count": len(text.split()),
        "char_count": len(text),
        "trajectory": normalized,
        "icc_result": result,
    }


def print_analysis(analysis: Dict[str, Any], quick: bool = False):
    """Print analysis results in a writer-friendly format."""
    result = analysis["icc_result"]
    guide = ICC_WRITER_GUIDE.get(result.icc_class, ICC_WRITER_GUIDE["ICC-0"])

    # Header
    console.print()
    console.print(Panel(
        f"[bold cyan]{analysis['title']}[/bold cyan]\n"
        f"[dim]{analysis['word_count']:,} words[/dim]",
        title="Manuscript Analysis",
        border_style="cyan"
    ))

    # ICC Classification
    console.print()
    console.print(Panel(
        f"[bold yellow]{result.icc_class}[/bold yellow] - "
        f"[bold]{guide['name']}[/bold]\n"
        f"[italic]{guide['tagline']}[/italic]\n\n"
        f"{guide['description']}",
        title="Narrative Pattern",
        border_style="yellow"
    ))

    if quick:
        return

    # Key Metrics
    features = result.features
    metrics_table = Table(title="Key Metrics", show_header=True, header_style="bold magenta")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", justify="right")
    metrics_table.add_column("Interpretation", style="dim")

    # Net change
    net_change = features['net_change']
    if net_change > 0.15:
        nc_interp = "Rising arc"
    elif net_change < -0.15:
        nc_interp = "Falling arc"
    else:
        nc_interp = "Balanced/cyclical"
    metrics_table.add_row("Emotional Arc", f"{net_change:+.2f}", nc_interp)

    # Peaks
    n_peaks = features['n_peaks']
    if n_peaks <= 3:
        peak_interp = "Minimal (contemplative)"
    elif n_peaks <= 6:
        peak_interp = "Moderate (balanced)"
    else:
        peak_interp = "Many (dramatic)"
    metrics_table.add_row("Dramatic Peaks", str(n_peaks), peak_interp)

    # Volatility
    volatility = features['volatility']
    if volatility < 0.05:
        vol_interp = "Smooth (literary)"
    elif volatility < 0.1:
        vol_interp = "Moderate (balanced)"
    else:
        vol_interp = "Turbulent (thriller)"
    metrics_table.add_row("Emotional Volatility", f"{volatility:.3f}", vol_interp)

    console.print()
    console.print(metrics_table)

    # Strengths
    console.print()
    console.print("[bold green]Strengths of Your Narrative Pattern:[/bold green]")
    for strength in guide["strengths"]:
        console.print(f"  [green]+[/green] {strength}")

    # Considerations
    console.print()
    console.print("[bold yellow]Things to Consider:[/bold yellow]")
    for consideration in guide["considerations"]:
        console.print(f"  [yellow]![/yellow] {consideration}")

    # Similar works
    console.print()
    console.print(f"[bold blue]Similar Works:[/bold blue] {', '.join(guide['examples'])}")

    # Confidence
    console.print()
    console.print(f"[dim]Classification confidence: {result.confidence:.0%}[/dim]")


def suggest_improvements(analysis: Dict[str, Any], target_class: Optional[str] = None):
    """Provide writing suggestions based on analysis."""
    result = analysis["icc_result"]
    features = result.features
    current_class = result.icc_class

    console.print()
    console.print(Panel(
        f"Current pattern: [bold]{current_class}[/bold] ({ICC_WRITER_GUIDE[current_class]['name']})",
        title="Writing Suggestions",
        border_style="green"
    ))

    suggestions = []

    # General suggestions based on features
    if features['volatility'] < 0.03:
        suggestions.append({
            "area": "Emotional Range",
            "observation": "Your narrative is very smooth, almost flat emotionally.",
            "suggestion": "Consider adding more emotional variation - moments of tension, "
                         "surprise, or heightened emotion can engage readers more deeply."
        })

    if features['n_peaks'] < 2:
        suggestions.append({
            "area": "Dramatic Structure",
            "observation": "Your story has very few dramatic peaks.",
            "suggestion": "Most readers expect some dramatic moments. Consider adding "
                         "turning points, revelations, or conflicts that create peaks."
        })

    if features['n_peaks'] > 10:
        suggestions.append({
            "area": "Pacing",
            "observation": "Your narrative has many dramatic peaks.",
            "suggestion": "Too many peaks can exhaust readers. Consider which peaks are "
                         "essential and whether some could be consolidated or softened."
        })

    if abs(features['net_change']) < 0.05:
        suggestions.append({
            "area": "Overall Arc",
            "observation": "Your story ends roughly where it began emotionally.",
            "suggestion": "This can work well for cyclical stories, but ensure your characters "
                         "have transformed even if circumstances return to baseline."
        })

    # Target-specific suggestions
    if target_class and target_class != current_class:
        target_guide = ICC_WRITER_GUIDE.get(target_class, {})
        if target_guide:
            console.print()
            console.print(f"[bold]To move toward {target_class} ({target_guide['name']}):[/bold]")

            if target_class == "ICC-4" and features['net_change'] < 0.15:
                suggestions.append({
                    "area": "Rising Arc",
                    "observation": f"ICC-4 (Triumphant Climb) requires a rising arc.",
                    "suggestion": "Build toward a more positive ending. Add moments of "
                                 "triumph and let your protagonist overcome obstacles."
                })

            if target_class == "ICC-5" and features['net_change'] > -0.15:
                suggestions.append({
                    "area": "Falling Arc",
                    "observation": f"ICC-5 (Tragic Fall) requires a descending arc.",
                    "suggestion": "Build toward a darker ending. Let consequences accumulate "
                                 "and create a sense of inevitable tragedy."
                })

            if target_class in ["ICC-1", "ICC-2"] and features['volatility'] > 0.07:
                suggestions.append({
                    "area": "Smoothness",
                    "observation": f"{target_class} patterns are characterized by smooth transitions.",
                    "suggestion": "Soften your dramatic peaks. Let changes happen gradually "
                                 "rather than in sudden bursts."
                })

            if target_class in ["ICC-3", "ICC-4", "ICC-5"] and features['n_peaks'] < 4:
                suggestions.append({
                    "area": "Dramatic Complexity",
                    "observation": f"{target_class} patterns typically have multiple dramatic peaks.",
                    "suggestion": "Add more turning points and dramatic moments. Western-style "
                                 "narratives expect more peaks and valleys."
                })

    # Print suggestions
    if suggestions:
        for i, sugg in enumerate(suggestions, 1):
            console.print()
            console.print(f"[bold cyan]{i}. {sugg['area']}[/bold cyan]")
            console.print(f"   [dim]Observation:[/dim] {sugg['observation']}")
            console.print(f"   [green]Suggestion:[/green] {sugg['suggestion']}")
    else:
        console.print()
        console.print("[green]Your narrative structure looks well-balanced! No specific suggestions.[/green]")


def compare_manuscripts(files: List[str]):
    """Compare multiple manuscripts side by side."""
    analyses = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing manuscripts...", total=len(files))

        for file_path in files:
            text, title = load_text(file_path)
            analysis = analyze_manuscript(text, title)
            analyses.append(analysis)
            progress.advance(task)

    # Comparison table
    console.print()
    table = Table(title="Manuscript Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")

    for analysis in analyses:
        table.add_column(analysis["title"][:20], justify="center")

    # Add rows
    table.add_row(
        "Word Count",
        *[f"{a['word_count']:,}" for a in analyses]
    )

    table.add_row(
        "ICC Class",
        *[f"{a['icc_result'].icc_class}" for a in analyses]
    )

    table.add_row(
        "Pattern Name",
        *[ICC_WRITER_GUIDE[a['icc_result'].icc_class]['name'] for a in analyses]
    )

    table.add_row(
        "Cultural Style",
        *[a['icc_result'].cultural_prediction.title() for a in analyses]
    )

    table.add_row(
        "Emotional Arc",
        *[f"{a['icc_result'].features['net_change']:+.2f}" for a in analyses]
    )

    table.add_row(
        "Dramatic Peaks",
        *[str(a['icc_result'].features['n_peaks']) for a in analyses]
    )

    table.add_row(
        "Volatility",
        *[f"{a['icc_result'].features['volatility']:.3f}" for a in analyses]
    )

    table.add_row(
        "Confidence",
        *[f"{a['icc_result'].confidence:.0%}" for a in analyses]
    )

    console.print(table)

    # Brief interpretation
    console.print()
    console.print("[bold]Interpretation:[/bold]")
    for analysis in analyses:
        result = analysis["icc_result"]
        guide = ICC_WRITER_GUIDE[result.icc_class]
        console.print(f"  [cyan]{analysis['title']}[/cyan]: {guide['tagline']}")


def generate_terrain(file_path: str, output: Optional[str] = None):
    """Generate terrain visualization for a manuscript."""
    text, title = load_text(file_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating terrain...", total=3)

        # Analyze
        progress.update(task, description="Analyzing manuscript...")
        analysis = analyze_manuscript(text, title)
        progress.advance(task)

        # Generate visualization
        progress.update(task, description="Creating terrain visualization...")

        from scripts.visualize_narrative import plot_narrative_terrain

        trajectory = analysis["trajectory"]
        icc_result = analysis["icc_result"]

        output_dir = Path(output).parent if output else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)

        terrain_file = plot_narrative_terrain(
            trajectory.values,
            title,
            output_dir,
            icc_result
        )
        progress.advance(task)

        progress.update(task, description="Done!")
        progress.advance(task)

    console.print()
    console.print(f"[green]Terrain saved to: {terrain_file}[/green]")

    return terrain_file


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Writer CLI - Analyze your manuscript's narrative structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  writer_cli.py analyze manuscript.txt           Analyze your manuscript
  writer_cli.py analyze manuscript.txt --quick   Quick overview only
  writer_cli.py suggest manuscript.txt           Get writing suggestions
  writer_cli.py suggest manuscript.txt --target ICC-4   Suggestions to achieve ICC-4
  writer_cli.py compare draft_v1.txt draft_v2.txt   Compare versions
  writer_cli.py terrain manuscript.txt           Generate terrain visualization
  writer_cli.py classes                          List all ICC classes
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a manuscript")
    analyze_parser.add_argument("file", help="Path to manuscript file")
    analyze_parser.add_argument("--quick", "-q", action="store_true",
                                help="Quick overview only")
    analyze_parser.add_argument("--output", "-o", help="Save results to JSON")

    # Suggest command
    suggest_parser = subparsers.add_parser("suggest", help="Get writing suggestions")
    suggest_parser.add_argument("file", help="Path to manuscript file")
    suggest_parser.add_argument("--target", "-t",
                                choices=["ICC-0", "ICC-1", "ICC-2", "ICC-3", "ICC-4", "ICC-5"],
                                help="Target ICC class to move toward")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare manuscripts")
    compare_parser.add_argument("files", nargs="+", help="Manuscript files to compare")

    # Terrain command
    terrain_parser = subparsers.add_parser("terrain", help="Generate terrain visualization")
    terrain_parser.add_argument("file", help="Path to manuscript file")
    terrain_parser.add_argument("--output", "-o", help="Output file path")

    # Classes command
    classes_parser = subparsers.add_parser("classes", help="List all ICC classes")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    if args.command == "analyze":
        text, title = load_text(args.file)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing...", total=1)
            analysis = analyze_manuscript(text, title)
            progress.advance(task)

        print_analysis(analysis, quick=args.quick)

        if args.output:
            result_dict = {
                "title": analysis["title"],
                "word_count": analysis["word_count"],
                "icc_class": analysis["icc_result"].icc_class,
                "features": analysis["icc_result"].features,
            }
            Path(args.output).write_text(json.dumps(result_dict, indent=2))
            console.print(f"\n[dim]Results saved to {args.output}[/dim]")

    elif args.command == "suggest":
        text, title = load_text(args.file)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing...", total=1)
            analysis = analyze_manuscript(text, title)
            progress.advance(task)

        print_analysis(analysis, quick=True)
        suggest_improvements(analysis, target_class=args.target)

    elif args.command == "compare":
        compare_manuscripts(args.files)

    elif args.command == "terrain":
        generate_terrain(args.file, args.output)

    elif args.command == "classes":
        console.print()
        console.print(Panel(
            "[bold]Information Complexity Classes (ICC)[/bold]\n"
            "A data-driven narrative classification system",
            border_style="cyan"
        ))

        for class_id in ["ICC-0", "ICC-1", "ICC-2", "ICC-3", "ICC-4", "ICC-5"]:
            guide = ICC_WRITER_GUIDE[class_id]
            console.print()
            console.print(f"[bold yellow]{class_id}[/bold yellow] - "
                         f"[bold]{guide['name']}[/bold] ({guide['tagline']})")
            console.print(f"  {guide['description']}")
            console.print(f"  [dim]Examples: {', '.join(guide['examples'])}[/dim]")


if __name__ == "__main__":
    main()
