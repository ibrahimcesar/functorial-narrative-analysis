#!/usr/bin/env python3
"""
Generate Interactive Web Visualizations

Creates a web interface with:
- Interactive 3D terrain visualizations
- Side-by-side comparison of multiple narratives
- Exportable HTML files

Usage:
    # Generate terrain for a single book
    python scripts/generate_web.py terrain --gutenberg 1399 -o anna_terrain.html

    # Generate comparison page for multiple books
    python scripts/generate_web.py compare --gutenberg 1399 2600 84 -o comparison.html

    # Generate from local files
    python scripts/generate_web.py terrain --file manuscript.txt -o my_terrain.html

    # Generate full gallery from corpus
    python scripts/generate_web.py gallery --corpus-dir /path/to/corpus -o gallery/
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import urllib.request
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.visualization.interactive import (
    NarrativeData,
    create_interactive_terrain,
    create_comparison_page,
    create_trajectory_comparison_2d
)

console = Console()


def download_gutenberg(book_id: int) -> Tuple[str, str]:
    """Download a book from Project Gutenberg."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"

    try:
        with urllib.request.urlopen(url) as response:
            text = response.read().decode('utf-8-sig')
    except Exception as e:
        raise RuntimeError(f"Failed to download book {book_id}: {e}")

    # Extract title
    title = f"Gutenberg #{book_id}"
    for line in text[:5000].split('\n'):
        if line.startswith('Title:'):
            title = line.replace('Title:', '').strip()
            break

    # Remove header/footer
    start_markers = ["*** START OF", "***START OF"]
    end_markers = ["*** END OF", "***END OF"]

    for marker in start_markers:
        if marker in text:
            text = text.split(marker, 1)[1]
            text = text.split('\n', 1)[1] if '\n' in text else text
            break

    for marker in end_markers:
        if marker in text:
            text = text.split(marker, 1)[0]
            break

    return text.strip(), title


def analyze_text(text: str, title: str) -> NarrativeData:
    """Analyze text and return NarrativeData."""
    from src.functors.sentiment import SentimentFunctor
    from src.detectors.icc import ICCDetector
    from scipy.signal import find_peaks

    # Extract sentiment
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

    # Find peaks and valleys
    peaks, _ = find_peaks(normalized.values, prominence=0.1)
    valleys, _ = find_peaks(-normalized.values, prominence=0.1)

    return NarrativeData(
        title=title,
        trajectory=normalized.values,
        icc_class=result.icc_class,
        class_name=result.class_name,
        cultural_prediction=result.cultural_prediction,
        features=result.features,
        peaks=peaks,
        valleys=valleys
    )


def generate_terrain(
    gutenberg_id: Optional[int] = None,
    file_path: Optional[str] = None,
    title: Optional[str] = None,
    output: str = "terrain.html"
):
    """Generate interactive terrain for a single narrative."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Load text
        task = progress.add_task("Loading text...", total=3)

        if gutenberg_id:
            text, auto_title = download_gutenberg(gutenberg_id)
            title = title or auto_title
        elif file_path:
            path = Path(file_path)
            if not path.exists():
                console.print(f"[red]File not found: {file_path}[/red]")
                sys.exit(1)
            text = path.read_text(encoding='utf-8')
            title = title or path.stem.replace('_', ' ').title()
        else:
            console.print("[red]Must specify --gutenberg or --file[/red]")
            sys.exit(1)

        progress.advance(task)

        # Analyze
        progress.update(task, description="Analyzing narrative...")
        narrative = analyze_text(text, title)
        progress.advance(task)

        # Generate
        progress.update(task, description="Generating terrain...")
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        create_interactive_terrain(narrative, output_path)
        progress.advance(task)

    console.print(f"[green]Terrain saved to: {output}[/green]")
    console.print(f"[dim]Open in browser to interact with 3D visualization[/dim]")


def generate_comparison(
    gutenberg_ids: List[int] = None,
    file_paths: List[str] = None,
    output: str = "comparison.html",
    include_2d: bool = True
):
    """Generate comparison page for multiple narratives."""
    narratives = []

    sources = []
    if gutenberg_ids:
        sources.extend([('gutenberg', gid) for gid in gutenberg_ids])
    if file_paths:
        sources.extend([('file', fp) for fp in file_paths])

    if not sources:
        console.print("[red]Must specify at least one --gutenberg or --file[/red]")
        sys.exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing narratives...", total=len(sources) + 1)

        for source_type, source in sources:
            if source_type == 'gutenberg':
                progress.update(task, description=f"Downloading Gutenberg #{source}...")
                text, title = download_gutenberg(source)
            else:
                progress.update(task, description=f"Loading {Path(source).name}...")
                path = Path(source)
                if not path.exists():
                    console.print(f"[yellow]Warning: File not found: {source}, skipping[/yellow]")
                    progress.advance(task)
                    continue
                text = path.read_text(encoding='utf-8')
                title = path.stem.replace('_', ' ').title()

            progress.update(task, description=f"Analyzing {title[:30]}...")
            narrative = analyze_text(text, title)
            narratives.append(narrative)
            progress.advance(task)

        if not narratives:
            console.print("[red]No valid narratives to compare[/red]")
            sys.exit(1)

        # Generate comparison
        progress.update(task, description="Generating comparison page...")
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        create_comparison_page(narratives, output_path)
        progress.advance(task)

        # Generate 2D comparison if requested
        if include_2d:
            output_2d = output_path.parent / f"{output_path.stem}_2d.html"
            create_trajectory_comparison_2d(narratives, output_2d)
            console.print(f"[green]2D comparison saved to: {output_2d}[/green]")

    console.print(f"[green]3D comparison saved to: {output}[/green]")
    console.print(f"[dim]Open in browser to interact with visualizations[/dim]")


def generate_gallery(
    corpus_dir: str,
    output_dir: str = "gallery",
    max_books: int = 20
):
    """Generate a gallery of terrains from corpus."""
    corpus_path = Path(corpus_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find books in corpus
    index_file = corpus_path / "index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text())
        books = index.get("books", [])[:max_books]
    else:
        # Try to find JSON files directly
        books = []
        for json_file in list(corpus_path.glob("**/*.json"))[:max_books]:
            try:
                data = json.loads(json_file.read_text())
                if "content" in data or "text" in data:
                    books.append({
                        "id": json_file.stem,
                        "title": data.get("title", json_file.stem),
                        "file": str(json_file)
                    })
            except Exception:
                continue

    if not books:
        console.print(f"[red]No books found in {corpus_dir}[/red]")
        sys.exit(1)

    console.print(f"[blue]Found {len(books)} books in corpus[/blue]")

    narratives = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing books...", total=len(books))

        for book in books:
            title = book.get("title", book.get("id", "Unknown"))
            progress.update(task, description=f"Analyzing: {title[:40]}...")

            try:
                # Load book content
                if "file" in book:
                    book_path = Path(book["file"])
                else:
                    book_path = corpus_path / "books" / f"{book['id']}.json"

                if not book_path.exists():
                    progress.advance(task)
                    continue

                data = json.loads(book_path.read_text(encoding='utf-8'))
                text = data.get("content") or data.get("text", "")

                if not text or len(text) < 1000:
                    progress.advance(task)
                    continue

                narrative = analyze_text(text, title)
                narratives.append(narrative)

                # Generate individual terrain
                safe_name = "".join(c if c.isalnum() else "_" for c in title)[:50]
                terrain_path = output_path / f"{safe_name}_terrain.html"
                create_interactive_terrain(narrative, terrain_path)

            except Exception as e:
                console.print(f"[yellow]Warning: Failed to process {title}: {e}[/yellow]")

            progress.advance(task)

    # Generate combined comparison
    if len(narratives) >= 2:
        console.print(f"\n[blue]Generating comparison page for {len(narratives)} books...[/blue]")
        comparison_path = output_path / "comparison.html"
        create_comparison_page(narratives, comparison_path, title="Corpus Gallery")

        # 2D overlay
        overlay_path = output_path / "trajectories_overlay.html"
        create_trajectory_comparison_2d(narratives, overlay_path)

    # Generate index page
    generate_index_page(narratives, output_path)

    console.print(f"\n[bold green]Gallery generated in: {output_path}[/bold green]")
    console.print(f"[green]  - {len(narratives)} individual terrains[/green]")
    console.print(f"[green]  - comparison.html (side-by-side)[/green]")
    console.print(f"[green]  - trajectories_overlay.html (2D overlay)[/green]")
    console.print(f"[green]  - index.html (gallery index)[/green]")


def generate_index_page(narratives: List[NarrativeData], output_dir: Path):
    """Generate an index HTML page for the gallery."""

    # Group by ICC class
    by_class = {}
    for n in narratives:
        if n.icc_class not in by_class:
            by_class[n.icc_class] = []
        by_class[n.icc_class].append(n)

    cards_html = []
    for n in sorted(narratives, key=lambda x: x.icc_class):
        safe_name = "".join(c if c.isalnum() else "_" for c in n.title)[:50]
        cards_html.append(f"""
        <div class="card" data-icc="{n.icc_class}">
            <div class="card-header">
                <span class="icc-badge icc-{n.icc_class.split('-')[1]}">{n.icc_class}</span>
                <span class="cultural cultural-{n.cultural_prediction}">{n.cultural_prediction.title()}</span>
            </div>
            <h3>{n.title}</h3>
            <p class="class-name">{n.class_name}</p>
            <div class="metrics">
                <span>Arc: {n.features.get('net_change', 0):+.2f}</span>
                <span>Peaks: {n.features.get('n_peaks', 0)}</span>
            </div>
            <a href="{safe_name}_terrain.html" class="btn">View Terrain</a>
        </div>
        """)

    # Summary stats
    class_counts = {cls: len(items) for cls, items in by_class.items()}
    cultural_counts = {}
    for n in narratives:
        cult = n.cultural_prediction
        cultural_counts[cult] = cultural_counts.get(cult, 0) + 1

    # Helper function to generate filter buttons (avoids f-string backslash issue)
    def generate_filter_buttons():
        buttons = []
        for i in range(6):
            buttons.append(f'<button class="filter-btn" onclick="filterCards(\'ICC-{i}\')">ICC-{i}</button>')
        return ' '.join(buttons)

    index_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Narrative Terrain Gallery</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #fff; margin-bottom: 10px; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}

        .stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .stat {{
            background: rgba(255,255,255,0.1);
            padding: 15px 25px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #fff; }}
        .stat-label {{ font-size: 12px; color: #888; }}

        .actions {{
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .action-btn {{
            background: rgba(255,255,255,0.1);
            color: #fff;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            transition: all 0.3s;
        }}
        .action-btn:hover {{ background: rgba(255,255,255,0.2); }}
        .action-btn.primary {{ background: #4a90d9; }}
        .action-btn.primary:hover {{ background: #5a9fe8; }}

        .filters {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .filter-btn {{
            padding: 8px 16px;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.2);
            background: transparent;
            color: #fff;
            cursor: pointer;
            transition: all 0.3s;
        }}
        .filter-btn:hover, .filter-btn.active {{
            background: rgba(255,255,255,0.2);
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }}
        .card h3 {{
            margin: 0 0 5px 0;
            color: #fff;
            font-size: 16px;
        }}
        .class-name {{ color: #888; font-size: 13px; margin: 0 0 15px 0; }}
        .metrics {{
            display: flex;
            gap: 15px;
            font-size: 12px;
            color: #666;
            margin-bottom: 15px;
        }}
        .btn {{
            display: inline-block;
            padding: 8px 16px;
            background: #4a90d9;
            color: #fff;
            text-decoration: none;
            border-radius: 6px;
            font-size: 13px;
            transition: background 0.3s;
        }}
        .btn:hover {{ background: #5a9fe8; }}

        .icc-badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
        }}
        .icc-0 {{ background: #6c757d; }}
        .icc-1 {{ background: #198754; }}
        .icc-2 {{ background: #0dcaf0; color: #000; }}
        .icc-3 {{ background: #ffc107; color: #000; }}
        .icc-4 {{ background: #fd7e14; }}
        .icc-5 {{ background: #dc3545; }}

        .cultural {{ font-size: 11px; padding: 4px 8px; border-radius: 4px; }}
        .cultural-western {{ color: #ffc107; }}
        .cultural-japanese {{ color: #0dcaf0; }}
        .cultural-neutral {{ color: #6c757d; }}

        .card.hidden {{ display: none; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Narrative Terrain Gallery</h1>
        <p class="subtitle">Interactive 3D visualizations of emotional landscapes</p>

        <div class="stats">
            <div class="stat">
                <div class="stat-value">{len(narratives)}</div>
                <div class="stat-label">Total Narratives</div>
            </div>
            {' '.join(f'<div class="stat"><div class="stat-value">{count}</div><div class="stat-label">{cls}</div></div>' for cls, count in sorted(class_counts.items()))}
        </div>

        <div class="actions">
            <a href="comparison.html" class="action-btn primary">View All Side-by-Side</a>
            <a href="trajectories_overlay.html" class="action-btn">2D Trajectory Overlay</a>
        </div>

        <div class="filters">
            <button class="filter-btn active" onclick="filterCards('all')">All</button>
            {generate_filter_buttons()}
        </div>

        <div class="grid">
            {''.join(cards_html)}
        </div>

        <p style="text-align:center; color:#555; margin-top:40px; font-size:12px;">
            Generated by Functorial Narrative Analysis | ICC Model v2
        </p>
    </div>

    <script>
        function filterCards(icc) {{
            document.querySelectorAll('.filter-btn').forEach(btn => {{
                btn.classList.remove('active');
                if (btn.textContent === 'All' && icc === 'all') btn.classList.add('active');
                if (btn.textContent === icc) btn.classList.add('active');
            }});

            document.querySelectorAll('.card').forEach(card => {{
                if (icc === 'all' || card.dataset.icc === icc) {{
                    card.classList.remove('hidden');
                }} else {{
                    card.classList.add('hidden');
                }}
            }});
        }}
    </script>
</body>
</html>
"""

    index_path = output_dir / "index.html"
    index_path.write_text(index_html, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive web visualizations for narrative analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Terrain command
    terrain_parser = subparsers.add_parser("terrain", help="Generate single terrain")
    terrain_parser.add_argument("--gutenberg", "-g", type=int, help="Gutenberg book ID")
    terrain_parser.add_argument("--file", "-f", type=str, help="Local file path")
    terrain_parser.add_argument("--title", "-t", type=str, help="Override title")
    terrain_parser.add_argument("--output", "-o", default="terrain.html", help="Output file")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple narratives")
    compare_parser.add_argument("--gutenberg", "-g", type=int, nargs="+", help="Gutenberg IDs")
    compare_parser.add_argument("--file", "-f", type=str, nargs="+", help="Local files")
    compare_parser.add_argument("--output", "-o", default="comparison.html", help="Output file")
    compare_parser.add_argument("--no-2d", action="store_true", help="Skip 2D comparison")

    # Gallery command
    gallery_parser = subparsers.add_parser("gallery", help="Generate gallery from corpus")
    gallery_parser.add_argument("--corpus-dir", "-c", required=True, help="Corpus directory")
    gallery_parser.add_argument("--output", "-o", default="gallery", help="Output directory")
    gallery_parser.add_argument("--max", type=int, default=20, help="Max books to process")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "terrain":
        generate_terrain(
            gutenberg_id=args.gutenberg,
            file_path=args.file,
            title=args.title,
            output=args.output
        )

    elif args.command == "compare":
        generate_comparison(
            gutenberg_ids=args.gutenberg,
            file_paths=args.file,
            output=args.output,
            include_2d=not args.no_2d
        )

    elif args.command == "gallery":
        generate_gallery(
            corpus_dir=args.corpus_dir,
            output_dir=args.output,
            max_books=args.max
        )


if __name__ == "__main__":
    main()
