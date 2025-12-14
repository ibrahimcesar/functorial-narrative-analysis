#!/usr/bin/env python3
"""
Author and Genre Analysis

Addresses Limitation 7.3 from the falsifiability article:
- Can ICC profiles distinguish individual authors?
- Do certain genres cluster in specific ICC classes?
- Track ICC distributions across literary periods

This script provides tools to analyze narrative patterns by:
1. Author fingerprinting (do authors have consistent ICC signatures?)
2. Genre clustering (do genres map to specific ICC classes?)
3. Temporal analysis (how have ICC distributions changed over time?)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np
import click
from rich.console import Console
from rich.table import Table
from scipy import stats

console = Console()


@dataclass
class AuthorProfile:
    """ICC profile for an author."""
    author: str
    n_works: int
    icc_distribution: Dict[str, int]  # ICC class -> count
    dominant_class: str
    class_consistency: float  # How often they use dominant class
    mean_volatility: float
    mean_peaks: float
    cultural_leaning: str  # Western-typical, Japanese-typical, Neutral


@dataclass
class GenreProfile:
    """ICC profile for a genre."""
    genre: str
    n_works: int
    icc_distribution: Dict[str, int]
    dominant_class: str
    class_entropy: float  # Higher = more diverse ICC usage
    typical_features: Dict[str, float]


def load_analysis_results(results_dir: Path) -> List[Dict]:
    """Load analysis results from a directory."""
    results = []

    # Try manifest.json first
    manifest_path = results_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
            if "trajectories" in manifest:
                return manifest["trajectories"]

    # Otherwise load individual files
    for f in results_dir.glob("*.json"):
        if f.name in ["manifest.json", "metadata.json"]:
            continue
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                results.append(data)
        except:
            continue

    return results


def analyze_by_author(results: List[Dict]) -> Dict[str, AuthorProfile]:
    """
    Analyze ICC patterns by author.

    Returns dict mapping author name to their ICC profile.
    """
    author_works = defaultdict(list)

    for r in results:
        author = r.get("author", "Unknown")
        if author and author != "Unknown":
            author_works[author].append(r)

    profiles = {}

    for author, works in author_works.items():
        if len(works) < 2:
            continue  # Need at least 2 works for meaningful profile

        # Count ICC classes
        icc_counts = defaultdict(int)
        volatilities = []
        peaks = []

        for w in works:
            icc = w.get("icc_class", w.get("class", "ICC-0"))
            icc_counts[icc] += 1

            if "volatility" in w:
                volatilities.append(w["volatility"])
            if "n_peaks" in w:
                peaks.append(w["n_peaks"])

        # Find dominant class
        dominant = max(icc_counts.keys(), key=lambda k: icc_counts[k])
        consistency = icc_counts[dominant] / len(works)

        # Determine cultural leaning
        western_classes = sum(icc_counts.get(c, 0) for c in ["ICC-3", "ICC-4", "ICC-5"])
        japanese_classes = sum(icc_counts.get(c, 0) for c in ["ICC-1", "ICC-2"])

        if western_classes > japanese_classes * 1.5:
            cultural = "Western-typical"
        elif japanese_classes > western_classes * 1.5:
            cultural = "Japanese-typical"
        else:
            cultural = "Neutral"

        profiles[author] = AuthorProfile(
            author=author,
            n_works=len(works),
            icc_distribution=dict(icc_counts),
            dominant_class=dominant,
            class_consistency=consistency,
            mean_volatility=np.mean(volatilities) if volatilities else 0.0,
            mean_peaks=np.mean(peaks) if peaks else 0.0,
            cultural_leaning=cultural
        )

    return profiles


def analyze_by_genre(results: List[Dict], genre_mappings: Optional[Dict] = None) -> Dict[str, GenreProfile]:
    """
    Analyze ICC patterns by genre.

    Args:
        results: Analysis results
        genre_mappings: Optional dict mapping work IDs to genres

    Returns dict mapping genre to ICC profile.
    """
    # Default genre inference from titles/metadata if no mappings provided
    if genre_mappings is None:
        genre_mappings = infer_genres(results)

    genre_works = defaultdict(list)

    for r in results:
        work_id = r.get("id", r.get("source_id", ""))
        genre = genre_mappings.get(work_id, "Unknown")
        if genre != "Unknown":
            genre_works[genre].append(r)

    profiles = {}

    for genre, works in genre_works.items():
        if len(works) < 3:
            continue

        icc_counts = defaultdict(int)
        features = defaultdict(list)

        for w in works:
            icc = w.get("icc_class", w.get("class", "ICC-0"))
            icc_counts[icc] += 1

            for key in ["volatility", "n_peaks", "net_change", "trend_r2"]:
                if key in w:
                    features[key].append(w[key])

        dominant = max(icc_counts.keys(), key=lambda k: icc_counts[k])

        # Compute entropy of class distribution
        total = sum(icc_counts.values())
        probs = [c / total for c in icc_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        max_entropy = np.log2(6)  # 6 ICC classes
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        typical = {k: float(np.mean(v)) for k, v in features.items()}

        profiles[genre] = GenreProfile(
            genre=genre,
            n_works=len(works),
            icc_distribution=dict(icc_counts),
            dominant_class=dominant,
            class_entropy=normalized_entropy,
            typical_features=typical
        )

    return profiles


def infer_genres(results: List[Dict]) -> Dict[str, str]:
    """
    Infer genres from titles and metadata.

    Simple heuristic-based inference.
    """
    genre_keywords = {
        "Mystery": ["mystery", "detective", "murder", "crime", "sherlock", "poirot",
                    "affair", "secret", "case"],
        "Romance": ["love", "heart", "passion", "marriage", "pride", "prejudice",
                    "romance", "beloved"],
        "Adventure": ["adventure", "journey", "quest", "expedition", "voyage",
                      "treasure", "island", "around the world"],
        "Horror": ["horror", "terror", "ghost", "vampire", "haunted", "dark",
                   "gothic", "frankenstein", "dracula"],
        "Literary Fiction": ["life", "death", "soul", "portrait", "brothers",
                             "war and peace", "anna karenina", "idiot"],
        "Children's": ["alice", "wonderland", "peter pan", "wizard of oz",
                       "anne of green", "little women", "secret garden"],
        "Science Fiction": ["robot", "future", "space", "machine", "r.u.r."],
        "Satire": ["gulliver", "candide", "satire", "pantagruel", "quixote"],
    }

    mappings = {}

    for r in results:
        work_id = r.get("id", r.get("source_id", ""))
        title = r.get("title", "").lower()
        author = r.get("author", "").lower()

        matched_genre = "General Fiction"

        for genre, keywords in genre_keywords.items():
            if any(kw in title or kw in author for kw in keywords):
                matched_genre = genre
                break

        mappings[work_id] = matched_genre

    return mappings


def author_distinguishability_test(profiles: Dict[str, AuthorProfile]) -> Dict:
    """
    Test whether ICC profiles can distinguish authors.

    Uses chi-squared test on ICC distributions.
    """
    # Build contingency table: authors x ICC classes
    authors = list(profiles.keys())
    classes = ["ICC-0", "ICC-1", "ICC-2", "ICC-3", "ICC-4", "ICC-5"]

    if len(authors) < 2:
        return {"distinguishable": False, "reason": "Need at least 2 authors"}

    # Build observed frequencies
    observed = []
    for author in authors:
        row = [profiles[author].icc_distribution.get(c, 0) for c in classes]
        observed.append(row)

    observed = np.array(observed)

    # Remove columns with all zeros
    col_sums = observed.sum(axis=0)
    non_zero_cols = col_sums > 0
    observed = observed[:, non_zero_cols]

    if observed.shape[1] < 2:
        return {"distinguishable": False, "reason": "Insufficient class variation"}

    # Chi-squared test
    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        distinguishable = p_value < 0.05
    except:
        return {"distinguishable": False, "reason": "Statistical test failed"}

    return {
        "distinguishable": distinguishable,
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "interpretation": (
            "Authors show statistically significant ICC differences"
            if distinguishable else
            "No significant ICC differences between authors"
        )
    }


def display_author_profiles(profiles: Dict[str, AuthorProfile]):
    """Display author profiles in a rich table."""
    table = Table(title="Author ICC Profiles")

    table.add_column("Author", style="cyan", max_width=25)
    table.add_column("Works", justify="right")
    table.add_column("Dominant ICC", style="magenta")
    table.add_column("Consistency", justify="right")
    table.add_column("Volatility", justify="right")
    table.add_column("Cultural Lean", style="yellow")

    sorted_profiles = sorted(profiles.values(), key=lambda p: p.n_works, reverse=True)

    for p in sorted_profiles[:15]:
        table.add_row(
            p.author[:23] + ".." if len(p.author) > 25 else p.author,
            str(p.n_works),
            p.dominant_class,
            f"{p.class_consistency:.0%}",
            f"{p.mean_volatility:.3f}" if p.mean_volatility else "-",
            p.cultural_leaning
        )

    console.print(table)


def display_genre_profiles(profiles: Dict[str, GenreProfile]):
    """Display genre profiles in a rich table."""
    table = Table(title="Genre ICC Profiles")

    table.add_column("Genre", style="cyan")
    table.add_column("Works", justify="right")
    table.add_column("Dominant ICC", style="magenta")
    table.add_column("Diversity", justify="right")
    table.add_column("Distribution")

    sorted_profiles = sorted(profiles.values(), key=lambda p: p.n_works, reverse=True)

    for p in sorted_profiles:
        # Format distribution
        dist_str = ", ".join(f"{k}:{v}" for k, v in
                            sorted(p.icc_distribution.items()) if v > 0)

        table.add_row(
            p.genre,
            str(p.n_works),
            p.dominant_class,
            f"{p.class_entropy:.2f}",
            dist_str[:40] + "..." if len(dist_str) > 40 else dist_str
        )

    console.print(table)


@click.command()
@click.option('--input', '-i', 'input_dir', required=True,
              type=click.Path(exists=True), help="Directory with analysis results")
@click.option('--output', '-o', 'output_file', default='output/author_genre_analysis.json',
              type=click.Path(), help="Output file")
@click.option('--analysis', '-a', default='both',
              type=click.Choice(['author', 'genre', 'both']),
              help="Type of analysis")
def main(input_dir: str, output_file: str, analysis: str):
    """
    Analyze ICC patterns by author and/or genre.

    This addresses Limitation 7.3: Can ICC profiles distinguish authors?
    Do genres cluster in specific ICC classes?
    """
    input_dir = Path(input_dir)
    results = load_analysis_results(input_dir)

    if not results:
        console.print("[red]No analysis results found![/red]")
        return

    console.print(f"[blue]Loaded {len(results)} analysis results[/blue]\n")

    output_data = {"input": str(input_dir), "n_works": len(results)}

    if analysis in ['author', 'both']:
        console.print("[bold]Author Analysis[/bold]")
        author_profiles = analyze_by_author(results)

        if author_profiles:
            display_author_profiles(author_profiles)

            # Distinguishability test
            test_result = author_distinguishability_test(author_profiles)
            console.print(f"\n[bold]Author Distinguishability Test:[/bold]")
            console.print(f"  Result: {test_result['interpretation']}")
            if 'p_value' in test_result:
                console.print(f"  χ² = {test_result['chi2_statistic']:.2f}, p = {test_result['p_value']:.4f}")

            output_data["author_profiles"] = {
                k: asdict(v) for k, v in author_profiles.items()
            }
            output_data["author_distinguishability"] = test_result
        else:
            console.print("[yellow]Not enough multi-work authors for analysis[/yellow]")

    if analysis in ['genre', 'both']:
        console.print("\n[bold]Genre Analysis[/bold]")
        genre_profiles = analyze_by_genre(results)

        if genre_profiles:
            display_genre_profiles(genre_profiles)

            output_data["genre_profiles"] = {
                k: asdict(v) for k, v in genre_profiles.items()
            }
        else:
            console.print("[yellow]Not enough genre data for analysis[/yellow]")

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    console.print(f"\n[green]Results saved to {output_file}[/green]")


if __name__ == "__main__":
    main()
