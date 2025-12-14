#!/usr/bin/env python3
"""
Analyze a narrative using the full functorial framework.

Usage:
    # Analyze Anna Karenina from Project Gutenberg
    python scripts/analyze_narrative.py --gutenberg 1399

    # Analyze from local file
    python scripts/analyze_narrative.py --file /path/to/text.txt --title "My Book"

    # Analyze from corpus
    python scripts/analyze_narrative.py --corpus-search "Anna Karenina"

    # Full analysis with all functors
    python scripts/analyze_narrative.py --gutenberg 1399 --full
"""

import argparse
import json
import sys
from pathlib import Path
import urllib.request

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


from typing import Optional, Tuple, Dict, Any


def download_gutenberg(book_id: int) -> Tuple[str, str]:
    """Download a book from Project Gutenberg."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    print(f"Downloading from {url}...")

    try:
        with urllib.request.urlopen(url) as response:
            text = response.read().decode('utf-8-sig')
    except Exception as e:
        print(f"Error downloading: {e}")
        sys.exit(1)

    # Try to extract title from Gutenberg header
    title = f"Gutenberg #{book_id}"
    for line in text[:5000].split('\n'):
        if line.startswith('Title:'):
            title = line.replace('Title:', '').strip()
            break

    # Remove Gutenberg header/footer
    start_markers = ["*** START OF", "***START OF"]
    end_markers = ["*** END OF", "***END OF"]

    for marker in start_markers:
        if marker in text:
            text = text.split(marker, 1)[1]
            # Skip the rest of the marker line
            text = text.split('\n', 1)[1] if '\n' in text else text
            break

    for marker in end_markers:
        if marker in text:
            text = text.split(marker, 1)[0]
            break

    return text.strip(), title


def search_corpus(query: str, corpus_dir: str) -> Optional[Tuple[str, str]]:
    """Search for a book in the corpus by title."""
    corpus_path = Path(corpus_dir)
    index_file = corpus_path / "index.json"

    if not index_file.exists():
        print(f"Corpus index not found at {index_file}")
        return None

    index = json.loads(index_file.read_text(encoding='utf-8'))
    query_lower = query.lower()

    for book in index.get("books", []):
        title = book.get("title", "").lower()
        if query_lower in title:
            book_file = corpus_path / "books" / f"{book['id']}.json"
            if book_file.exists():
                data = json.loads(book_file.read_text(encoding='utf-8'))
                return data.get("content", ""), data.get("title", "Unknown")

    print(f"No book matching '{query}' found in corpus")
    return None


def analyze_full(text: str, title: str, narrative_id: str):
    """Full functorial analysis with all components."""
    from src.categories import CategoryNarr
    from src.categories.integration import (
        analyze_morphism_patterns,
        NarrativeFunctorialAnalyzer
    )
    from src.detectors.icc import ICCDetector
    from src.functors.sentiment import SentimentFunctor
    from src.functors.entropy import EntropyFunctor

    print("=" * 70)
    print(f"FUNCTORIAL NARRATIVE ANALYSIS: {title}")
    print("=" * 70)

    # 1. Construct Category Narr
    print("\n[1/5] Constructing Category Narr...")

    # Feature extractors for category construction
    def tension_heuristic(text: str) -> float:
        """Simple tension based on punctuation density."""
        if not text:
            return 0.5
        exclaim = text.count('!')
        question = text.count('?')
        ellipsis = text.count('...')
        words = max(1, len(text.split()))
        return min(1.0, (exclaim * 0.3 + question * 0.2 + ellipsis * 0.1) / (words / 100))

    category = CategoryNarr.from_text(
        text=text,
        narrative_id=narrative_id,
        title=title,
        n_states=30,  # 30 states for a novel
        feature_extractors={
            "tension": tension_heuristic,
            "length": lambda t: len(t.split()),
        }
    )

    print(f"   Objects (states): {len(category.objects)}")
    print(f"   Morphisms (transitions): {len(category.morphisms)}")

    # Verify category laws
    laws = category.verify_category_laws()
    law_status = "✓" if all(laws.values()) else "✗"
    print(f"   Category laws: {law_status}")

    # 2. Analyze morphism patterns
    print("\n[2/5] Analyzing morphism patterns...")
    patterns = analyze_morphism_patterns(category)

    print(f"   Arc length: {patterns['arc_length']}")
    print(f"   Peak intensity: {patterns['peak_intensity']:.2f} at position {patterns['peak_position']:.1%}")
    print(f"   Morphism types:")
    for mtype, count in sorted(patterns['morphism_types'].items(), key=lambda x: -x[1]):
        print(f"      {mtype}: {count}")

    # 3. Apply observation functors
    print("\n[3/5] Applying observation functors...")

    try:
        sentiment_functor = SentimentFunctor()
        sentiment_traj = sentiment_functor.process_text(text)
        print(f"   Sentiment: range [{sentiment_traj.values.min():.2f}, {sentiment_traj.values.max():.2f}]")
    except Exception as e:
        print(f"   Sentiment: failed ({e})")
        sentiment_traj = None

    try:
        entropy_functor = EntropyFunctor()
        entropy_traj = entropy_functor.process_text(text)
        print(f"   Entropy: range [{entropy_traj.values.min():.2f}, {entropy_traj.values.max():.2f}]")
    except Exception as e:
        print(f"   Entropy: failed ({e})")
        entropy_traj = None

    # 4. ICC Classification
    print("\n[4/5] ICC Classification...")
    detector = ICCDetector()

    # Use sentiment trajectory for ICC if available
    if sentiment_traj:
        normalized = sentiment_traj.normalize()
        icc_result = detector.detect(normalized.values, trajectory_id=narrative_id, title=title)
    else:
        # Create simple trajectory from category features
        states = sorted(category.objects, key=lambda s: s.position)
        simple_traj = np.array([s.features.get('tension', 0.5) for s in states])
        icc_result = detector.detect(simple_traj, trajectory_id=narrative_id, title=title)

    print(f"   ICC Class: {icc_result.icc_class} - {icc_result.class_name}")
    print(f"   Cultural prediction: {icc_result.cultural_prediction}")
    print(f"   Confidence: {icc_result.confidence:.2f}")

    if icc_result.notes:
        for note in icc_result.notes:
            print(f"   {note}")

    # 5. Key features
    print("\n[5/5] Key narrative features:")
    features = icc_result.features
    print(f"   Net change: {features['net_change']:+.2f}")
    print(f"   Peak count: {features['n_peaks']}")
    print(f"   Volatility: {features['volatility']:.3f}")
    print(f"   Trend R²: {features['trend_r2']:.3f}")
    print(f"   Symmetry: {features['symmetry']:.3f}")
    print(f"   Structure score: {features['structure_score']:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Title: {title}
Category Narr: {len(category.objects)} states, {len(category.morphisms)} morphisms
ICC Classification: {icc_result.icc_class} ({icc_result.class_name})
Cultural Pattern: {icc_result.cultural_prediction.upper()}
Peak Position: {patterns['peak_position']:.1%} through narrative
Overall Arc: {"Rising" if features['net_change'] > 0.1 else "Falling" if features['net_change'] < -0.1 else "Cyclical/Stable"}
""")

    return {
        "title": title,
        "category": category.summary(),
        "morphism_patterns": patterns,
        "icc_result": icc_result.to_dict(),
    }


def analyze_simple(text: str, title: str, narrative_id: str):
    """Simple ICC-only analysis."""
    from src.detectors.icc import ICCDetector
    from src.functors.sentiment import SentimentFunctor

    print("=" * 70)
    print(f"NARRATIVE ANALYSIS: {title}")
    print("=" * 70)

    # Apply sentiment functor
    print("\nApplying sentiment functor...")
    try:
        functor = SentimentFunctor()
        trajectory = functor.process_text(text)
        print(f"   Trajectory length: {len(trajectory.values)} points")
        print(f"   Sentiment range: [{trajectory.values.min():.2f}, {trajectory.values.max():.2f}]")
    except Exception as e:
        print(f"   Failed: {e}")
        # Fallback to simple word-count based trajectory
        words = text.split()
        chunk_size = max(1, len(words) // 100)
        values = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            # Simple heuristic
            pos = sum(1 for w in ['happy', 'good', 'love', 'joy', 'hope'] if w in chunk.lower())
            neg = sum(1 for w in ['sad', 'bad', 'hate', 'fear', 'death', 'die'] if w in chunk.lower())
            values.append((pos - neg) / max(1, len(chunk.split())))
        from src.functors.base import Trajectory
        trajectory = Trajectory(
            values=np.array(values),
            time_points=np.linspace(0, 1, len(values)),
            functor_name="simple_sentiment"
        )

    # ICC Classification
    print("\nICC Classification...")
    detector = ICCDetector()
    normalized = trajectory.normalize()
    result = detector.detect(normalized.values, trajectory_id=narrative_id, title=title)

    print(f"\n   Class: {result.icc_class} - {result.class_name}")
    print(f"   Cultural prediction: {result.cultural_prediction}")
    print(f"   Confidence: {result.confidence:.2f}")

    print("\nKey Features:")
    features = result.features
    print(f"   Net change: {features['net_change']:+.3f}")
    print(f"   Peak count: {features['n_peaks']}")
    print(f"   Volatility: {features['volatility']:.3f}")
    print(f"   Trend R²: {features['trend_r2']:.3f}")

    if result.notes:
        print("\nNotes:")
        for note in result.notes:
            print(f"   {note}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a narrative using functorial framework"
    )
    parser.add_argument(
        "--gutenberg",
        type=int,
        help="Project Gutenberg book ID (e.g., 1399 for Anna Karenina)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to local text file"
    )
    parser.add_argument(
        "--corpus-search",
        type=str,
        help="Search for book in corpus by title"
    )
    parser.add_argument(
        "--corpus-dir",
        default="/Volumes/MacExt/narrative_corpus",
        help="Corpus directory"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Override title"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full analysis with Category Narr and all functors"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Get text
    text = None
    title = args.title

    if args.gutenberg:
        text, auto_title = download_gutenberg(args.gutenberg)
        title = title or auto_title
        narrative_id = f"gutenberg_{args.gutenberg}"
    elif args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {args.file}")
            sys.exit(1)
        text = path.read_text(encoding='utf-8')
        title = title or path.stem
        narrative_id = path.stem
    elif args.corpus_search:
        result = search_corpus(args.corpus_search, args.corpus_dir)
        if result:
            text, auto_title = result
            title = title or auto_title
            narrative_id = auto_title.lower().replace(' ', '_')[:30]

    if not text:
        print("No text provided. Use --gutenberg, --file, or --corpus-search")
        parser.print_help()
        sys.exit(1)

    print(f"Text loaded: {len(text):,} characters, ~{len(text.split()):,} words")

    # Analyze
    if args.full:
        result = analyze_full(text, title, narrative_id)
    else:
        result = analyze_simple(text, title, narrative_id)

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        if hasattr(result, 'to_dict'):
            result = result.to_dict()
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
