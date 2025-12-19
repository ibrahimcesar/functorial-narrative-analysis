#!/usr/bin/env python3
"""
Compound Sentiment vs Story Shapes Analysis

Compares the compound (cumulative) sentiment trajectory against:
1. Reagan et al.'s Six Story Shapes
2. Information Complexity Classes (ICC)

Key question: Does the compound sentiment preserve the same shape classification
as instantaneous sentiment, or does integration reveal different patterns?

Theory:
- Instantaneous sentiment: "How does the reader feel at this moment?"
- Compound sentiment: "What is the cumulative emotional debt/credit?"

The compound view may reveal that a novel with many ups and downs
(high instantaneous volatility) actually has a clear directional arc
when integrated - like a stock price with daily fluctuations but
a clear long-term trend.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detectors.reagan_shapes import ReaganClassifier, REAGAN_SHAPES
from detectors.icc import ICCDetector, ICC_CLASSES


def load_text(file_path: Path) -> Tuple[str, str]:
    """Load text and title from JSON or TXT file."""
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('text', ''), data.get('title', file_path.stem)
    else:  # .txt
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Clean up title from filename
        title = file_path.stem.replace('_', ' ').title()
        return text, title


def get_sentiment_analyzer(language: str):
    """Get appropriate sentiment analyzer for language."""
    if language == 'russian':
        from functors.russian_sentiment import RussianSentimentAnalyzer
        return RussianSentimentAnalyzer(use_stemming=True), 'compound'
    elif language == 'japanese':
        from functors.japanese_sentiment import JapaneseSentimentFunctor
        functor = JapaneseSentimentFunctor()
        return functor, 'functor'
    else:  # English and others - use VADER
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer(), 'vader'


def analyze_sentiment(text: str, language: str = 'english',
                      window_size: int = 500) -> np.ndarray:
    """Analyze sentiment for sliding windows."""
    analyzer, analyzer_type = get_sentiment_analyzer(language)
    words = text.split()
    step = window_size // 2

    sentiments = []
    for i in range(0, max(1, len(words) - window_size), step):
        window = ' '.join(words[i:i + window_size])
        if not window.strip():
            continue

        if analyzer_type == 'compound':
            result = analyzer.analyze(window)
            sentiments.append(result.compound)
        elif analyzer_type == 'vader':
            scores = analyzer.polarity_scores(window)
            sentiments.append(scores['compound'])
        elif analyzer_type == 'functor':
            # Japanese functor returns trajectory
            traj = analyzer.process_text(window, window_size=100, overlap=50)
            sentiments.append(float(np.mean(traj.values)))

    if not sentiments:
        return np.array([0.0])

    return np.array(sentiments)


def compute_compound(sentiments: np.ndarray, center: bool = True) -> np.ndarray:
    """Compute compound (cumulative) sentiment."""
    if center:
        centered = sentiments - np.mean(sentiments)
    else:
        centered = sentiments

    compound = np.cumsum(centered)

    # Normalize
    max_abs = max(abs(compound.min()), abs(compound.max()), 1)
    return compound / max_abs


def compare_shapes(instantaneous: np.ndarray, compound: np.ndarray,
                   title: str) -> Dict:
    """
    Compare Reagan shape and ICC classifications for both
    instantaneous and compound sentiment.
    """
    reagan = ReaganClassifier(smooth_sigma=3.0)
    icc = ICCDetector()

    # Classify instantaneous
    inst_reagan = reagan.classify(instantaneous, title=title)
    inst_icc = icc.detect(instantaneous, title=title)

    # Classify compound
    comp_reagan = reagan.classify(compound, title=title)
    comp_icc = icc.detect(compound, title=title)

    return {
        "instantaneous": {
            "reagan_shape": inst_reagan.best_shape_name,
            "reagan_confidence": inst_reagan.confidence,
            "reagan_pattern": REAGAN_SHAPES[inst_reagan.best_shape].pattern,
            "icc_class": inst_icc.icc_class,
            "icc_name": inst_icc.class_name,
            "icc_cultural": inst_icc.cultural_prediction,
            "features": inst_icc.features,
        },
        "compound": {
            "reagan_shape": comp_reagan.best_shape_name,
            "reagan_confidence": comp_reagan.confidence,
            "reagan_pattern": REAGAN_SHAPES[comp_reagan.best_shape].pattern,
            "icc_class": comp_icc.icc_class,
            "icc_name": comp_icc.class_name,
            "icc_cultural": comp_icc.cultural_prediction,
            "features": comp_icc.features,
        },
        "shape_preserved": inst_reagan.best_shape == comp_reagan.best_shape,
        "icc_preserved": inst_icc.icc_class == comp_icc.icc_class,
    }


def plot_comparison(instantaneous: np.ndarray, compound: np.ndarray,
                    comparison: Dict, title: str, output_path: Path):
    """Create visualization comparing instantaneous vs compound classifications."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Smooth for visualization
    inst_smooth = gaussian_filter1d(instantaneous, sigma=3)
    comp_smooth = gaussian_filter1d(compound, sigma=3)
    x = np.linspace(0, 1, len(instantaneous))

    # Top left: Instantaneous sentiment
    ax1 = axes[0, 0]
    ax1.fill_between(x, 0, inst_smooth, where=inst_smooth > 0,
                     color='#2ecc71', alpha=0.4, label='Positive')
    ax1.fill_between(x, 0, inst_smooth, where=inst_smooth < 0,
                     color='#e74c3c', alpha=0.4, label='Negative')
    ax1.plot(x, inst_smooth, color='#2c3e50', linewidth=2)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title(f'Instantaneous Sentiment\n'
                  f'Reagan: {comparison["instantaneous"]["reagan_shape"]} '
                  f'({comparison["instantaneous"]["reagan_pattern"]}) '
                  f'[{comparison["instantaneous"]["reagan_confidence"]:.2f}]\n'
                  f'ICC: {comparison["instantaneous"]["icc_class"]} - '
                  f'{comparison["instantaneous"]["icc_name"]}',
                  fontsize=11)
    ax1.set_ylabel('Sentiment')
    ax1.set_ylim(-1.1, 1.1)
    ax1.legend(loc='upper right')

    # Top right: Compound sentiment
    ax2 = axes[0, 1]
    # Color gradient based on value
    for i in range(len(x) - 1):
        color = '#2ecc71' if comp_smooth[i] > 0 else '#e74c3c'
        ax2.fill_between([x[i], x[i+1]], [0, 0],
                        [comp_smooth[i], comp_smooth[i+1]],
                        color=color, alpha=0.5)
    ax2.plot(x, comp_smooth, color='#2c3e50', linewidth=2)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title(f'Compound (Cumulative) Sentiment\n'
                  f'Reagan: {comparison["compound"]["reagan_shape"]} '
                  f'({comparison["compound"]["reagan_pattern"]}) '
                  f'[{comparison["compound"]["reagan_confidence"]:.2f}]\n'
                  f'ICC: {comparison["compound"]["icc_class"]} - '
                  f'{comparison["compound"]["icc_name"]}',
                  fontsize=11)
    ax2.set_ylabel('Emotional Altitude')

    # Bottom left: Reagan canonical shapes overlay
    ax3 = axes[1, 0]
    reagan = ReaganClassifier()

    # Plot the classified shape for instantaneous
    inst_shape_key = [k for k, v in REAGAN_SHAPES.items()
                      if v.name == comparison["instantaneous"]["reagan_shape"]][0]
    canonical = reagan.canonical_patterns[inst_shape_key]

    # Normalize instantaneous to [0,1] for comparison
    inst_norm = (inst_smooth - inst_smooth.min()) / (inst_smooth.max() - inst_smooth.min() + 1e-8)

    ax3.plot(x, inst_norm, color='#3498db', linewidth=2, label='Actual (normalized)')
    ax3.plot(reagan.x, canonical, color='#e74c3c', linewidth=2,
             linestyle='--', label=f'Canonical: {comparison["instantaneous"]["reagan_shape"]}')
    ax3.set_title('Instantaneous vs Reagan Canonical Shape', fontsize=11)
    ax3.set_ylabel('Normalized Value')
    ax3.set_xlabel('Narrative Progress')
    ax3.legend()
    ax3.set_ylim(-0.1, 1.1)

    # Bottom right: Compound shape comparison
    ax4 = axes[1, 1]
    comp_shape_key = [k for k, v in REAGAN_SHAPES.items()
                      if v.name == comparison["compound"]["reagan_shape"]][0]
    canonical_comp = reagan.canonical_patterns[comp_shape_key]

    # Normalize compound to [0,1]
    comp_norm = (comp_smooth - comp_smooth.min()) / (comp_smooth.max() - comp_smooth.min() + 1e-8)

    ax4.plot(x, comp_norm, color='#9b59b6', linewidth=2, label='Actual (normalized)')
    ax4.plot(reagan.x, canonical_comp, color='#e74c3c', linewidth=2,
             linestyle='--', label=f'Canonical: {comparison["compound"]["reagan_shape"]}')
    ax4.set_title('Compound vs Reagan Canonical Shape', fontsize=11)
    ax4.set_ylabel('Normalized Value')
    ax4.set_xlabel('Narrative Progress')
    ax4.legend()
    ax4.set_ylim(-0.1, 1.1)

    # Main title
    shape_match = "✓ SAME" if comparison["shape_preserved"] else "✗ DIFFERENT"
    icc_match = "✓ SAME" if comparison["icc_preserved"] else "✗ DIFFERENT"

    fig.suptitle(f'{title}\n'
                 f'Reagan Shape: {shape_match}  |  ICC Class: {icc_match}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return fig


def analyze_corpus(corpus_path: Path, language: str, output_dir: Path,
                   corpus_name: str) -> List[Dict]:
    """Analyze a single corpus."""
    output_subdir = output_dir / corpus_name
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Find all text files
    novels = list(corpus_path.glob("*.json")) + list(corpus_path.glob("*.txt"))

    # Also check subdirectories
    if not novels:
        for subdir in corpus_path.iterdir():
            if subdir.is_dir():
                novels.extend(subdir.glob("*.json"))
                novels.extend(subdir.glob("*.txt"))

    if not novels:
        print(f"  No texts found in {corpus_path}")
        return []

    print(f"\n  Found {len(novels)} texts")

    results = []

    for novel_path in novels[:15]:  # Limit to 15 per corpus for speed
        try:
            text, title = load_text(novel_path)
            if not text or len(text) < 10000:  # Skip very short texts
                continue

            print(f"    - {title[:50]}...", end=" ", flush=True)

            # Analyze
            sentiments = analyze_sentiment(text, language=language)
            if len(sentiments) < 10:
                print("[too short]")
                continue

            compound = compute_compound(sentiments)

            # Compare
            comparison = compare_shapes(sentiments, compound, title)
            comparison["title"] = title
            comparison["file"] = novel_path.stem
            comparison["corpus"] = corpus_name
            comparison["language"] = language
            results.append(comparison)

            # Status
            shape_status = "=" if comparison["shape_preserved"] else "→"
            print(f"Reagan: {comparison['instantaneous']['reagan_shape'][:10]} "
                  f"{shape_status} {comparison['compound']['reagan_shape'][:10]}")

            # Create visualization
            plot_path = output_subdir / f"{novel_path.stem}_shape_comparison.png"
            plot_comparison(sentiments, compound, comparison, title, plot_path)

        except Exception as e:
            print(f"[error: {str(e)[:30]}]")
            continue

    return results


def main():
    """Analyze multiple corpora for compound vs instantaneous shape classification."""

    parser = argparse.ArgumentParser(description='Compound vs Shapes Analysis')
    parser.add_argument('--corpus', type=str, default='all',
                        help='Corpus to analyze: russian, english, japanese, french, or all')
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent / "data/raw"
    output_dir = Path(__file__).parent.parent / "output/compound_vs_shapes"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define corpora with their paths and languages
    corpora = {
        'russian': (base_path / "russian/texts", 'russian'),
        'english': (base_path / "english", 'english'),
        'gutenberg': (base_path / "gutenberg", 'english'),
        'french': (base_path / "french", 'french'),
        'german': (base_path / "german", 'german'),
        'aozora': (base_path / "aozora", 'japanese'),
    }

    if args.corpus != 'all':
        if args.corpus in corpora:
            corpora = {args.corpus: corpora[args.corpus]}
        else:
            print(f"Unknown corpus: {args.corpus}")
            print(f"Available: {', '.join(corpora.keys())}")
            return

    all_results = []

    print("=" * 70)
    print("COMPOUND VS INSTANTANEOUS SENTIMENT: MULTI-CORPUS ANALYSIS")
    print("=" * 70)

    for corpus_name, (corpus_path, language) in corpora.items():
        print(f"\n{'='*70}")
        print(f"CORPUS: {corpus_name.upper()} ({language})")
        print(f"{'='*70}")

        if not corpus_path.exists():
            print(f"  Path not found: {corpus_path}")
            continue

        results = analyze_corpus(corpus_path, language, output_dir, corpus_name)
        all_results.extend(results)

    # Summary statistics across all corpora
    if not all_results:
        print("\nNo results to summarize.")
        return

    print("\n" + "=" * 70)
    print("CROSS-CORPUS SUMMARY")
    print("=" * 70)

    n_total = len(all_results)
    n_reagan_preserved = sum(1 for r in all_results if r["shape_preserved"])
    n_icc_preserved = sum(1 for r in all_results if r["icc_preserved"])

    print(f"\nTotal texts analyzed: {n_total}")
    print(f"Reagan shape preserved: {n_reagan_preserved}/{n_total} "
          f"({100*n_reagan_preserved/n_total:.0f}%)")
    print(f"ICC class preserved: {n_icc_preserved}/{n_total} "
          f"({100*n_icc_preserved/n_total:.0f}%)")

    # By corpus
    print("\nBy Corpus:")
    corpus_stats = {}
    for r in all_results:
        corpus = r.get("corpus", "unknown")
        if corpus not in corpus_stats:
            corpus_stats[corpus] = {"total": 0, "reagan_preserved": 0, "icc_preserved": 0}
        corpus_stats[corpus]["total"] += 1
        if r["shape_preserved"]:
            corpus_stats[corpus]["reagan_preserved"] += 1
        if r["icc_preserved"]:
            corpus_stats[corpus]["icc_preserved"] += 1

    for corpus, stats in corpus_stats.items():
        reagan_pct = 100 * stats["reagan_preserved"] / stats["total"] if stats["total"] > 0 else 0
        icc_pct = 100 * stats["icc_preserved"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {corpus}: {stats['total']} texts | "
              f"Reagan preserved: {reagan_pct:.0f}% | ICC preserved: {icc_pct:.0f}%")

    # Group by transition type
    print("\nReagan shape transitions (all corpora):")
    transitions = {}
    for r in all_results:
        key = f"{r['instantaneous']['reagan_shape']} → {r['compound']['reagan_shape']}"
        transitions[key] = transitions.get(key, [])
        transitions[key].append(f"{r['title'][:30]} ({r.get('corpus', '?')})")

    for trans, titles in sorted(transitions.items(), key=lambda x: -len(x[1])):
        print(f"  {trans}: {len(titles)} texts")

    # ICC transitions
    print("\nICC class transitions (all corpora):")
    icc_transitions = {}
    for r in all_results:
        key = f"{r['instantaneous']['icc_class']} → {r['compound']['icc_class']}"
        icc_transitions[key] = icc_transitions.get(key, [])
        icc_transitions[key].append(r["title"])

    for trans, titles in sorted(icc_transitions.items(), key=lambda x: -len(x[1])):
        print(f"  {trans}: {len(titles)} texts")

    # Save results
    results_file = output_dir / "all_corpora_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json_results = []
        for r in all_results:
            jr = {
                "title": r["title"],
                "file": r["file"],
                "corpus": r.get("corpus", "unknown"),
                "language": r.get("language", "unknown"),
                "shape_preserved": r["shape_preserved"],
                "icc_preserved": r["icc_preserved"],
                "instantaneous": {
                    "reagan_shape": r["instantaneous"]["reagan_shape"],
                    "reagan_confidence": float(r["instantaneous"]["reagan_confidence"]),
                    "icc_class": r["instantaneous"]["icc_class"],
                    "icc_name": r["instantaneous"]["icc_name"],
                },
                "compound": {
                    "reagan_shape": r["compound"]["reagan_shape"],
                    "reagan_confidence": float(r["compound"]["reagan_confidence"]),
                    "icc_class": r["compound"]["icc_class"],
                    "icc_name": r["compound"]["icc_name"],
                }
            }
            json_results.append(jr)
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_file}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
The compound (cumulative) sentiment reveals a different perspective:

1. INSTANTANEOUS sentiment answers: "How does the reader feel NOW?"
   - Captures moment-to-moment emotional swings
   - High volatility = many ups and downs
   - Reagan shapes describe the oscillation pattern

2. COMPOUND sentiment answers: "What is the emotional debt/credit accumulated?"
   - Integrates sentiment over time
   - A novel with balanced positive/negative stays near zero
   - A novel with net negative gradually descends
   - This reveals the "emotional gravity" of the narrative

When shapes DIFFER between instantaneous and compound:
   - The novel has oscillations that don't average out
   - There's a hidden directional tendency
   - Example: Crime and Punishment oscillates (Man in Hole)
     but compounds to a rise (redemption arc)

When shapes are PRESERVED:
   - The instantaneous shape IS the fundamental arc
   - The oscillations reinforce rather than cancel
   - Example: A tragedy falls both moment-to-moment AND cumulatively
""")


if __name__ == "__main__":
    main()
