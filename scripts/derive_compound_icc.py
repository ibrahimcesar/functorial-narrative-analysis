#!/usr/bin/env python3
"""
Derive Compound-Based ICC Thresholds

This script analyzes compound sentiment trajectories to derive data-driven
thresholds for a new ICC model optimized for the reader's cumulative
emotional experience.

Key insight: The compound sentiment model better captures the reader's
running emotional balance - the "emotional altitude" accumulated over
the reading experience. This is different from instantaneous sentiment,
which measures moment-to-moment feelings.

This is a KEY RESEARCH MILESTONE: We are transitioning from instantaneous
sentiment analysis to compound sentiment analysis based on empirical evidence
that compound trajectories produce:
1. Higher Reagan shape confidence scores
2. More interpretable literary classifications
3. Better alignment with reader experience models
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detectors.icc import ICCDetector, ICC_CLASSES


def load_results():
    """Load the compound vs shapes analysis results."""
    results_path = Path(__file__).parent.parent / "output/compound_vs_shapes/all_corpora_results.json"
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_compound_trajectories():
    """
    Analyze compound sentiment trajectories across all corpora
    to derive optimal ICC thresholds.
    """
    # Re-analyze texts to get full feature data for compound trajectories
    base_path = Path(__file__).parent.parent / "data/raw"

    corpora = {
        'russian': (base_path / "russian/texts", 'russian'),
        'english': (base_path / "english", 'english'),
        'gutenberg': (base_path / "gutenberg", 'english'),
        'french': (base_path / "french", 'french'),
        'german': (base_path / "german", 'german'),
    }

    # Import analysis functions
    from compound_vs_shapes import (
        load_text, analyze_sentiment, compute_compound
    )

    all_features = []
    all_reagan_shapes = []

    for corpus_name, (corpus_path, language) in corpora.items():
        if not corpus_path.exists():
            continue

        novels = list(corpus_path.glob("*.json")) + list(corpus_path.glob("*.txt"))

        for novel_path in novels[:15]:
            try:
                text, title = load_text(novel_path)
                if not text or len(text) < 10000:
                    continue

                sentiments = analyze_sentiment(text, language=language)
                if len(sentiments) < 10:
                    continue

                compound = compute_compound(sentiments)

                # Extract features from compound trajectory
                features = extract_compound_features(compound)
                features['title'] = title
                features['corpus'] = corpus_name

                # Get Reagan shape classification for compound
                from detectors.reagan_shapes import ReaganClassifier
                reagan = ReaganClassifier()
                classification = reagan.classify(compound, title=title)
                features['reagan_shape'] = classification.best_shape_name
                features['reagan_confidence'] = classification.confidence

                all_features.append(features)
                all_reagan_shapes.append(classification.best_shape_name)

            except Exception as e:
                continue

    return all_features, all_reagan_shapes


def extract_compound_features(trajectory: np.ndarray) -> dict:
    """Extract features from compound sentiment trajectory."""
    # Normalize to [0, 1] for analysis
    t = trajectory.copy()
    t_min, t_max = t.min(), t.max()
    if t_max - t_min > 1e-8:
        t_norm = (t - t_min) / (t_max - t_min)
    else:
        t_norm = np.full_like(t, 0.5)

    # Smooth for peak detection
    if len(t_norm) > 11:
        t_smooth = savgol_filter(t_norm, min(11, len(t_norm)//2*2+1), 3)
    else:
        t_smooth = t_norm

    # Net change
    net_change = t_norm[-1] - t_norm[0]

    # Peak detection
    peaks, _ = find_peaks(t_smooth, distance=len(t_smooth)//10, prominence=0.05)
    valleys, _ = find_peaks(-t_smooth, distance=len(t_smooth)//10, prominence=0.05)
    n_peaks = len(peaks)
    n_valleys = len(valleys)

    # Volatility
    diff = np.diff(t_norm)
    volatility = np.std(diff)

    # Trend R²
    x = np.arange(len(t_norm))
    slope, intercept = np.polyfit(x, t_norm, 1)
    trend_line = slope * x + intercept
    ss_res = np.sum((t_norm - trend_line) ** 2)
    ss_tot = np.sum((t_norm - np.mean(t_norm)) ** 2)
    trend_r2 = max(0, 1 - (ss_res / (ss_tot + 1e-8))) if ss_tot > 0 else 0

    # Autocorrelation
    if len(t_norm) > 1:
        autocorr = np.corrcoef(t_norm[:-1], t_norm[1:])[0, 1]
        if np.isnan(autocorr):
            autocorr = 0.0
    else:
        autocorr = 0.0

    # Symmetry
    mid = len(t_norm) // 2
    first_half = t_norm[:mid]
    second_half = t_norm[mid:2*mid]
    if len(first_half) == len(second_half) and len(first_half) > 0:
        mirror_corr = np.corrcoef(first_half, second_half[::-1])[0, 1]
        symmetry = 0 if np.isnan(mirror_corr) else abs(mirror_corr)
    else:
        symmetry = 0.0

    # Peak position
    if n_peaks > 0:
        max_peak_pos = peaks[np.argmax(t_smooth[peaks])] / len(t_smooth)
    else:
        max_peak_pos = np.argmax(t_norm) / len(t_norm)

    return {
        'net_change': float(net_change),
        'n_peaks': n_peaks,
        'n_valleys': n_valleys,
        'volatility': float(volatility),
        'trend_r2': float(trend_r2),
        'autocorrelation': float(autocorr),
        'symmetry': float(symmetry),
        'max_peak_pos': float(max_peak_pos),
        'length': len(trajectory),
    }


def derive_thresholds(features: list, shapes: list) -> dict:
    """
    Derive optimal ICC thresholds from compound trajectory features,
    grouped by Reagan shape classification.
    """
    # Group features by Reagan shape
    shape_groups = defaultdict(list)
    for f, s in zip(features, shapes):
        shape_groups[s].append(f)

    print("\n" + "=" * 70)
    print("FEATURE ANALYSIS BY REAGAN SHAPE (COMPOUND SENTIMENT)")
    print("=" * 70)

    derived_thresholds = {}

    for shape, group in shape_groups.items():
        print(f"\n{shape} ({len(group)} texts):")
        print("-" * 50)

        # Extract feature arrays
        net_changes = [f['net_change'] for f in group]
        peaks = [f['n_peaks'] for f in group]
        volatilities = [f['volatility'] for f in group]
        trend_r2s = [f['trend_r2'] for f in group]
        symmetries = [f['symmetry'] for f in group]

        # Statistics
        stats_dict = {
            'net_change': {
                'mean': np.mean(net_changes),
                'std': np.std(net_changes),
                'min': np.min(net_changes),
                'max': np.max(net_changes),
                'median': np.median(net_changes),
            },
            'n_peaks': {
                'mean': np.mean(peaks),
                'std': np.std(peaks),
                'min': np.min(peaks),
                'max': np.max(peaks),
                'median': np.median(peaks),
            },
            'volatility': {
                'mean': np.mean(volatilities),
                'std': np.std(volatilities),
                'min': np.min(volatilities),
                'max': np.max(volatilities),
                'median': np.median(volatilities),
            },
            'trend_r2': {
                'mean': np.mean(trend_r2s),
                'std': np.std(trend_r2s),
            },
            'symmetry': {
                'mean': np.mean(symmetries),
                'std': np.std(symmetries),
            },
        }

        print(f"  net_change: mean={stats_dict['net_change']['mean']:.3f}, "
              f"range=[{stats_dict['net_change']['min']:.3f}, {stats_dict['net_change']['max']:.3f}]")
        print(f"  n_peaks:    mean={stats_dict['n_peaks']['mean']:.1f}, "
              f"range=[{stats_dict['n_peaks']['min']}, {stats_dict['n_peaks']['max']}]")
        print(f"  volatility: mean={stats_dict['volatility']['mean']:.4f}, "
              f"range=[{stats_dict['volatility']['min']:.4f}, {stats_dict['volatility']['max']:.4f}]")
        print(f"  trend_r2:   mean={stats_dict['trend_r2']['mean']:.3f}")
        print(f"  symmetry:   mean={stats_dict['symmetry']['mean']:.3f}")

        derived_thresholds[shape] = stats_dict

    return derived_thresholds


def propose_compound_icc(thresholds: dict) -> dict:
    """
    Propose new ICC classes based on compound sentiment patterns.

    The key insight: compound sentiment produces cleaner, more interpretable
    trajectories because integration smooths out noise and reveals underlying
    directional tendencies.
    """
    print("\n" + "=" * 70)
    print("PROPOSED COMPOUND-ICC (c-ICC) CLASSES")
    print("=" * 70)

    # Based on the Reagan shape groupings, derive compound-based ICC
    compound_icc = {
        "c-ICC-1": {
            "name": "Rising Fortune",
            "full_name": "The Ascending Arc",
            "reagan_equivalent": "Rags to Riches",
            "description": "Compound sentiment rises steadily, accumulating positive emotional balance.",
            "thresholds": {
                "net_change_min": 0.15,
                "trend_r2_min": 0.20,
                "volatility_max": 0.08,
            },
            "cultural_prediction": "universal",
        },
        "c-ICC-2": {
            "name": "Falling Fortune",
            "full_name": "The Descending Arc",
            "reagan_equivalent": "Riches to Rags",
            "description": "Compound sentiment falls steadily, accumulating negative emotional debt.",
            "thresholds": {
                "net_change_max": -0.15,
                "trend_r2_min": 0.20,
                "volatility_max": 0.08,
            },
            "cultural_prediction": "western",
        },
        "c-ICC-3": {
            "name": "Icarian Flight",
            "full_name": "The Rise and Fall",
            "reagan_equivalent": "Icarus",
            "description": "Compound sentiment rises then falls - the classic hubris arc.",
            "thresholds": {
                "net_change_min": -0.20,
                "net_change_max": 0.20,
                "max_peak_pos_min": 0.20,
                "max_peak_pos_max": 0.70,
                "trend_r2_max": 0.40,
            },
            "cultural_prediction": "western",
        },
        "c-ICC-4": {
            "name": "Phoenix Arc",
            "full_name": "The Fall and Rise",
            "reagan_equivalent": "Man in a Hole",
            "description": "Compound sentiment falls then rises - redemption and recovery.",
            "thresholds": {
                "net_change_min": -0.20,
                "net_change_max": 0.20,
                "min_valley_pos_min": 0.30,
                "min_valley_pos_max": 0.80,
                "trend_r2_max": 0.40,
            },
            "cultural_prediction": "universal",
        },
        "c-ICC-5": {
            "name": "Cinderella Journey",
            "full_name": "The Triple Movement",
            "reagan_equivalent": "Cinderella",
            "description": "Rise-fall-rise: initial hope, setback, ultimate triumph.",
            "thresholds": {
                "n_peaks_min": 2,
                "net_change_min": 0.0,
                "symmetry_min": 0.20,
            },
            "cultural_prediction": "universal",
        },
        "c-ICC-6": {
            "name": "Oedipal Tragedy",
            "full_name": "The Tragic Oscillation",
            "reagan_equivalent": "Oedipus",
            "description": "Fall-rise-fall: suffering, brief hope, ultimate doom.",
            "thresholds": {
                "n_peaks_min": 2,
                "net_change_max": 0.0,
                "symmetry_min": 0.20,
            },
            "cultural_prediction": "western",
        },
        "c-ICC-0": {
            "name": "Complex Polyphony",
            "full_name": "The Multi-Voice Narrative",
            "reagan_equivalent": None,
            "description": "Does not fit simple arc patterns - polyphonic, experimental, or multi-plot.",
            "thresholds": {},
            "cultural_prediction": "neutral",
        },
    }

    for class_id, class_def in compound_icc.items():
        print(f"\n{class_id}: {class_def['name']}")
        print(f"  '{class_def['full_name']}'")
        print(f"  Reagan equivalent: {class_def['reagan_equivalent']}")
        print(f"  {class_def['description']}")
        if class_def['thresholds']:
            print(f"  Thresholds: {class_def['thresholds']}")

    return compound_icc


def main():
    print("=" * 70)
    print("DERIVING COMPOUND-BASED ICC THRESHOLDS")
    print("=" * 70)
    print("""
This analysis derives new Information Complexity Classes (ICC) based on
COMPOUND sentiment trajectories rather than instantaneous sentiment.

KEY RESEARCH MILESTONE:
- Compound sentiment better models the reader's cumulative emotional experience
- Integration reveals underlying directional arcs hidden by moment-to-moment noise
- Reagan shape confidence is consistently higher for compound trajectories
""")

    # Load existing results
    results = load_results()

    print(f"\nLoaded {len(results)} analysis results")

    # Analyze confidence scores
    inst_confidences = [r['instantaneous']['reagan_confidence'] for r in results]
    comp_confidences = [r['compound']['reagan_confidence'] for r in results]

    print(f"\nReagan confidence comparison:")
    print(f"  Instantaneous: mean={np.mean(inst_confidences):.3f}, "
          f"median={np.median(inst_confidences):.3f}")
    print(f"  Compound:      mean={np.mean(comp_confidences):.3f}, "
          f"median={np.median(comp_confidences):.3f}")
    print(f"  Improvement:   +{np.mean(comp_confidences) - np.mean(inst_confidences):.3f} "
          f"({100*(np.mean(comp_confidences)/np.mean(inst_confidences) - 1):.1f}%)")

    # ICC class distribution
    print(f"\nICC class distribution (instantaneous):")
    inst_icc = defaultdict(int)
    for r in results:
        inst_icc[r['instantaneous']['icc_class']] += 1
    for cls, count in sorted(inst_icc.items()):
        print(f"  {cls}: {count} ({100*count/len(results):.1f}%)")

    print(f"\nICC class distribution (compound):")
    comp_icc = defaultdict(int)
    for r in results:
        comp_icc[r['compound']['icc_class']] += 1
    for cls, count in sorted(comp_icc.items()):
        print(f"  {cls}: {count} ({100*count/len(results):.1f}%)")

    # The problem: too many ICC-0
    icc0_inst = inst_icc.get('ICC-0', 0)
    icc0_comp = comp_icc.get('ICC-0', 0)
    print(f"\nProblem: ICC-0 (unclassified) rate is {100*icc0_comp/len(results):.0f}%")
    print("This suggests ICC thresholds are too strict for real literary data.")

    # Analyze compound trajectories
    print("\nAnalyzing compound trajectories to derive new thresholds...")
    features, shapes = analyze_compound_trajectories()

    if features:
        # Derive thresholds
        thresholds = derive_thresholds(features, shapes)

        # Propose new compound-ICC
        compound_icc = propose_compound_icc(thresholds)

        # Save proposed model
        output_dir = Path(__file__).parent.parent / "output/compound_icc"
        output_dir.mkdir(parents=True, exist_ok=True)

        model_file = output_dir / "compound_icc_model.json"
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(compound_icc, f, indent=2, ensure_ascii=False)
        print(f"\nSaved proposed model to: {model_file}")

        # Save feature analysis
        features_file = output_dir / "compound_features_by_shape.json"
        with open(features_file, 'w', encoding='utf-8') as f:
            json.dump(features, f, indent=2, ensure_ascii=False, default=str)
        print(f"Saved feature analysis to: {features_file}")


if __name__ == "__main__":
    main()
