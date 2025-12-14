#!/usr/bin/env python3
"""
Multi-Functor Visualization Suite

Creates comprehensive visualizations for multi-functor narrative analysis:
1. Multi-trajectory plot (all functors on one timeline)
2. Functor correlation heatmap
3. Radar chart of functor means
4. Pacing vs Arousal phase space
5. Character presence timeline
6. Narrative voice/POV distribution

Usage:
    python scripts/visualize_multi_functor.py -i data/raw/tolstoy/war_and_peace.json -o output/visualizations
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Wedge
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.monospace'] = ['JetBrains Mono', 'Fira Code', 'SF Mono', 'Monaco', 'Consolas', 'monospace']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'medium'

# Color palette for functors
FUNCTOR_COLORS = {
    'sentiment': '#E74C3C',     # Red
    'entropy': '#3498DB',        # Blue
    'arousal': '#F39C12',        # Orange
    'epistemic': '#9B59B6',      # Purple
    'pacing': '#2ECC71',         # Green
    'character': '#1ABC9C',      # Teal
    'voice': '#E91E63',          # Pink
}


def create_windows(text: str, window_size: int = 1000, overlap: int = 500) -> List[str]:
    """Create overlapping windows from text."""
    words = text.split()
    step = window_size - overlap
    windows = []
    for i in range(0, len(words), step):
        window = ' '.join(words[i:i + window_size])
        if len(window.split()) >= window_size // 2:
            windows.append(window)
    return windows if windows else [text]


def get_all_trajectories(text: str, window_size: int = 1000, overlap: int = 500):
    """Extract all functor trajectories from text."""
    from src.functors.sentiment import SentimentFunctor
    from src.functors.entropy import EntropyFunctor
    from src.functors.arousal import ArousalFunctor
    from src.functors.epistemic import EpistemicFunctor
    from src.functors.pacing import PacingFunctor
    from src.functors.character_presence import CharacterPresenceFunctor
    from src.functors.narrative_voice import NarrativeVoiceFunctor
    from src.detectors.icc import ICCDetector

    windows = create_windows(text, window_size, overlap)
    print(f"Created {len(windows)} windows")

    trajectories = {}

    # Sentiment
    print("Computing sentiment...")
    sentiment = SentimentFunctor(method="vader")
    trajectories['sentiment'] = sentiment(windows)

    # Entropy
    print("Computing entropy...")
    entropy = EntropyFunctor(method="combined")
    trajectories['entropy'] = entropy(windows)

    # Arousal
    print("Computing arousal...")
    arousal = ArousalFunctor()
    trajectories['arousal'] = arousal(windows)

    # Epistemic
    print("Computing epistemic...")
    epistemic = EpistemicFunctor()
    trajectories['epistemic'] = epistemic(windows)

    # Pacing
    print("Computing pacing...")
    pacing = PacingFunctor()
    trajectories['pacing'] = pacing(windows)

    # Character presence
    print("Computing character presence...")
    character = CharacterPresenceFunctor()
    trajectories['character'] = character(windows)

    # Narrative voice
    print("Computing narrative voice...")
    voice = NarrativeVoiceFunctor()
    trajectories['voice'] = voice(windows)

    # ICC classification
    detector = ICCDetector()
    icc_result = detector.detect(trajectories['sentiment'].values)

    return trajectories, icc_result


def smooth_trajectory(values: np.ndarray, window: int = 11) -> np.ndarray:
    """Smooth trajectory for visualization."""
    if len(values) > window:
        return savgol_filter(values, window, 3)
    return values


def plot_multi_trajectory(trajectories: Dict, title: str, icc_result, output_path: Path):
    """Plot all functor trajectories on a single timeline."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle(f'{title}\nICC Classification: {icc_result.icc_class} ({icc_result.class_name})',
                 fontsize=14, fontweight='bold')

    functor_list = ['sentiment', 'entropy', 'arousal', 'epistemic', 'pacing', 'character', 'voice']

    for idx, functor_name in enumerate(functor_list):
        ax = axes.flat[idx]
        traj = trajectories[functor_name]
        values = smooth_trajectory(traj.values)
        time = np.linspace(0, 100, len(values))

        ax.fill_between(time, values, alpha=0.3, color=FUNCTOR_COLORS[functor_name])
        ax.plot(time, values, color=FUNCTOR_COLORS[functor_name], linewidth=2, label=functor_name)
        ax.axhline(y=np.mean(values), color=FUNCTOR_COLORS[functor_name],
                   linestyle='--', alpha=0.5, label=f'mean: {np.mean(values):.2f}')

        ax.set_title(f'{functor_name.capitalize()} Trajectory', fontweight='bold')
        ax.set_xlabel('Narrative Progress (%)')
        ax.set_ylabel('Score')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Use last subplot for combined view
    ax = axes.flat[7]
    for functor_name in ['sentiment', 'arousal', 'pacing']:
        traj = trajectories[functor_name]
        values = smooth_trajectory(traj.values)
        time = np.linspace(0, 100, len(values))
        ax.plot(time, values, color=FUNCTOR_COLORS[functor_name],
                linewidth=2, label=functor_name.capitalize(), alpha=0.8)

    ax.set_title('Combined: Sentiment, Arousal, Pacing', fontweight='bold')
    ax.set_xlabel('Narrative Progress (%)')
    ax.set_ylabel('Score')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'{title.replace(" ", "_")}_multi_trajectory.png'
    plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / filename}")


def plot_correlation_heatmap(trajectories: Dict, title: str, output_path: Path):
    """Plot correlation heatmap between functors."""
    functor_names = list(trajectories.keys())
    n = len(functor_names)

    corr_matrix = np.zeros((n, n))

    for i, name1 in enumerate(functor_names):
        for j, name2 in enumerate(functor_names):
            v1 = trajectories[name1].values
            v2 = trajectories[name2].values
            # Align lengths
            min_len = min(len(v1), len(v2))
            if min_len > 2 and np.std(v1[:min_len]) > 1e-10 and np.std(v2[:min_len]) > 1e-10:
                corr, _ = pearsonr(v1[:min_len], v2[:min_len])
                corr_matrix[i, j] = corr
            else:
                corr_matrix[i, j] = 0 if i != j else 1

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([name.capitalize() for name in functor_names])
    ax.set_yticklabels([name.capitalize() for name in functor_names])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add correlation values
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")

    ax.set_title(f'{title}\nCross-Functor Correlations', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Pearson Correlation')

    plt.tight_layout()
    filename = f'{title.replace(" ", "_")}_correlation.png'
    plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / filename}")


def plot_radar_chart(trajectories: Dict, title: str, icc_result, output_path: Path):
    """Plot radar chart of functor means."""
    functor_names = list(trajectories.keys())
    means = [np.mean(trajectories[name].values) for name in functor_names]
    stds = [np.std(trajectories[name].values) for name in functor_names]

    # Number of variables
    N = len(functor_names)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    means += means[:1]
    stds += stds[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Draw the mean values
    ax.plot(angles, means, 'o-', linewidth=2, color='#3498DB', label='Mean')
    ax.fill(angles, means, alpha=0.25, color='#3498DB')

    # Draw variance indicators
    means_array = np.array(means)
    stds_array = np.array(stds)
    ax.fill_between(angles, means_array - stds_array, means_array + stds_array,
                   alpha=0.1, color='#E74C3C', label='±1 Std Dev')

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([name.capitalize() for name in functor_names], fontsize=12)
    ax.set_ylim(0, 1)

    ax.set_title(f'{title}\nFunctor Profile (ICC: {icc_result.icc_class})',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    filename = f'{title.replace(" ", "_")}_radar.png'
    plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / filename}")


def plot_phase_space(trajectories: Dict, title: str, output_path: Path):
    """Plot pacing vs arousal phase space."""
    pacing = trajectories['pacing'].values
    arousal = trajectories['arousal'].values

    # Align lengths
    min_len = min(len(pacing), len(arousal))
    pacing = pacing[:min_len]
    arousal = arousal[:min_len]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Color by narrative time
    colors = np.linspace(0, 1, min_len)
    scatter = ax.scatter(pacing, arousal, c=colors, cmap='viridis', s=30, alpha=0.7)

    # Draw trajectory line
    ax.plot(pacing, arousal, 'k-', alpha=0.2, linewidth=0.5)

    # Mark start and end
    ax.scatter(pacing[0], arousal[0], c='green', s=200, marker='^', label='Start', zorder=5)
    ax.scatter(pacing[-1], arousal[-1], c='red', s=200, marker='s', label='End', zorder=5)

    ax.set_xlabel('Pacing', fontsize=12)
    ax.set_ylabel('Arousal', fontsize=12)
    ax.set_title(f'{title}\nPacing-Arousal Phase Space', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add quadrant labels
    ax.text(0.25, 0.75, 'Slow & Tense', fontsize=10, ha='center', alpha=0.5)
    ax.text(0.75, 0.75, 'Fast & Tense', fontsize=10, ha='center', alpha=0.5)
    ax.text(0.25, 0.25, 'Slow & Calm', fontsize=10, ha='center', alpha=0.5)
    ax.text(0.75, 0.25, 'Fast & Calm', fontsize=10, ha='center', alpha=0.5)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Narrative Progress')
    ax.legend(loc='upper right')

    plt.tight_layout()
    filename = f'{title.replace(" ", "_")}_phase_space.png'
    plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / filename}")


def plot_narrative_summary(trajectories: Dict, title: str, icc_result, output_path: Path):
    """Create a comprehensive summary visualization."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Title and metadata
    fig.suptitle(f'{title}\nMulti-Functor Narrative Analysis',
                fontsize=16, fontweight='bold', y=0.98)

    # 1. Main trajectory (sentiment) with ICC zones
    ax1 = fig.add_subplot(gs[0, :2])
    sentiment = smooth_trajectory(trajectories['sentiment'].values)
    time = np.linspace(0, 100, len(sentiment))
    ax1.fill_between(time, sentiment, alpha=0.3, color=FUNCTOR_COLORS['sentiment'])
    ax1.plot(time, sentiment, color=FUNCTOR_COLORS['sentiment'], linewidth=2)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title(f'Sentiment Trajectory (ICC: {icc_result.icc_class})', fontweight='bold')
    ax1.set_xlabel('Narrative Progress (%)')
    ax1.set_ylabel('Sentiment')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1)

    # 2. Pacing and Arousal overlay
    ax2 = fig.add_subplot(gs[0, 2:])
    pacing = smooth_trajectory(trajectories['pacing'].values)
    arousal = smooth_trajectory(trajectories['arousal'].values)
    ax2.plot(time, pacing, color=FUNCTOR_COLORS['pacing'], linewidth=2, label='Pacing')
    ax2.plot(time, arousal, color=FUNCTOR_COLORS['arousal'], linewidth=2, label='Arousal')
    ax2.fill_between(time, pacing, arousal, alpha=0.2, color='gray')
    ax2.set_title('Pacing vs Arousal', fontweight='bold')
    ax2.set_xlabel('Narrative Progress (%)')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1)

    # 3. Entropy and Epistemic
    ax3 = fig.add_subplot(gs[1, :2])
    entropy = smooth_trajectory(trajectories['entropy'].values)
    epistemic = smooth_trajectory(trajectories['epistemic'].values)
    ax3.plot(time, entropy, color=FUNCTOR_COLORS['entropy'], linewidth=2, label='Entropy')
    ax3.plot(time, epistemic, color=FUNCTOR_COLORS['epistemic'], linewidth=2, label='Epistemic')
    ax3.set_title('Entropy vs Epistemic (Information Dynamics)', fontweight='bold')
    ax3.set_xlabel('Narrative Progress (%)')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 1)

    # 4. Character presence
    ax4 = fig.add_subplot(gs[1, 2:])
    character = smooth_trajectory(trajectories['character'].values)
    ax4.fill_between(time, character, alpha=0.4, color=FUNCTOR_COLORS['character'])
    ax4.plot(time, character, color=FUNCTOR_COLORS['character'], linewidth=2)
    ax4.set_title('Character Presence Density', fontweight='bold')
    ax4.set_xlabel('Narrative Progress (%)')
    ax4.set_ylabel('Character Focus')
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 1)

    # 5. Statistics table
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')
    stats_text = "Functor Statistics\n" + "="*30 + "\n"
    for name in trajectories.keys():
        mean = np.mean(trajectories[name].values)
        std = np.std(trajectories[name].values)
        stats_text += f"{name.capitalize():12} μ={mean:.3f} σ={std:.3f}\n"
    ax5.text(0.1, 0.9, stats_text, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax5.transAxes)

    # 6. ICC details
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    icc_text = f"ICC Classification\n" + "="*30 + "\n"
    icc_text += f"Class: {icc_result.icc_class}\n"
    icc_text += f"Name: {icc_result.class_name}\n"
    icc_text += f"Confidence: {icc_result.confidence:.2%}\n"
    icc_text += f"Culture: {icc_result.cultural_prediction}\n"
    ax6.text(0.1, 0.9, icc_text, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax6.transAxes)

    # 7. Correlation highlights
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    # Compute key correlations
    corrs = []
    pairs = [('sentiment', 'arousal'), ('pacing', 'arousal'), ('entropy', 'epistemic')]
    corr_text = "Key Correlations\n" + "="*30 + "\n"
    for n1, n2 in pairs:
        v1 = trajectories[n1].values
        v2 = trajectories[n2].values
        min_len = min(len(v1), len(v2))
        if min_len > 2:
            corr, _ = pearsonr(v1[:min_len], v2[:min_len])
            corr_text += f"{n1[:4]}-{n2[:4]}: {corr:+.3f}\n"

    ax7.text(0.1, 0.9, corr_text, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax7.transAxes)

    # 8. Voice/POV info
    ax8 = fig.add_subplot(gs[2, 3])
    ax8.axis('off')
    voice_meta = trajectories['voice'].metadata
    voice_text = "Narrative Voice\n" + "="*30 + "\n"
    voice_text += f"POV: {voice_meta.get('dominant_pov', 'unknown')}\n"
    voice_text += f"POV Shifts: {voice_meta.get('pov_shifts', 0)}\n"
    voice_text += f"Mean Distance: {np.mean(trajectories['voice'].values):.3f}\n"

    char_meta = trajectories['character'].metadata
    voice_text += f"\nProtagonist Dom: {char_meta.get('protagonist_dominance', 0):.2%}\n"
    voice_text += f"Unique Chars: {char_meta.get('total_unique_characters', 0)}\n"

    ax8.text(0.1, 0.9, voice_text, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax8.transAxes)

    filename = f'{title.replace(" ", "_")}_summary.png'
    plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / filename}")


def main():
    parser = argparse.ArgumentParser(description="Multi-functor narrative visualization")
    parser.add_argument('-i', '--input', required=True, help="Input JSON file with text")
    parser.add_argument('-o', '--output', default='visualizations', help="Output directory")
    parser.add_argument('--window-size', type=int, default=1000, help="Window size")
    parser.add_argument('--overlap', type=int, default=500, help="Window overlap")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load text
    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        if input_path.suffix == '.json':
            data = json.load(f)
            text = data.get('text', '')
            title = data.get('title', input_path.stem)
        else:
            text = f.read()
            title = input_path.stem

    print(f"Title: {title}")
    print(f"Text length: {len(text.split())} words")

    # Get all trajectories
    trajectories, icc_result = get_all_trajectories(text, args.window_size, args.overlap)

    print(f"\nICC Classification: {icc_result.icc_class} - {icc_result.class_name}")
    print(f"Confidence: {icc_result.confidence:.2%}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_multi_trajectory(trajectories, title, icc_result, output_path)
    plot_correlation_heatmap(trajectories, title, output_path)
    plot_radar_chart(trajectories, title, icc_result, output_path)
    plot_phase_space(trajectories, title, output_path)
    plot_narrative_summary(trajectories, title, icc_result, output_path)

    print(f"\nAll visualizations saved to {output_path}")


if __name__ == "__main__":
    main()
