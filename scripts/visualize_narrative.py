#!/usr/bin/env python3
"""
Comprehensive narrative visualization suite.

Creates multiple visualizations for narrative analysis:
1. Sentiment trajectory with ICC classification
2. ICC feature space (radar chart)
3. 3D narrative manifold (sentiment × entropy × time)
4. Phase space portrait (velocity vs position)
5. Category Narr morphism diagram
6. Information geometry (curvature analysis)
7. Comparative ICC class visualization

Usage:
    python scripts/visualize_narrative.py --gutenberg 1399 --output-dir ./visualizations
"""

import argparse
import sys
from pathlib import Path
import urllib.request

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.monospace'] = ['JetBrains Mono', 'Fira Code', 'SF Mono', 'Monaco', 'Consolas', 'monospace']


def download_gutenberg(book_id: int) -> tuple:
    """Download a book from Project Gutenberg."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    print(f"Downloading from {url}...")

    with urllib.request.urlopen(url) as response:
        text = response.read().decode('utf-8-sig')

    # Extract title
    title = f"Gutenberg #{book_id}"
    for line in text[:5000].split('\n'):
        if line.startswith('Title:'):
            title = line.replace('Title:', '').strip()
            break

    # Remove header/footer
    for marker in ["*** START OF", "***START OF"]:
        if marker in text:
            text = text.split(marker, 1)[1].split('\n', 1)[1]
            break
    for marker in ["*** END OF", "***END OF"]:
        if marker in text:
            text = text.split(marker, 1)[0]
            break

    return text.strip(), title


def get_trajectories(text: str):
    """Extract multiple trajectories from text."""
    from src.functors.sentiment import SentimentFunctor
    from src.functors.entropy import EntropyFunctor

    print("Extracting sentiment trajectory...")
    sentiment_functor = SentimentFunctor()
    sentiment_traj = sentiment_functor.process_text(text)

    print("Extracting entropy trajectory...")
    entropy_functor = EntropyFunctor()
    entropy_traj = entropy_functor.process_text(text)

    return sentiment_traj, entropy_traj


def get_icc_result(trajectory):
    """Get ICC classification."""
    from src.detectors.icc import ICCDetector
    detector = ICCDetector()
    normalized = trajectory.normalize()
    return detector.detect(normalized.values, trajectory_id="analysis", title="Analysis")


# =============================================================================
# VISUALIZATION 1: Sentiment Trajectory with ICC
# =============================================================================
def plot_sentiment_trajectory(sentiment_traj, icc_result, title, output_dir):
    """Plot sentiment trajectory with ICC classification overlay."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

    values = sentiment_traj.values
    time = sentiment_traj.time_points
    normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)

    # Smooth for visualization
    smooth = gaussian_filter1d(normalized, sigma=3)

    # Main trajectory plot
    ax1 = axes[0]

    # Color gradient based on value
    colors = plt.cm.RdYlGn(smooth)
    for i in range(len(time) - 1):
        ax1.plot(time[i:i+2], smooth[i:i+2], color=colors[i], linewidth=2, alpha=0.8)

    # Add smoothed trend line
    very_smooth = gaussian_filter1d(normalized, sigma=10)
    ax1.plot(time, very_smooth, 'k-', linewidth=3, alpha=0.5, label='Trend')

    # Mark peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(smooth, prominence=0.1)
    ax1.scatter(time[peaks], smooth[peaks], c='red', s=100, zorder=5,
                marker='^', label=f'Peaks ({len(peaks)})')

    # Add ICC classification box
    icc_color = {
        'ICC-0': '#9E9E9E', 'ICC-1': '#4CAF50', 'ICC-2': '#2196F3',
        'ICC-3': '#FF9800', 'ICC-4': '#E91E63', 'ICC-5': '#9C27B0'
    }.get(icc_result.icc_class, '#757575')

    box_text = f"{icc_result.icc_class}: {icc_result.class_name}\n{icc_result.cultural_prediction.upper()}"
    ax1.text(0.02, 0.98, box_text, transform=ax1.transAxes, fontsize=14,
             verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=icc_color, alpha=0.3))

    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel('Narrative Progress', fontsize=12)
    ax1.set_ylabel('Sentiment (normalized)', fontsize=12)
    ax1.set_title(f'{title}\nSentiment Trajectory', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right')

    # Add narrative markers
    for pos, label in [(0.25, 'Act I'), (0.5, 'Midpoint'), (0.75, 'Act III'), (0.9, 'Climax')]:
        ax1.axvline(x=pos, color='gray', linestyle='--', alpha=0.3)
        ax1.text(pos, 1.02, label, ha='center', fontsize=10, alpha=0.7)

    # Feature bar chart
    ax2 = axes[1]
    features = icc_result.features
    feature_names = ['Net Change', 'Volatility', 'Trend R²', 'Symmetry', 'Structure']
    feature_values = [
        features['net_change'],
        features['volatility'] * 10,  # Scale for visibility
        features['trend_r2'],
        features['symmetry'],
        features['structure_score']
    ]

    colors = ['#2196F3' if v > 0 else '#F44336' for v in feature_values]
    bars = ax2.bar(feature_names, feature_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Feature Value', fontsize=12)
    ax2.set_title('ICC Features', fontsize=12)

    # Add value labels
    for bar, val in zip(bars, feature_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'sentiment_trajectory.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# VISUALIZATION 2: ICC Feature Radar Chart
# =============================================================================
def plot_icc_radar(icc_result, title, output_dir):
    """Plot ICC features as radar chart."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    features = icc_result.features

    # Select features for radar
    categories = ['Net Change', 'Peak Count', 'Volatility', 'Trend R²',
                  'Symmetry', 'Structure', 'Autocorr.']

    # Normalize values to 0-1 scale
    raw_values = [
        (features['net_change'] + 1) / 2,  # -1 to 1 → 0 to 1
        min(features['n_peaks'] / 10, 1),  # Cap at 10 peaks
        min(features['volatility'] / 0.15, 1),  # Cap at 0.15
        features['trend_r2'],
        features['symmetry'],
        features['structure_score'],
        (features['autocorrelation'] + 1) / 2,  # -1 to 1 → 0 to 1
    ]

    # Number of variables
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values = raw_values + [raw_values[0]]  # Close the polygon
    angles += angles[:1]

    # Plot
    ax.plot(angles, values, 'o-', linewidth=2, color='#2196F3', markersize=8)
    ax.fill(angles, values, alpha=0.25, color='#2196F3')

    # Add reference circles
    for r in [0.25, 0.5, 0.75, 1.0]:
        circle_angles = np.linspace(0, 2*np.pi, 100)
        ax.plot(circle_angles, [r]*100, '--', color='gray', alpha=0.3, linewidth=0.5)

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)

    # Title with ICC class
    icc_color = {
        'ICC-0': '#9E9E9E', 'ICC-1': '#4CAF50', 'ICC-2': '#2196F3',
        'ICC-3': '#FF9800', 'ICC-4': '#E91E63', 'ICC-5': '#9C27B0'
    }.get(icc_result.icc_class, '#757575')

    ax.set_title(f'{title}\nICC Feature Profile: {icc_result.icc_class} ({icc_result.class_name})',
                 fontsize=14, fontweight='bold', pad=20, color=icc_color)

    output_path = output_dir / 'icc_radar.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# VISUALIZATION 3: 3D Narrative Manifold (Enhanced with Chapter Labels)
# =============================================================================
def extract_chapters(text: str) -> list:
    """Extract chapter markers from text."""
    import re

    chapters = []

    # Common chapter patterns
    patterns = [
        r'PART\s+([IVXLC]+|\d+)',
        r'Part\s+([IVXLC]+|\d+)',
        r'CHAPTER\s+([IVXLC]+|\d+)',
        r'Chapter\s+([IVXLC]+|\d+)',
        r'BOOK\s+([IVXLC]+|\d+)',
        r'Book\s+([IVXLC]+|\d+)',
        r'^([IVXLC]+)\.\s*$',  # Roman numerals alone on a line
        r'^(\d+)\.\s*$',  # Numbers alone
    ]

    lines = text.split('\n')
    total_chars = len(text)
    current_pos = 0

    for i, line in enumerate(lines):
        for pattern in patterns:
            match = re.search(pattern, line.strip())
            if match:
                # Calculate position as fraction of total
                position = current_pos / total_chars
                chapter_id = match.group(1) if match.groups() else str(len(chapters) + 1)

                # Determine chapter type
                if 'PART' in line.upper() or 'Part' in line:
                    ch_type = 'Part'
                elif 'BOOK' in line.upper() or 'Book' in line:
                    ch_type = 'Book'
                else:
                    ch_type = 'Ch'

                chapters.append({
                    'position': position,
                    'label': f"{ch_type} {chapter_id}",
                    'line': line.strip()[:50]
                })
                break
        current_pos += len(line) + 1

    # Remove duplicates close together
    filtered = []
    for ch in chapters:
        if not filtered or abs(ch['position'] - filtered[-1]['position']) > 0.02:
            filtered.append(ch)

    return filtered


def plot_3d_manifold(sentiment_traj, entropy_traj, title, output_dir, text=None):
    """Plot 3D narrative manifold (sentiment × entropy × time) with chapter labels."""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Resample to same length
    n_points = min(len(sentiment_traj.values), len(entropy_traj.values), 200)
    time = np.linspace(0, 1, n_points)

    sentiment = np.interp(time, sentiment_traj.time_points, sentiment_traj.values)
    entropy = np.interp(time, entropy_traj.time_points, entropy_traj.values)

    # Normalize
    sentiment = (sentiment - sentiment.min()) / (sentiment.max() - sentiment.min() + 1e-8)
    entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)

    # Smooth
    sentiment = gaussian_filter1d(sentiment, sigma=2)
    entropy = gaussian_filter1d(entropy, sigma=2)

    # Color by time
    colors = plt.cm.viridis(time)

    # Plot 3D trajectory
    for i in range(len(time) - 1):
        ax.plot3D(time[i:i+2], sentiment[i:i+2], entropy[i:i+2],
                  color=colors[i], linewidth=2, alpha=0.8)

    # Add projections (shadows)
    ax.plot(time, sentiment, zs=0, zdir='z', color='blue', alpha=0.2, linewidth=1)
    ax.plot(time, entropy, zs=0, zdir='y', color='green', alpha=0.2, linewidth=1)
    ax.plot(sentiment, entropy, zs=0, zdir='x', color='red', alpha=0.2, linewidth=1)

    # Mark start and end
    ax.scatter([0], [sentiment[0]], [entropy[0]], c='green', s=200, marker='o',
               label='Start', zorder=5)
    ax.scatter([1], [sentiment[-1]], [entropy[-1]], c='red', s=200, marker='s',
               label='End', zorder=5)

    # Find peaks in sentiment
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(sentiment, prominence=0.08, distance=10)

    # Extract chapters if text provided
    chapters = []
    if text:
        chapters = extract_chapters(text)

    # Mark peaks with labels
    peak_colors = plt.cm.Set1(np.linspace(0, 1, max(len(peaks), 1)))

    if len(peaks) > 0:
        ax.scatter(time[peaks], sentiment[peaks], entropy[peaks],
                   c='orange', s=150, marker='^', edgecolors='black', linewidths=1,
                   label=f'Peaks ({len(peaks)})', zorder=5)

        # Add labels for each peak
        for i, peak_idx in enumerate(peaks):
            peak_time = time[peak_idx]
            peak_sent = sentiment[peak_idx]
            peak_ent = entropy[peak_idx]

            # Find nearest chapter
            peak_label = f"Peak {i+1}\n({peak_time:.0%})"
            if chapters:
                # Find closest chapter before this peak
                nearest_ch = None
                for ch in chapters:
                    if ch['position'] <= peak_time:
                        nearest_ch = ch
                    else:
                        break
                if nearest_ch:
                    peak_label = f"{nearest_ch['label']}\n({peak_time:.0%})"

            # Add 3D text annotation
            ax.text(peak_time, peak_sent + 0.1, peak_ent + 0.05,
                    peak_label, fontsize=8, ha='center', va='bottom',
                    color=peak_colors[i % len(peak_colors)], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # Add chapter markers on the time axis
    if chapters and len(chapters) <= 20:  # Only show if not too many
        for ch in chapters:
            pos = ch['position']
            if 0 < pos < 1:
                # Find sentiment and entropy at this position
                idx = int(pos * (len(time) - 1))
                ch_sent = sentiment[idx]
                ch_ent = entropy[idx]

                # Vertical line from bottom
                ax.plot([pos, pos], [0, ch_sent], [0, 0],
                        color='gray', linestyle=':', alpha=0.5, linewidth=1)

    ax.set_xlabel('Narrative Time', fontsize=12, labelpad=10)
    ax.set_ylabel('Sentiment', fontsize=12, labelpad=10)
    ax.set_zlabel('Entropy', fontsize=12, labelpad=10)
    ax.set_title(f'{title}\n3D Narrative Manifold with Peak Labels', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    # Better viewing angle
    ax.view_init(elev=25, azim=45)

    output_path = output_dir / 'manifold_3d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    # Also create a detailed peak legend view
    fig2 = plt.figure(figsize=(18, 12))

    # Main 3D plot
    ax2 = fig2.add_subplot(121, projection='3d')

    for i in range(len(time) - 1):
        ax2.plot3D(time[i:i+2], sentiment[i:i+2], entropy[i:i+2],
                   color=colors[i], linewidth=2, alpha=0.8)

    ax2.scatter([0], [sentiment[0]], [entropy[0]], c='green', s=200, marker='o', label='Start')
    ax2.scatter([1], [sentiment[-1]], [entropy[-1]], c='red', s=200, marker='s', label='End')

    # Add peaks with labels
    if len(peaks) > 0:
        ax2.scatter(time[peaks], sentiment[peaks], entropy[peaks],
                    c='orange', s=150, marker='^', edgecolors='black', linewidths=1)

        for i, peak_idx in enumerate(peaks):
            ax2.text(time[peak_idx], sentiment[peak_idx] + 0.08, entropy[peak_idx] + 0.03,
                     f"P{i+1}", fontsize=9, ha='center', fontweight='bold',
                     color=peak_colors[i % len(peak_colors)])

    ax2.set_xlabel('Narrative Time', fontsize=11, labelpad=8)
    ax2.set_ylabel('Sentiment', fontsize=11, labelpad=8)
    ax2.set_zlabel('Entropy', fontsize=11, labelpad=8)
    ax2.set_title(f'{title}\n3D Manifold with Peak Markers', fontsize=14, fontweight='bold')
    ax2.view_init(elev=30, azim=60)
    ax2.legend(loc='upper left')

    # Peak legend table
    ax3 = fig2.add_subplot(122)
    ax3.axis('off')

    # Create legend table
    table_data = [['Peak', 'Position', 'Chapter', 'Sentiment', 'Entropy']]

    for i, peak_idx in enumerate(peaks):
        peak_time = time[peak_idx]
        peak_sent = sentiment[peak_idx]
        peak_ent = entropy[peak_idx]

        # Find chapter
        ch_label = "—"
        if chapters:
            for ch in reversed(chapters):
                if ch['position'] <= peak_time:
                    ch_label = ch['label']
                    break

        table_data.append([
            f"P{i+1}",
            f"{peak_time:.1%}",
            ch_label,
            f"{peak_sent:.3f}",
            f"{peak_ent:.3f}"
        ])

    # Add start/end
    table_data.append(['Start', '0%', chapters[0]['label'] if chapters else '—',
                       f"{sentiment[0]:.3f}", f"{entropy[0]:.3f}"])
    table_data.append(['End', '100%', chapters[-1]['label'] if chapters else '—',
                       f"{sentiment[-1]:.3f}", f"{entropy[-1]:.3f}"])

    table = ax3.table(
        cellText=table_data,
        colLabels=None,
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.15, 0.25, 0.18, 0.18]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header row
    for j in range(5):
        table[(0, j)].set_facecolor('#4CAF50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Color peak rows
    for i in range(1, len(peaks) + 1):
        color = peak_colors[(i-1) % len(peak_colors)]
        table[(i, 0)].set_facecolor(color)
        table[(i, 0)].set_text_props(color='white', fontweight='bold')

    ax3.set_title('Peak Legend with Chapter Locations', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path2 = output_dir / 'manifold_3d_peaks.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path2}")

    # Top view
    fig3 = plt.figure(figsize=(14, 10))
    ax3 = fig3.add_subplot(111, projection='3d')

    for i in range(len(time) - 1):
        ax3.plot3D(time[i:i+2], sentiment[i:i+2], entropy[i:i+2],
                   color=colors[i], linewidth=2, alpha=0.8)

    ax3.scatter([0], [sentiment[0]], [entropy[0]], c='green', s=200, marker='o')
    ax3.scatter([1], [sentiment[-1]], [entropy[-1]], c='red', s=200, marker='s')

    ax3.set_xlabel('Narrative Time', fontsize=12, labelpad=10)
    ax3.set_ylabel('Sentiment', fontsize=12, labelpad=10)
    ax3.set_zlabel('Entropy', fontsize=12, labelpad=10)
    ax3.set_title(f'{title}\n3D Narrative Manifold (Top View)', fontsize=16, fontweight='bold')
    ax3.view_init(elev=60, azim=135)

    output_path3 = output_dir / 'manifold_3d_top.png'
    plt.savefig(output_path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path3}")


# =============================================================================
# VISUALIZATION 4: Phase Space Portrait
# =============================================================================
def plot_phase_space(sentiment_traj, title, output_dir):
    """Plot phase space (position vs velocity)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    values = sentiment_traj.values
    normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)
    smooth = gaussian_filter1d(normalized, sigma=3)

    # Compute velocity (first derivative)
    velocity = np.gradient(smooth)

    # Compute acceleration (second derivative)
    acceleration = np.gradient(velocity)

    time = np.linspace(0, 1, len(smooth))

    # Phase portrait: position vs velocity
    ax1 = axes[0]
    colors = plt.cm.viridis(time)

    for i in range(len(smooth) - 1):
        ax1.plot(smooth[i:i+2], velocity[i:i+2], color=colors[i], linewidth=1.5, alpha=0.7)

    ax1.scatter([smooth[0]], [velocity[0]], c='green', s=200, marker='o',
                label='Start', zorder=5)
    ax1.scatter([smooth[-1]], [velocity[-1]], c='red', s=200, marker='s',
                label='End', zorder=5)

    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    ax1.set_xlabel('Sentiment (position)', fontsize=12)
    ax1.set_ylabel('Rate of Change (velocity)', fontsize=12)
    ax1.set_title('Phase Portrait: Sentiment Dynamics', fontsize=14, fontweight='bold')
    ax1.legend()

    # Add quadrant labels
    ax1.text(0.75, 0.05, 'Rising\nPositive', ha='center', fontsize=10, alpha=0.5)
    ax1.text(0.25, 0.05, 'Rising\nNegative', ha='center', fontsize=10, alpha=0.5)
    ax1.text(0.75, -0.05, 'Falling\nPositive', ha='center', fontsize=10, alpha=0.5)
    ax1.text(0.25, -0.05, 'Falling\nNegative', ha='center', fontsize=10, alpha=0.5)

    # Velocity vs Acceleration
    ax2 = axes[1]
    for i in range(len(velocity) - 1):
        ax2.plot(velocity[i:i+2], acceleration[i:i+2], color=colors[i], linewidth=1.5, alpha=0.7)

    ax2.scatter([velocity[0]], [acceleration[0]], c='green', s=200, marker='o',
                label='Start', zorder=5)
    ax2.scatter([velocity[-1]], [acceleration[-1]], c='red', s=200, marker='s',
                label='End', zorder=5)

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Velocity (rate of change)', fontsize=12)
    ax2.set_ylabel('Acceleration', fontsize=12)
    ax2.set_title('Phase Portrait: Momentum Dynamics', fontsize=14, fontweight='bold')
    ax2.legend()

    plt.suptitle(f'{title}\nPhase Space Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / 'phase_space.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# VISUALIZATION 5: Category Narr Morphism Diagram
# =============================================================================
def plot_category_diagram(text, title, output_dir):
    """Plot Category Narr structure as morphism diagram."""
    from src.categories import CategoryNarr

    category = CategoryNarr.from_text(
        text=text,
        narrative_id="analysis",
        title=title,
        n_states=8
    )

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-1, 2)
    ax.axis('off')

    states = sorted(category.objects, key=lambda s: s.position)
    arc = category.narrative_arc()

    # Draw states as nodes
    node_positions = {}
    for i, state in enumerate(states):
        x = i
        y = 0.5
        node_positions[state.state_id] = (x, y)

        # Node circle
        circle = plt.Circle((x, y), 0.3, color='#2196F3', alpha=0.7, ec='black', linewidth=2)
        ax.add_patch(circle)

        # State label
        ax.text(x, y, f'S{i}', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

        # Position label
        ax.text(x, y - 0.5, f'{state.position:.0%}', ha='center', fontsize=10, alpha=0.7)

    # Draw morphisms as arrows
    morphism_colors = {
        'identity': '#9E9E9E',
        'transition': '#4CAF50',
        'complication': '#FF9800',
        'escalation': '#F44336',
        'climax': '#E91E63',
        'resolution': '#2196F3',
        'denouement': '#9C27B0',
        'revelation': '#00BCD4',
        'reversal': '#FF5722',
        'reflection': '#607D8B',
    }

    for i, morphism in enumerate(arc):
        src_pos = node_positions[morphism.source.state_id]
        tgt_pos = node_positions[morphism.target.state_id]

        # Curved arrow
        color = morphism_colors.get(morphism.morphism_type.value, '#757575')

        # Calculate control point for curve
        mid_x = (src_pos[0] + tgt_pos[0]) / 2
        mid_y = src_pos[1] + 0.5 + morphism.intensity * 0.5

        arrow = FancyArrowPatch(
            (src_pos[0] + 0.3, src_pos[1]),
            (tgt_pos[0] - 0.3, tgt_pos[1]),
            connectionstyle=f"arc3,rad=0.3",
            arrowstyle='->,head_length=10,head_width=6',
            color=color,
            linewidth=2 + morphism.intensity * 3,
            alpha=0.8
        )
        ax.add_patch(arrow)

        # Morphism label
        ax.text(mid_x, mid_y, morphism.morphism_type.value[:4],
                ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

    # Legend
    legend_y = 1.7
    legend_x = 0
    for mtype, color in list(morphism_colors.items())[:6]:
        ax.add_patch(plt.Rectangle((legend_x, legend_y), 0.2, 0.15, color=color, alpha=0.7))
        ax.text(legend_x + 0.25, legend_y + 0.075, mtype, va='center', fontsize=9)
        legend_x += 1.3

    ax.set_title(f'{title}\nCategory Narr: Objects and Morphisms', fontsize=16, fontweight='bold', pad=20)

    # Add composition example
    ax.text(0, -0.8, 'Composition: S0 → S1 → S2 ≡ S0 → S2 (morphisms compose)',
            fontsize=11, style='italic', alpha=0.7)

    output_path = output_dir / 'category_diagram.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# VISUALIZATION 6: Information Geometry / Curvature
# =============================================================================
def plot_information_geometry(sentiment_traj, entropy_traj, title, output_dir):
    """Plot information geometry: curvature and divergence."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Resample
    n_points = min(len(sentiment_traj.values), len(entropy_traj.values), 200)
    time = np.linspace(0, 1, n_points)

    sentiment = np.interp(time, sentiment_traj.time_points, sentiment_traj.values)
    entropy = np.interp(time, entropy_traj.time_points, entropy_traj.values)

    # Normalize
    sentiment = (sentiment - sentiment.min()) / (sentiment.max() - sentiment.min() + 1e-8)
    entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)

    # Smooth
    sentiment = gaussian_filter1d(sentiment, sigma=2)
    entropy = gaussian_filter1d(entropy, sigma=2)

    # Compute curvature
    dx = np.gradient(sentiment)
    dy = np.gradient(entropy)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8)**1.5
    curvature = gaussian_filter1d(curvature, sigma=3)

    # 1. Curvature over time
    ax1 = axes[0, 0]
    ax1.fill_between(time, curvature, alpha=0.3, color='#E91E63')
    ax1.plot(time, curvature, color='#E91E63', linewidth=2)
    ax1.set_xlabel('Narrative Time', fontsize=12)
    ax1.set_ylabel('Curvature', fontsize=12)
    ax1.set_title('Narrative Curvature (geometric complexity)', fontsize=12, fontweight='bold')

    # Mark high curvature points
    threshold = np.percentile(curvature, 90)
    high_curv = curvature > threshold
    ax1.fill_between(time, 0, curvature, where=high_curv, alpha=0.5, color='red', label='High curvature')
    ax1.legend()

    # 2. Local divergence (KL-like measure)
    ax2 = axes[0, 1]

    # Compute local "divergence" as change in distribution
    window = 10
    divergence = []
    for i in range(len(sentiment) - window):
        p = sentiment[i:i+window]
        q = sentiment[i+1:i+window+1]
        # Simple divergence proxy
        p_norm = (p - p.min()) / (p.max() - p.min() + 1e-8) + 1e-8
        q_norm = (q - q.min()) / (q.max() - q.min() + 1e-8) + 1e-8
        p_norm = p_norm / p_norm.sum()
        q_norm = q_norm / q_norm.sum()
        kl = np.sum(p_norm * np.log(p_norm / q_norm + 1e-8))
        divergence.append(abs(kl))

    divergence = np.array(divergence)
    divergence = gaussian_filter1d(divergence, sigma=2)
    div_time = time[:len(divergence)]

    ax2.fill_between(div_time, divergence, alpha=0.3, color='#9C27B0')
    ax2.plot(div_time, divergence, color='#9C27B0', linewidth=2)
    ax2.set_xlabel('Narrative Time', fontsize=12)
    ax2.set_ylabel('Local Divergence', fontsize=12)
    ax2.set_title('Information Divergence (narrative surprise)', fontsize=12, fontweight='bold')

    # 3. Trajectory in feature space with curvature coloring
    ax3 = axes[1, 0]

    # Normalize curvature for coloring
    curv_colors = plt.cm.hot(curvature / curvature.max())

    for i in range(len(sentiment) - 1):
        ax3.plot(sentiment[i:i+2], entropy[i:i+2], color=curv_colors[i], linewidth=2, alpha=0.8)

    ax3.scatter([sentiment[0]], [entropy[0]], c='green', s=200, marker='o', label='Start', zorder=5)
    ax3.scatter([sentiment[-1]], [entropy[-1]], c='blue', s=200, marker='s', label='End', zorder=5)

    ax3.set_xlabel('Sentiment', fontsize=12)
    ax3.set_ylabel('Entropy', fontsize=12)
    ax3.set_title('Feature Space (color = curvature)', fontsize=12, fontweight='bold')
    ax3.legend()

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(0, curvature.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax3, label='Curvature')

    # 4. Cumulative arc length (narrative "distance traveled")
    ax4 = axes[1, 1]

    ds = np.sqrt(np.diff(sentiment)**2 + np.diff(entropy)**2)
    arc_length = np.concatenate([[0], np.cumsum(ds)])

    ax4.plot(time, arc_length, color='#00BCD4', linewidth=3)
    ax4.fill_between(time, arc_length, alpha=0.2, color='#00BCD4')

    ax4.set_xlabel('Narrative Time', fontsize=12)
    ax4.set_ylabel('Cumulative Arc Length', fontsize=12)
    ax4.set_title('Narrative Distance Traveled', fontsize=12, fontweight='bold')

    # Add reference line for "direct path"
    direct_distance = np.sqrt((sentiment[-1] - sentiment[0])**2 + (entropy[-1] - entropy[0])**2)
    ax4.axhline(y=direct_distance, color='red', linestyle='--', alpha=0.5, label=f'Direct: {direct_distance:.2f}')
    ax4.axhline(y=arc_length[-1], color='green', linestyle='--', alpha=0.5, label=f'Actual: {arc_length[-1]:.2f}')
    ax4.legend()

    # Tortuosity
    tortuosity = arc_length[-1] / (direct_distance + 1e-8)
    ax4.text(0.5, arc_length[-1] * 0.5, f'Tortuosity: {tortuosity:.2f}',
             fontsize=12, fontweight='bold', ha='center')

    plt.suptitle(f'{title}\nInformation Geometry Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / 'information_geometry.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# VISUALIZATION 7: ICC Class Comparison
# =============================================================================
def plot_icc_comparison(icc_result, title, output_dir):
    """Plot comparison of this narrative against all ICC classes."""
    from src.detectors.icc import ICC_CLASSES

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    features = icc_result.features

    class_colors = {
        'ICC-0': '#9E9E9E', 'ICC-1': '#4CAF50', 'ICC-2': '#2196F3',
        'ICC-3': '#FF9800', 'ICC-4': '#E91E63', 'ICC-5': '#9C27B0'
    }

    for idx, (class_id, class_def) in enumerate(ICC_CLASSES.items()):
        ax = axes[idx]

        thresholds = class_def.get("thresholds", {})
        color = class_colors.get(class_id, '#757575')
        is_match = (class_id == icc_result.icc_class)

        # Create mini bar chart showing feature vs threshold
        feature_checks = []
        labels = []

        if "net_change_min" in thresholds:
            val = features['net_change']
            thresh_min = thresholds.get('net_change_min', -1)
            thresh_max = thresholds.get('net_change_max', 1)
            in_range = thresh_min <= val <= thresh_max
            feature_checks.append((val, thresh_min, thresh_max, in_range))
            labels.append('Net Δ')

        if "peaks_max" in thresholds:
            val = features['n_peaks']
            thresh = thresholds['peaks_max']
            ok = val <= thresh
            feature_checks.append((val/10, 0, thresh/10, ok))  # Normalize
            labels.append('Peaks')
        elif "peaks_min" in thresholds:
            val = features['n_peaks']
            thresh = thresholds['peaks_min']
            ok = val >= thresh
            feature_checks.append((val/10, thresh/10, 1, ok))
            labels.append('Peaks')

        if "volatility_max" in thresholds:
            val = features['volatility']
            thresh = thresholds['volatility_max']
            ok = val <= thresh
            feature_checks.append((val, 0, thresh, ok))
            labels.append('Volat.')
        elif "volatility_min" in thresholds:
            val = features['volatility']
            thresh = thresholds['volatility_min']
            ok = val >= thresh
            feature_checks.append((val, thresh, 0.2, ok))
            labels.append('Volat.')

        if "trend_r2_min" in thresholds:
            val = features['trend_r2']
            thresh = thresholds['trend_r2_min']
            ok = val >= thresh
            feature_checks.append((val, thresh, 1, ok))
            labels.append('Trend')

        if "symmetry_min" in thresholds:
            val = features['symmetry']
            thresh = thresholds['symmetry_min']
            ok = val >= thresh
            feature_checks.append((val, thresh, 1, ok))
            labels.append('Symm.')

        # Plot
        x = np.arange(len(labels))
        values = [fc[0] for fc in feature_checks]
        oks = [fc[3] for fc in feature_checks]

        bar_colors = ['green' if ok else 'red' for ok in oks]
        bars = ax.bar(x, values, color=bar_colors, alpha=0.7, edgecolor='black')

        # Add threshold markers
        for i, (val, tmin, tmax, ok) in enumerate(feature_checks):
            if tmin != 0:
                ax.axhline(y=tmin, xmin=(i-0.4+0.5)/len(labels), xmax=(i+0.4+0.5)/len(labels),
                          color='blue', linestyle='--', linewidth=2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(-0.5, 1.2)

        # Title
        match_str = "✓ MATCH" if is_match else ""
        ax.set_title(f'{class_id}: {class_def["name"]}\n{match_str}',
                     fontsize=12, fontweight='bold', color=color if is_match else 'black')

        if is_match:
            ax.patch.set_facecolor(color)
            ax.patch.set_alpha(0.1)

    plt.suptitle(f'{title}\nICC Class Matching Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / 'icc_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# VISUALIZATION 8: Peak Distribution Area Graph
# =============================================================================
def plot_peak_distribution(sentiment_traj, title, output_dir):
    """Plot peak distribution as area graph showing emotional intensity zones."""
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    values = sentiment_traj.values
    time = sentiment_traj.time_points

    # Normalize to 0-1
    values_norm = (values - values.min()) / (values.max() - values.min() + 1e-8)

    # Smooth for peak detection
    smooth = gaussian_filter1d(values_norm, sigma=3)

    # Find peaks and valleys
    peaks, peak_props = find_peaks(smooth, prominence=0.05, distance=5)
    valleys, valley_props = find_peaks(-smooth, prominence=0.05, distance=5)

    # =========================================================================
    # 1. Stacked Area: Positive vs Negative Emotional Energy
    # =========================================================================
    ax1 = axes[0, 0]

    # Center around 0.5 (neutral)
    centered = values_norm - 0.5
    positive = np.maximum(centered, 0)
    negative = np.abs(np.minimum(centered, 0))

    ax1.fill_between(time, 0, positive, alpha=0.6, color='#4CAF50', label='Positive emotion')
    ax1.fill_between(time, 0, -negative, alpha=0.6, color='#F44336', label='Negative emotion')
    ax1.plot(time, centered, color='black', linewidth=1, alpha=0.5)

    # Mark peaks on positive side
    for peak_idx in peaks:
        if centered[peak_idx] > 0:
            ax1.axvline(x=time[peak_idx], color='green', linestyle=':', alpha=0.5)
            ax1.scatter([time[peak_idx]], [centered[peak_idx]], c='green', s=100, zorder=5, marker='^')

    # Mark valleys on negative side
    for valley_idx in valleys:
        if centered[valley_idx] < 0:
            ax1.axvline(x=time[valley_idx], color='red', linestyle=':', alpha=0.5)
            ax1.scatter([time[valley_idx]], [centered[valley_idx]], c='red', s=100, zorder=5, marker='v')

    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=2)
    ax1.set_xlabel('Narrative Time', fontsize=12)
    ax1.set_ylabel('Emotional Valence', fontsize=12)
    ax1.set_title('Emotional Energy Distribution', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, 1)

    # =========================================================================
    # 2. Peak Density Heatmap
    # =========================================================================
    ax2 = axes[0, 1]

    # Create density estimate of peaks
    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)

    # Count peaks and valleys in each bin
    peak_counts = np.zeros(n_bins)
    valley_counts = np.zeros(n_bins)
    intensity_sum = np.zeros(n_bins)

    for peak_idx in peaks:
        t = time[peak_idx]
        bin_idx = min(int(t * n_bins), n_bins - 1)
        peak_counts[bin_idx] += 1
        intensity_sum[bin_idx] += smooth[peak_idx]

    for valley_idx in valleys:
        t = time[valley_idx]
        bin_idx = min(int(t * n_bins), n_bins - 1)
        valley_counts[bin_idx] += 1

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = 1 / n_bins * 0.8

    # Stacked bar chart
    ax2.bar(bin_centers, peak_counts, width=width, color='#4CAF50', alpha=0.7, label=f'Peaks ({len(peaks)})')
    ax2.bar(bin_centers, -valley_counts, width=width, color='#F44336', alpha=0.7, label=f'Valleys ({len(valleys)})')

    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_xlabel('Narrative Time', fontsize=12)
    ax2.set_ylabel('Count (peaks ↑ / valleys ↓)', fontsize=12)
    ax2.set_title('Peak & Valley Distribution by Narrative Section', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 1)

    # Add section labels
    sections = ['Exposition', 'Rising Action', 'Midpoint', 'Climax', 'Resolution']
    for i, section in enumerate(sections):
        x_pos = (i + 0.5) / 5
        ax2.text(x_pos, ax2.get_ylim()[1] * 0.9, section, ha='center', fontsize=9, alpha=0.6)

    # =========================================================================
    # 3. Cumulative Peak Area (Running Emotional Intensity)
    # =========================================================================
    ax3 = axes[1, 0]

    # Compute running emotional intensity
    window_size = max(len(values) // 20, 5)
    intensity = np.zeros(len(values))

    for i in range(len(values)):
        start = max(0, i - window_size)
        end = min(len(values), i + window_size)
        intensity[i] = np.std(values_norm[start:end])  # Local volatility as intensity

    # Smooth the intensity
    intensity_smooth = gaussian_filter1d(intensity, sigma=5)

    # Create gradient fill
    colors = plt.cm.YlOrRd(intensity_smooth / intensity_smooth.max())

    for i in range(len(time) - 1):
        ax3.fill_between(time[i:i+2], 0, intensity_smooth[i:i+2],
                         color=colors[i], alpha=0.8)

    ax3.plot(time, intensity_smooth, color='darkred', linewidth=2)

    # Mark peak positions
    for peak_idx in peaks:
        ax3.axvline(x=time[peak_idx], color='black', linestyle=':', alpha=0.3)

    ax3.set_xlabel('Narrative Time', fontsize=12)
    ax3.set_ylabel('Emotional Intensity (local volatility)', fontsize=12)
    ax3.set_title('Emotional Intensity Heatmap', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 1)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(0, intensity_smooth.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax3, label='Intensity', shrink=0.8)

    # =========================================================================
    # 4. Peak Timeline with Prominence
    # =========================================================================
    ax4 = axes[1, 1]

    # Plot the smoothed trajectory
    ax4.fill_between(time, 0, smooth, alpha=0.2, color='#2196F3')
    ax4.plot(time, smooth, color='#2196F3', linewidth=2, label='Sentiment')

    # Mark peaks with size proportional to prominence
    if len(peaks) > 0 and 'prominences' in peak_props:
        prominences = peak_props['prominences']
        prom_norm = prominences / prominences.max() * 300 + 50  # Scale for marker size

        scatter = ax4.scatter(time[peaks], smooth[peaks],
                              s=prom_norm, c=prominences, cmap='Reds',
                              edgecolors='black', linewidths=1, zorder=5,
                              label='Peaks (size = prominence)')
        plt.colorbar(scatter, ax=ax4, label='Peak Prominence', shrink=0.8)
    else:
        ax4.scatter(time[peaks], smooth[peaks], s=100, c='red',
                    edgecolors='black', linewidths=1, zorder=5, label='Peaks')

    # Mark valleys
    ax4.scatter(time[valleys], smooth[valleys], s=80, c='blue', marker='v',
                edgecolors='black', linewidths=1, zorder=5, label='Valleys')

    # Annotate significant peaks
    if len(peaks) > 0:
        sorted_peak_indices = np.argsort(smooth[peaks])[::-1][:5]  # Top 5 peaks
        for rank, idx in enumerate(sorted_peak_indices):
            peak_idx = peaks[idx]
            ax4.annotate(f'#{rank+1}',
                         xy=(time[peak_idx], smooth[peak_idx]),
                         xytext=(time[peak_idx], smooth[peak_idx] + 0.1),
                         fontsize=10, fontweight='bold', ha='center',
                         arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))

    ax4.set_xlabel('Narrative Time', fontsize=12)
    ax4.set_ylabel('Sentiment (normalized)', fontsize=12)
    ax4.set_title('Peak Timeline with Prominence Ranking', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.set_xlim(0, 1)

    plt.suptitle(f'{title}\nPeak Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / 'peak_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    # =========================================================================
    # BONUS: Create a summary statistics table
    # =========================================================================
    fig2, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Compute statistics
    peak_positions = time[peaks] if len(peaks) > 0 else []
    valley_positions = time[valleys] if len(valleys) > 0 else []

    # Peak distribution by narrative section (quintiles)
    quintile_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    peak_quintiles = [sum((p >= i/5) & (p < (i+1)/5) for p in peak_positions) for i in range(5)]
    valley_quintiles = [sum((v >= i/5) & (v < (i+1)/5) for v in valley_positions) for i in range(5)]

    table_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['Total Peaks', str(len(peaks)), 'High = dramatic, Low = smooth'],
        ['Total Valleys', str(len(valleys)), 'Emotional low points'],
        ['Peak/Valley Ratio', f'{len(peaks)/(len(valleys)+1):.2f}', '>1 = more highs than lows'],
        ['First Peak Position', f'{peak_positions[0]:.0%}' if len(peak_positions) > 0 else 'N/A', 'Earlier = fast start'],
        ['Last Peak Position', f'{peak_positions[-1]:.0%}' if len(peak_positions) > 0 else 'N/A', 'Later = strong finish'],
        ['Peak Spread (std)', f'{np.std(peak_positions):.2f}' if len(peak_positions) > 1 else 'N/A', 'Higher = distributed'],
        ['', '', ''],
        ['Section', 'Peaks', 'Valleys'],
    ]

    for i, label in enumerate(quintile_labels):
        table_data.append([label, str(peak_quintiles[i]), str(valley_quintiles[i])])

    table = ax.table(
        cellText=table_data,
        colLabels=None,
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.25, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Style header rows
    for j in range(3):
        table[(0, j)].set_facecolor('#2196F3')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
        table[(8, j)].set_facecolor('#4CAF50')
        table[(8, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title(f'{title}\nPeak Distribution Statistics', fontsize=16, fontweight='bold', pad=20)

    output_path2 = output_dir / 'peak_statistics.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path2}")


# =============================================================================
# VISUALIZATION 9: 3D Narrative Terrain (Geographic Peaks)
# =============================================================================
def plot_narrative_terrain(sentiment_traj, entropy_traj, title, output_dir):
    """
    Plot narrative as 3D terrain/topography where emotional intensity creates mountains.

    X-axis: Narrative time (0-100%)
    Y-axis: Emotional breadth (sentiment spread)
    Z-axis: Emotional height (intensity/peaks)
    """
    from scipy.ndimage import gaussian_filter1d, gaussian_filter
    from scipy.signal import find_peaks
    from mpl_toolkits.mplot3d import Axes3D

    # Resample to consistent grid
    n_time = 100  # Resolution along narrative time
    n_breadth = 50  # Resolution for "emotional breadth"

    time = np.linspace(0, 1, n_time)
    sentiment = np.interp(time, sentiment_traj.time_points, sentiment_traj.values)
    entropy = np.interp(time, entropy_traj.time_points, entropy_traj.values)

    # Normalize
    sentiment = (sentiment - sentiment.min()) / (sentiment.max() - sentiment.min() + 1e-8)
    entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)

    # Create 2D terrain grid
    # X = narrative time, Y = "emotional dimension" (artificial spread), Z = height
    X, Y = np.meshgrid(time, np.linspace(-1, 1, n_breadth))

    # Create terrain height based on sentiment with gaussian spread
    Z = np.zeros_like(X)

    for i, t in enumerate(time):
        # Base height from sentiment
        base_height = sentiment[i]

        # Add entropy as "ruggedness"
        ruggedness = entropy[i] * 0.3

        # Create gaussian peak centered at y=0
        for j in range(n_breadth):
            y_val = Y[j, i]
            # Gaussian falloff from center
            spread = 0.3 + ruggedness  # Wider when entropy is high
            gaussian_weight = np.exp(-y_val**2 / (2 * spread**2))
            Z[j, i] = base_height * gaussian_weight

            # Add some noise based on entropy
            Z[j, i] += np.random.normal(0, ruggedness * 0.1)

    # Smooth the terrain
    Z = gaussian_filter(Z, sigma=1.5)

    # Find peaks in the original sentiment
    peaks, _ = find_peaks(sentiment, prominence=0.1, distance=5)
    valleys, _ = find_peaks(-sentiment, prominence=0.1, distance=5)

    # =========================================================================
    # FIGURE 1: Full 3D Terrain View
    # =========================================================================
    fig = plt.figure(figsize=(20, 16))

    # Main terrain view
    ax1 = fig.add_subplot(221, projection='3d')

    # Create custom colormap (ocean blue -> green -> yellow -> red -> white for peaks)
    colors_terrain = ['#1a237e', '#1976d2', '#4caf50', '#ffeb3b', '#ff5722', '#ffffff']
    n_bins_cmap = 256
    cmap_terrain = LinearSegmentedColormap.from_list('terrain_narrative', colors_terrain, N=n_bins_cmap)

    # Plot surface
    surf = ax1.plot_surface(X, Y, Z, cmap=cmap_terrain,
                            linewidth=0, antialiased=True, alpha=0.9,
                            rstride=1, cstride=1)

    # Mark peaks with flags
    for peak_idx in peaks:
        peak_height = sentiment[peak_idx]
        ax1.scatter([time[peak_idx]], [0], [peak_height + 0.05],
                   c='red', s=200, marker='^', edgecolors='black', linewidths=2, zorder=10)
        ax1.plot([time[peak_idx], time[peak_idx]], [0, 0], [0, peak_height],
                color='red', linewidth=2, linestyle='--', alpha=0.5)

    # Mark valleys
    for valley_idx in valleys:
        valley_height = sentiment[valley_idx]
        ax1.scatter([time[valley_idx]], [0], [valley_height],
                   c='blue', s=150, marker='v', edgecolors='black', linewidths=2, zorder=10)

    ax1.set_xlabel('Narrative Time', fontsize=12, labelpad=10)
    ax1.set_ylabel('Emotional Breadth', fontsize=12, labelpad=10)
    ax1.set_zlabel('Emotional Height', fontsize=12, labelpad=10)
    ax1.set_title('Narrative Terrain (Oblique View)', fontsize=14, fontweight='bold')
    ax1.view_init(elev=35, azim=45)

    # Add colorbar
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Emotional Intensity')

    # =========================================================================
    # FIGURE 2: Top-down view (like a topographic map)
    # =========================================================================
    ax2 = fig.add_subplot(222)

    # Contour plot
    levels = np.linspace(Z.min(), Z.max(), 20)
    contour = ax2.contourf(X, Y, Z, levels=levels, cmap=cmap_terrain)
    ax2.contour(X, Y, Z, levels=levels[::2], colors='black', linewidths=0.5, alpha=0.3)

    # Mark peaks
    for peak_idx in peaks:
        ax2.scatter([time[peak_idx]], [0], c='red', s=200, marker='^',
                   edgecolors='white', linewidths=2, zorder=10)
        ax2.annotate(f'{time[peak_idx]:.0%}',
                    xy=(time[peak_idx], 0),
                    xytext=(time[peak_idx], 0.3),
                    fontsize=9, ha='center', color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    # Mark valleys
    for valley_idx in valleys:
        ax2.scatter([time[valley_idx]], [0], c='blue', s=150, marker='v',
                   edgecolors='white', linewidths=2, zorder=10)

    ax2.set_xlabel('Narrative Time', fontsize=12)
    ax2.set_ylabel('Emotional Breadth', fontsize=12)
    ax2.set_title('Topographic Map View', fontsize=14, fontweight='bold')
    fig.colorbar(contour, ax=ax2, shrink=0.8, label='Elevation')

    # =========================================================================
    # FIGURE 3: Side profile (elevation chart)
    # =========================================================================
    ax3 = fig.add_subplot(223)

    # Plot filled elevation profile along the centerline (y=0)
    centerline_z = Z[n_breadth // 2, :]

    # Create gradient fill
    for i in range(len(time) - 1):
        height = centerline_z[i]
        color = cmap_terrain(height / centerline_z.max())
        ax3.fill_between(time[i:i+2], 0, centerline_z[i:i+2], color=color, alpha=0.8)

    ax3.plot(time, centerline_z, color='black', linewidth=2)

    # Mark peaks and valleys
    for peak_idx in peaks:
        ax3.scatter([time[peak_idx]], [centerline_z[peak_idx]],
                   c='red', s=200, marker='^', edgecolors='black', linewidths=2, zorder=10)
        ax3.annotate(f'Peak\n{time[peak_idx]:.0%}',
                    xy=(time[peak_idx], centerline_z[peak_idx]),
                    xytext=(time[peak_idx], centerline_z[peak_idx] + 0.15),
                    fontsize=9, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    for valley_idx in valleys:
        ax3.scatter([time[valley_idx]], [centerline_z[valley_idx]],
                   c='blue', s=150, marker='v', edgecolors='black', linewidths=2, zorder=10)

    ax3.set_xlabel('Narrative Time', fontsize=12)
    ax3.set_ylabel('Emotional Elevation', fontsize=12)
    ax3.set_title('Elevation Profile (Cross-Section)', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, centerline_z.max() * 1.2)
    ax3.fill_between(time, 0, 0, alpha=0.3, color='#1a237e', label='Sea Level')

    # =========================================================================
    # FIGURE 4: 3D from different angle (dramatic view)
    # =========================================================================
    ax4 = fig.add_subplot(224, projection='3d')

    surf2 = ax4.plot_surface(X, Y, Z, cmap=cmap_terrain,
                             linewidth=0, antialiased=True, alpha=0.9,
                             rstride=1, cstride=1)

    # Add "journey path" along the terrain
    path_z = centerline_z + 0.02  # Slightly above terrain
    ax4.plot(time, np.zeros_like(time), path_z,
            color='black', linewidth=3, label='Narrative Path')

    # Start and end markers
    ax4.scatter([0], [0], [centerline_z[0] + 0.05], c='green', s=300, marker='o',
               edgecolors='black', linewidths=2, label='Start')
    ax4.scatter([1], [0], [centerline_z[-1] + 0.05], c='red', s=300, marker='s',
               edgecolors='black', linewidths=2, label='End')

    ax4.set_xlabel('Narrative Time', fontsize=12, labelpad=10)
    ax4.set_ylabel('Emotional Breadth', fontsize=12, labelpad=10)
    ax4.set_zlabel('Emotional Height', fontsize=12, labelpad=10)
    ax4.set_title('Narrative Journey Through Terrain', fontsize=14, fontweight='bold')
    ax4.view_init(elev=20, azim=-60)
    ax4.legend(loc='upper left')

    plt.suptitle(f'{title}\nNarrative Terrain Geography', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / 'narrative_terrain.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    # =========================================================================
    # BONUS: High-resolution single terrain view
    # =========================================================================
    fig2 = plt.figure(figsize=(20, 14))
    ax = fig2.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cmap_terrain,
                          linewidth=0, antialiased=True, alpha=0.95,
                          rstride=1, cstride=1)

    # Mark all peaks with labels
    for i, peak_idx in enumerate(peaks):
        peak_height = sentiment[peak_idx]
        ax.scatter([time[peak_idx]], [0], [peak_height + 0.03],
                  c='red', s=250, marker='^', edgecolors='white', linewidths=2, zorder=10)
        ax.text(time[peak_idx], 0.2, peak_height + 0.08,
               f'Peak {i+1}\n({time[peak_idx]:.0%})',
               fontsize=10, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # Add path
    ax.plot(time, np.zeros_like(time), centerline_z + 0.01,
           color='black', linewidth=2.5, alpha=0.8)

    # Start/End
    ax.scatter([0], [0], [centerline_z[0] + 0.05], c='#4CAF50', s=400, marker='o',
              edgecolors='black', linewidths=2, label='Start', zorder=15)
    ax.scatter([1], [0], [centerline_z[-1] + 0.05], c='#F44336', s=400, marker='s',
              edgecolors='black', linewidths=2, label='End', zorder=15)

    ax.set_xlabel('Narrative Time', fontsize=14, labelpad=15)
    ax.set_ylabel('Emotional Breadth', fontsize=14, labelpad=15)
    ax.set_zlabel('Emotional Height', fontsize=14, labelpad=15)
    ax.set_title(f'{title}\n3D Narrative Terrain', fontsize=20, fontweight='bold', pad=20)
    ax.view_init(elev=30, azim=45)
    ax.legend(loc='upper left', fontsize=12)

    fig2.colorbar(surf, ax=ax, shrink=0.6, aspect=15, label='Emotional Intensity', pad=0.1)

    output_path2 = output_dir / 'narrative_terrain_hires.png'
    plt.savefig(output_path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path2}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Visualize narrative structure")
    parser.add_argument("--gutenberg", type=int, default=1399, help="Gutenberg book ID")
    parser.add_argument("--output-dir", type=str, default="./visualizations", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NARRATIVE VISUALIZATION SUITE")
    print("=" * 70)

    # Download text
    text, title = download_gutenberg(args.gutenberg)
    print(f"Title: {title}")
    print(f"Length: {len(text):,} characters")

    # Get trajectories
    sentiment_traj, entropy_traj = get_trajectories(text)

    # Get ICC result
    icc_result = get_icc_result(sentiment_traj)
    print(f"ICC Class: {icc_result.icc_class} - {icc_result.class_name}")

    print("\nGenerating visualizations...")

    # Generate all visualizations
    print("\n1. Sentiment trajectory...")
    plot_sentiment_trajectory(sentiment_traj, icc_result, title, output_dir)

    print("\n2. ICC radar chart...")
    plot_icc_radar(icc_result, title, output_dir)

    print("\n3. 3D narrative manifold...")
    plot_3d_manifold(sentiment_traj, entropy_traj, title, output_dir, text=text)

    print("\n4. Phase space portrait...")
    plot_phase_space(sentiment_traj, title, output_dir)

    print("\n5. Category Narr diagram...")
    plot_category_diagram(text, title, output_dir)

    print("\n6. Information geometry...")
    plot_information_geometry(sentiment_traj, entropy_traj, title, output_dir)

    print("\n7. ICC class comparison...")
    plot_icc_comparison(icc_result, title, output_dir)

    print("\n8. Peak distribution analysis...")
    plot_peak_distribution(sentiment_traj, title, output_dir)

    print("\n9. 3D narrative terrain...")
    plot_narrative_terrain(sentiment_traj, entropy_traj, title, output_dir)

    print("\n" + "=" * 70)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
