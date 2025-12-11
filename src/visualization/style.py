"""
Centralized visualization style configuration.

Sets JetBrains Mono as the default font and provides consistent styling
across all visualizations in the project.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def _get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    # Navigate up from src/visualization/style.py to project root
    return current.parent.parent.parent


def setup_style():
    """
    Configure matplotlib to use JetBrains Mono and project styling.

    Call this function at the start of any visualization script or notebook.
    """
    project_root = _get_project_root()
    fonts_dir = project_root / "assets" / "fonts"

    # Register JetBrains Mono fonts if available
    font_files = [
        fonts_dir / "JetBrainsMono-Regular.ttf",
        fonts_dir / "JetBrainsMono-Bold.ttf",
        fonts_dir / "JetBrainsMono-Italic.ttf",
    ]

    fonts_loaded = False
    for font_file in font_files:
        if font_file.exists():
            fm.fontManager.addfont(str(font_file))
            fonts_loaded = True

    # Apply style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Set JetBrains Mono as default font
    if fonts_loaded:
        plt.rcParams['font.family'] = 'JetBrains Mono'
    else:
        # Fallback to monospace if fonts not found
        plt.rcParams['font.family'] = 'monospace'

    # Additional style settings
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    return fonts_loaded


# Color palettes for consistent styling
COLORS = {
    'sentiment_positive': '#2ecc71',  # Green
    'sentiment_negative': '#e74c3c',  # Red
    'sentiment_neutral': '#3498db',   # Blue
    'arousal_high': '#9b59b6',        # Purple
    'arousal_low': '#1abc9c',         # Teal
    'primary': '#3498db',
    'secondary': '#e74c3c',
    'accent': '#f39c12',
    'dark': '#2c3e50',
    'light': '#ecf0f1',
}

# Sentiment color gradient (dark red -> orange -> green)
def get_sentiment_color(sentiment: float) -> str:
    """Get color based on sentiment value (-1 to 1 or 0 to 1)."""
    if sentiment < 0.1:
        return '#8b0000'  # Dark red
    elif sentiment < 0.3:
        return '#e74c3c'  # Red
    elif sentiment < 0.5:
        return '#f39c12'  # Orange
    else:
        return '#2ecc71'  # Green


def get_harmon_color(conformance: float) -> str:
    """Get color based on Harmon Circle conformance (0 to 1)."""
    if conformance >= 0.9:
        return '#2ecc71'  # Green - excellent fit
    elif conformance >= 0.7:
        return '#3498db'  # Blue - good fit
    else:
        return '#f39c12'  # Orange - moderate fit
