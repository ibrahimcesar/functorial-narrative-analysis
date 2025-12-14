"""
Interactive Web Visualizations using Plotly

Generates interactive HTML files with:
- 3D rotatable narrative terrain
- Side-by-side comparison of multiple narratives
- Hoverable data points with narrative context
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class NarrativeData:
    """Container for narrative visualization data."""
    title: str
    trajectory: np.ndarray
    icc_class: str
    class_name: str
    cultural_prediction: str
    features: Dict[str, Any]
    peaks: Optional[np.ndarray] = None
    valleys: Optional[np.ndarray] = None


def check_plotly():
    """Check if Plotly is available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for interactive visualizations. "
            "Install with: pip install plotly"
        )


def create_terrain_surface(
    trajectory: np.ndarray,
    resolution: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 3D terrain surface from a 1D trajectory.

    Args:
        trajectory: 1D array of values
        resolution: Grid resolution

    Returns:
        X, Y, Z arrays for surface plot
    """
    n_points = len(trajectory)

    # Create grid
    x = np.linspace(0, 100, resolution)  # Narrative progress (%)
    y = np.linspace(-30, 30, resolution)  # Emotional breadth
    X, Y = np.meshgrid(x, y)

    # Interpolate trajectory to match grid
    traj_x = np.linspace(0, 100, n_points)
    traj_interp = np.interp(x, traj_x, trajectory)

    # Create terrain with Gaussian spread
    Z = np.zeros_like(X)
    sigma = 8  # Spread width

    for i, (xi, val) in enumerate(zip(x, traj_interp)):
        # Gaussian cross-section
        cross_section = val * np.exp(-Y[0, :]**2 / (2 * sigma**2))
        col_idx = i
        Z[:, col_idx] = cross_section * 50 + 25  # Scale to reasonable height

    return X, Y, Z


def create_interactive_terrain(
    narrative: NarrativeData,
    output_path: Optional[Path] = None,
    include_annotations: bool = True
) -> str:
    """
    Create an interactive 3D terrain visualization.

    Args:
        narrative: NarrativeData object
        output_path: Where to save the HTML file
        include_annotations: Whether to add peak/valley markers

    Returns:
        HTML string or path to saved file
    """
    check_plotly()

    # Create terrain surface
    X, Y, Z = create_terrain_surface(narrative.trajectory)

    # Custom colorscale (terrain-like)
    colorscale = [
        [0.0, '#1a4c6e'],    # Deep blue (low)
        [0.2, '#2d6a4f'],    # Dark green
        [0.4, '#52b788'],    # Green
        [0.6, '#95d5b2'],    # Light green
        [0.7, '#d4a574'],    # Tan
        [0.85, '#bc6c25'],   # Brown/orange
        [0.95, '#9d0208'],   # Dark red
        [1.0, '#ffffff'],    # White (peaks)
    ]

    # Create figure
    fig = go.Figure()

    # Add terrain surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title="Emotional<br>Intensity",
            titleside="right",
            tickformat=".0f"
        ),
        opacity=0.95,
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="white",
                project=dict(z=True)
            )
        ),
        hovertemplate=(
            "Progress: %{x:.0f}%<br>"
            "Emotional Intensity: %{z:.1f}<br>"
            "<extra></extra>"
        )
    ))

    # Add journey line on top of terrain
    n_points = len(narrative.trajectory)
    journey_x = np.linspace(0, 100, n_points)
    journey_y = np.zeros(n_points)
    journey_z = narrative.trajectory * 50 + 25 + 2  # Slightly above surface

    fig.add_trace(go.Scatter3d(
        x=journey_x, y=journey_y, z=journey_z,
        mode='lines',
        line=dict(color='gold', width=4),
        name='Narrative Path',
        hovertemplate=(
            "Progress: %{x:.0f}%<br>"
            "Sentiment: %{customdata:.2f}<br>"
            "<extra>Narrative Path</extra>"
        ),
        customdata=narrative.trajectory
    ))

    # Add peak markers if available
    if narrative.peaks is not None and len(narrative.peaks) > 0:
        peak_x = narrative.peaks / n_points * 100
        peak_z = narrative.trajectory[narrative.peaks] * 50 + 25 + 5

        fig.add_trace(go.Scatter3d(
            x=peak_x,
            y=np.zeros(len(narrative.peaks)),
            z=peak_z,
            mode='markers+text',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond'
            ),
            text=[f'Peak {i+1}' for i in range(len(narrative.peaks))],
            textposition='top center',
            textfont=dict(size=10, color='red'),
            name='Dramatic Peaks',
            hovertemplate=(
                "Peak at %{x:.0f}%<br>"
                "Intensity: %{z:.1f}<br>"
                "<extra>Dramatic Peak</extra>"
            )
        ))

    # Add valley markers if available
    if narrative.valleys is not None and len(narrative.valleys) > 0:
        valley_x = narrative.valleys / n_points * 100
        valley_z = narrative.trajectory[narrative.valleys] * 50 + 25 + 3

        fig.add_trace(go.Scatter3d(
            x=valley_x,
            y=np.zeros(len(narrative.valleys)),
            z=valley_z,
            mode='markers',
            marker=dict(
                size=6,
                color='blue',
                symbol='circle'
            ),
            name='Valleys',
            hovertemplate=(
                "Valley at %{x:.0f}%<br>"
                "Intensity: %{z:.1f}<br>"
                "<extra>Emotional Valley</extra>"
            )
        ))

    # Start and end markers
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[journey_z[0] + 3],
        mode='markers+text',
        marker=dict(size=10, color='green', symbol='circle'),
        text=['START'],
        textposition='top center',
        textfont=dict(size=12, color='green'),
        name='Start',
        showlegend=False
    ))

    fig.add_trace(go.Scatter3d(
        x=[100], y=[0], z=[journey_z[-1] + 3],
        mode='markers+text',
        marker=dict(size=10, color='purple', symbol='square'),
        text=['END'],
        textposition='top center',
        textfont=dict(size=12, color='purple'),
        name='End',
        showlegend=False
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{narrative.title}</b><br>"
                f"<span style='font-size:14px'>{narrative.icc_class} - {narrative.class_name} "
                f"({narrative.cultural_prediction.title()} pattern)</span>"
            ),
            x=0.5,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title="Narrative Progress (%)",
            yaxis_title="Emotional Breadth",
            zaxis_title="Intensity",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1)
            ),
            aspectratio=dict(x=2, y=1, z=0.7),
        ),
        width=1000,
        height=700,
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Add annotation with key metrics
    features = narrative.features
    annotation_text = (
        f"Net Change: {features.get('net_change', 0):+.2f} | "
        f"Peaks: {features.get('n_peaks', 0)} | "
        f"Volatility: {features.get('volatility', 0):.3f}"
    )

    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.5, y=-0.02,
        showarrow=False,
        font=dict(size=12, color="gray")
    )

    # Generate HTML
    html = fig.to_html(
        full_html=True,
        include_plotlyjs='cdn',
        config={
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['orbitRotation', 'resetCameraDefault3d'],
            'displaylogo': False
        }
    )

    if output_path:
        output_path = Path(output_path)
        output_path.write_text(html, encoding='utf-8')
        return str(output_path)

    return html


def create_comparison_page(
    narratives: List[NarrativeData],
    output_path: Optional[Path] = None,
    title: str = "Narrative Terrain Comparison"
) -> str:
    """
    Create an interactive page comparing multiple narrative terrains side-by-side.

    Args:
        narratives: List of NarrativeData objects
        output_path: Where to save the HTML file
        title: Page title

    Returns:
        HTML string or path to saved file
    """
    check_plotly()

    n_narratives = len(narratives)

    if n_narratives == 0:
        raise ValueError("At least one narrative is required")

    # Determine grid layout
    if n_narratives == 1:
        rows, cols = 1, 1
    elif n_narratives == 2:
        rows, cols = 1, 2
    elif n_narratives <= 4:
        rows, cols = 2, 2
    elif n_narratives <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3

    # Create subplot specs for 3D scenes
    specs = [[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)]

    # Create subplot titles
    subplot_titles = [
        f"{n.title}<br><span style='font-size:11px'>{n.icc_class} - {n.class_name}</span>"
        for n in narratives
    ]
    # Pad with empty titles if needed
    while len(subplot_titles) < rows * cols:
        subplot_titles.append("")

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.12
    )

    # Custom colorscale
    colorscale = [
        [0.0, '#1a4c6e'],
        [0.2, '#2d6a4f'],
        [0.4, '#52b788'],
        [0.6, '#95d5b2'],
        [0.7, '#d4a574'],
        [0.85, '#bc6c25'],
        [0.95, '#9d0208'],
        [1.0, '#ffffff'],
    ]

    # Add each narrative
    for idx, narrative in enumerate(narratives):
        row = idx // cols + 1
        col = idx % cols + 1
        scene_name = f'scene{idx + 1}' if idx > 0 else 'scene'

        # Create terrain
        X, Y, Z = create_terrain_surface(narrative.trajectory, resolution=60)

        # Add surface
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z,
                colorscale=colorscale,
                showscale=(idx == 0),  # Only show colorbar for first
                opacity=0.9,
                hovertemplate=(
                    f"<b>{narrative.title}</b><br>"
                    "Progress: %{x:.0f}%<br>"
                    "Intensity: %{z:.1f}<br>"
                    "<extra></extra>"
                )
            ),
            row=row, col=col
        )

        # Add journey line
        n_points = len(narrative.trajectory)
        journey_x = np.linspace(0, 100, n_points)
        journey_z = narrative.trajectory * 50 + 27

        fig.add_trace(
            go.Scatter3d(
                x=journey_x,
                y=np.zeros(n_points),
                z=journey_z,
                mode='lines',
                line=dict(color='gold', width=3),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )

        # Configure scene
        scene_config = dict(
            xaxis_title="Progress %",
            yaxis_title="",
            zaxis_title="",
            camera=dict(
                eye=dict(x=1.8, y=1.2, z=0.8)
            ),
            aspectratio=dict(x=2, y=1, z=0.6)
        )
        fig.update_layout(**{scene_name: scene_config})

    # Create comparison table HTML
    table_html = create_comparison_table_html(narratives)

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            font=dict(size=22)
        ),
        width=400 * cols,
        height=400 * rows + 100,
        margin=dict(l=20, r=20, t=100, b=20),
        showlegend=False
    )

    # Generate HTML with custom styling
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs='cdn',
        config={'displayModeBar': True, 'displaylogo': False}
    )

    # Full HTML with table
    full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }}
        .container {{
            max-width: 1800px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #fff;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .plot-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        .table-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th {{
            background: rgba(255,255,255,0.1);
            padding: 12px 15px;
            text-align: left;
            border-bottom: 2px solid rgba(255,255,255,0.2);
        }}
        td {{
            padding: 10px 15px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        tr:hover {{
            background: rgba(255,255,255,0.05);
        }}
        .icc-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 12px;
        }}
        .icc-0 {{ background: #6c757d; }}
        .icc-1 {{ background: #198754; }}
        .icc-2 {{ background: #0dcaf0; color: #000; }}
        .icc-3 {{ background: #ffc107; color: #000; }}
        .icc-4 {{ background: #fd7e14; }}
        .icc-5 {{ background: #dc3545; }}
        .cultural-western {{ color: #ffc107; }}
        .cultural-japanese {{ color: #0dcaf0; }}
        .cultural-neutral {{ color: #6c757d; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Narrative Terrain Comparison</h1>
        <p class="subtitle">Interactive 3D visualization of emotional landscapes across narratives</p>

        <div class="plot-container">
            {plot_html}
        </div>

        <div class="table-container">
            <h3 style="margin-top:0;">Comparison Metrics</h3>
            {table_html}
        </div>

        <p style="text-align:center; color:#666; margin-top:30px; font-size:12px;">
            Generated by Functorial Narrative Analysis | ICC Model v2
        </p>
    </div>
</body>
</html>
"""

    if output_path:
        output_path = Path(output_path)
        output_path.write_text(full_html, encoding='utf-8')
        return str(output_path)

    return full_html


def create_comparison_table_html(narratives: List[NarrativeData]) -> str:
    """Create an HTML table comparing narrative metrics."""
    rows = []

    for n in narratives:
        features = n.features
        net_change = features.get('net_change', 0)
        net_class = 'positive' if net_change > 0 else 'negative' if net_change < 0 else ''

        cultural_class = f"cultural-{n.cultural_prediction}"
        icc_num = n.icc_class.split('-')[1]

        rows.append(f"""
        <tr>
            <td><strong>{n.title}</strong></td>
            <td><span class="icc-badge icc-{icc_num}">{n.icc_class}</span></td>
            <td>{n.class_name}</td>
            <td class="{cultural_class}">{n.cultural_prediction.title()}</td>
            <td class="{net_class}">{net_change:+.3f}</td>
            <td>{features.get('n_peaks', 0)}</td>
            <td>{features.get('volatility', 0):.4f}</td>
            <td>{features.get('trend_r2', 0):.3f}</td>
        </tr>
        """)

    return f"""
    <table>
        <thead>
            <tr>
                <th>Title</th>
                <th>ICC Class</th>
                <th>Pattern Name</th>
                <th>Cultural Style</th>
                <th>Net Change</th>
                <th>Peaks</th>
                <th>Volatility</th>
                <th>Trend R²</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def create_trajectory_comparison_2d(
    narratives: List[NarrativeData],
    output_path: Optional[Path] = None
) -> str:
    """
    Create a 2D overlay comparison of narrative trajectories.

    Args:
        narratives: List of NarrativeData objects
        output_path: Where to save the HTML file

    Returns:
        HTML string or path to saved file
    """
    check_plotly()

    fig = go.Figure()

    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
        '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'
    ]

    for idx, narrative in enumerate(narratives):
        color = colors[idx % len(colors)]
        x = np.linspace(0, 100, len(narrative.trajectory))

        fig.add_trace(go.Scatter(
            x=x,
            y=narrative.trajectory,
            mode='lines',
            name=f"{narrative.title} ({narrative.icc_class})",
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{narrative.title}</b><br>"
                "Progress: %{x:.0f}%<br>"
                "Sentiment: %{y:.3f}<br>"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title="Narrative Trajectory Comparison",
        xaxis_title="Narrative Progress (%)",
        yaxis_title="Emotional Valence",
        width=1000,
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    html = fig.to_html(
        full_html=True,
        include_plotlyjs='cdn',
        config={'displayModeBar': True, 'displaylogo': False}
    )

    if output_path:
        output_path = Path(output_path)
        output_path.write_text(html, encoding='utf-8')
        return str(output_path)

    return html
