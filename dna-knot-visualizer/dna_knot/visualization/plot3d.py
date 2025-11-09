"""
3D knot visualization using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional
from dna_knot.core.types import Knot
from dna_knot.core.constants import TUBE_RADIUS


def plot_knot_3d(
    knot: Knot,
    color: str = 'blue',
    alpha: float = 0.8,
    linewidth: float = 2.0,
    show_vertices: bool = False,
    title: Optional[str] = None,
    ax: Optional[Axes3D] = None,
) -> Axes3D:
    """
    Plot knot in 3D using matplotlib.
    
    Args:
        knot: Knot to visualize.
        color: Line color.
        alpha: Line transparency.
        linewidth: Line width.
        show_vertices: If True, show vertices as points.
        title: Plot title.
        ax: Matplotlib 3D axis (if None, create new figure).
    
    Returns:
        Matplotlib 3D axis.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Get vertices (exclude duplicate closure vertex for plotting)
    vertices = knot.vertices[:-1] if knot.is_closed() else knot.vertices
    
    # Plot edges
    for i, j in knot.edges:
        if i < len(vertices) and j < len(vertices):
            x = [vertices[i, 0], vertices[j, 0]]
            y = [vertices[i, 1], vertices[j, 1]]
            z = [vertices[i, 2], vertices[j, 2]]
            ax.plot(x, y, z, color=color, alpha=alpha, linewidth=linewidth)
    
    # Optionally show vertices
    if show_vertices:
        ax.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            color='red',
            s=20,
            alpha=0.6
        )
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if title:
        ax.set_title(title)
    elif 'knot_name' in knot.metadata:
        ax.set_title(f"Knot: {knot.metadata['knot_name']}")
    
    # Set equal aspect ratio
    _set_equal_aspect_3d(ax, vertices)
    
    return ax


def save_knot_3d(
    knot: Knot,
    filepath: str,
    dpi: int = 150,
    **plot_kwargs
):
    """
    Save 3D knot visualization to file.
    
    Args:
        knot: Knot to visualize.
        filepath: Output file path (PNG, PDF, etc.).
        dpi: Resolution for raster formats.
        **plot_kwargs: Additional arguments passed to plot_knot_3d.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    plot_knot_3d(knot, ax=ax, **plot_kwargs)
    
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 3D plot to {filepath}")


def plot_knot_with_projection(
    knot: Knot,
    direction: Optional[np.ndarray] = None,
    figsize: tuple = (16, 8)
):
    """
    Plot knot in 3D alongside its planar projection.
    
    Args:
        knot: Knot to visualize.
        direction: Projection direction (if None, use default).
        figsize: Figure size (width, height).
    """
    fig = plt.figure(figsize=figsize)
    
    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    plot_knot_3d(knot, ax=ax1, title="3D Knot")
    
    # 2D projection
    diagram = knot.project(direction)
    ax2 = fig.add_subplot(122)
    _plot_diagram_2d(diagram, ax=ax2)
    
    plt.tight_layout()
    plt.show()


def _plot_diagram_2d(diagram, ax):
    """Plot 2D planar diagram (simple version)."""
    vertices2 = diagram.vertices2
    
    # Plot edges
    for i in range(len(vertices2) - 1):
        ax.plot(
            [vertices2[i, 0], vertices2[i + 1, 0]],
            [vertices2[i, 1], vertices2[i + 1, 1]],
            'b-',
            linewidth=2
        )
    
    # Plot crossings
    for crossing in diagram.crossings:
        x, y = crossing.point2
        color = 'red' if crossing.sign > 0 else 'green'
        ax.plot(x, y, 'o', color=color, markersize=8, alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'2D Projection ({diagram.n_crossings} crossings, writhe={diagram.writhe():.1f})')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def _set_equal_aspect_3d(ax: Axes3D, vertices: np.ndarray):
    """Set equal aspect ratio for 3D plot."""
    # Get data ranges
    x_range = [vertices[:, 0].min(), vertices[:, 0].max()]
    y_range = [vertices[:, 1].min(), vertices[:, 1].max()]
    z_range = [vertices[:, 2].min(), vertices[:, 2].max()]
    
    # Compute maximum range
    max_range = max(
        x_range[1] - x_range[0],
        y_range[1] - y_range[0],
        z_range[1] - z_range[0]
    )
    
    # Set limits centered around data
    x_mid = (x_range[0] + x_range[1]) / 2
    y_mid = (y_range[0] + y_range[1]) / 2
    z_mid = (z_range[0] + z_range[1]) / 2
    
    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)
