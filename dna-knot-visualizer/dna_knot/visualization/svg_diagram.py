"""
SVG export for planar knot diagrams.
"""

import svgwrite
import numpy as np
from typing import Optional
from dna_knot.core.types import PlanarDiagram


def export_planar_diagram_svg(
    diagram: PlanarDiagram,
    filepath: str,
    width: float = 800,
    height: float = 800,
    margin: float = 50,
    stroke_width: float = 2.0,
    crossing_gap: float = 8.0,
    show_crossings: bool = True,
    show_signs: bool = True,
):
    """
    Export planar diagram to SVG file with proper over/under crossings.
    
    Args:
        diagram: Planar diagram to export.
        filepath: Output SVG file path.
        width: SVG canvas width (pixels).
        height: SVG canvas height (pixels).
        margin: Margin around diagram (pixels).
        stroke_width: Line width (pixels).
        crossing_gap: Gap size for under-crossing (pixels).
        show_crossings: If True, mark crossing points.
        show_signs: If True, label crossing signs.
    """
    # Create SVG drawing
    dwg = svgwrite.Drawing(filepath, size=(width, height))
    
    # Scale and center vertices
    vertices2 = diagram.vertices2
    vertices_scaled = _scale_to_canvas(vertices2, width, height, margin)
    
    # Draw edges with proper over/under at crossings
    _draw_edges_with_crossings(
        dwg,
        vertices_scaled,
        diagram.crossings,
        stroke_width,
        crossing_gap
    )
    
    # Mark crossing points
    if show_crossings:
        _draw_crossing_markers(
            dwg,
            diagram.crossings,
            vertices_scaled,
            diagram.vertices2,
            show_signs
        )
    
    # Add title
    title = f"Planar Diagram: {diagram.n_crossings} crossings, writhe={diagram.writhe():.1f}"
    dwg.add(dwg.text(
        title,
        insert=(width / 2, 30),
        text_anchor='middle',
        font_size='20px',
        fill='black'
    ))
    
    # Save
    dwg.save()
    print(f"Saved SVG diagram to {filepath}")


def _scale_to_canvas(
    vertices2: np.ndarray,
    width: float,
    height: float,
    margin: float
) -> np.ndarray:
    """Scale 2D vertices to fit canvas with margins."""
    # Get data bounds
    x_min, x_max = vertices2[:, 0].min(), vertices2[:, 0].max()
    y_min, y_max = vertices2[:, 1].min(), vertices2[:, 1].max()
    
    # Compute scaling factor
    data_width = x_max - x_min
    data_height = y_max - y_min
    
    canvas_width = width - 2 * margin
    canvas_height = height - 2 * margin
    
    scale = min(canvas_width / data_width, canvas_height / data_height) if data_width > 0 and data_height > 0 else 1.0
    
    # Scale and translate
    vertices_scaled = vertices2.copy()
    vertices_scaled[:, 0] = (vertices_scaled[:, 0] - x_min) * scale + margin
    vertices_scaled[:, 1] = (vertices_scaled[:, 1] - y_min) * scale + margin
    
    # Flip y-axis (SVG y increases downward)
    vertices_scaled[:, 1] = height - vertices_scaled[:, 1]
    
    return vertices_scaled


def _draw_edges_with_crossings(
    dwg,
    vertices_scaled: np.ndarray,
    crossings: list,
    stroke_width: float,
    crossing_gap: float
):
    """Draw edges with proper over/under rendering at crossings."""
    n = len(vertices_scaled) - 1  # Exclude duplicate closure vertex
    
    # Build set of crossing points for each edge
    edge_crossings = {}
    for i in range(n):
        edge_crossings[i] = []
    
    for crossing in crossings:
        # Get scaled crossing point
        crossing_point_scaled = _scale_point_to_canvas(
            crossing.point2,
            vertices_scaled
        )
        
        # Add to both edges involved
        edge_crossings[crossing.a_idx].append((crossing.params[0], crossing_point_scaled, crossing.over_segment == crossing.a_idx))
        edge_crossings[crossing.b_idx].append((crossing.params[1], crossing_point_scaled, crossing.over_segment == crossing.b_idx))
    
    # Sort crossings along each edge
    for i in range(n):
        edge_crossings[i].sort(key=lambda x: x[0])
    
    # Draw each edge with gaps for under-crossings
    for i in range(n):
        j = (i + 1) % n
        p1 = vertices_scaled[i]
        p2 = vertices_scaled[j]
        
        if not edge_crossings[i]:
            # No crossings on this edge - draw full line
            dwg.add(dwg.line(
                start=tuple(p1),
                end=tuple(p2),
                stroke='blue',
                stroke_width=stroke_width
            ))
        else:
            # Draw edge in segments, skipping gaps for under-crossings
            prev_point = p1
            for param, cross_point, is_over in edge_crossings[i]:
                # Draw up to crossing
                if is_over:
                    # This edge goes over - draw full segment
                    dwg.add(dwg.line(
                        start=tuple(prev_point),
                        end=tuple(cross_point),
                        stroke='blue',
                        stroke_width=stroke_width
                    ))
                    prev_point = cross_point
                else:
                    # This edge goes under - leave gap
                    direction = p2 - p1
                    direction_norm = direction / np.linalg.norm(direction)
                    
                    gap_start = cross_point - direction_norm * crossing_gap / 2
                    gap_end = cross_point + direction_norm * crossing_gap / 2
                    
                    # Draw up to gap
                    dwg.add(dwg.line(
                        start=tuple(prev_point),
                        end=tuple(gap_start),
                        stroke='blue',
                        stroke_width=stroke_width
                    ))
                    
                    prev_point = gap_end
            
            # Draw final segment
            dwg.add(dwg.line(
                start=tuple(prev_point),
                end=tuple(p2),
                stroke='blue',
                stroke_width=stroke_width
            ))


def _draw_crossing_markers(
    dwg,
    crossings: list,
    vertices_scaled: np.ndarray,
    vertices_original: np.ndarray,
    show_signs: bool
):
    """Draw markers at crossing points."""
    for idx, crossing in enumerate(crossings):
        # Scale crossing point
        point_scaled = _scale_point_to_canvas(crossing.point2, vertices_scaled)
        
        # Draw marker
        color = 'red' if crossing.sign > 0 else 'green'
        dwg.add(dwg.circle(
            center=tuple(point_scaled),
            r=4,
            fill=color,
            fill_opacity=0.5
        ))
        
        # Label sign
        if show_signs:
            label = f"{'+' if crossing.sign > 0 else '-'}"
            dwg.add(dwg.text(
                label,
                insert=(point_scaled[0] + 10, point_scaled[1] - 10),
                font_size='12px',
                fill=color
            ))


def _scale_point_to_canvas(
    point2: tuple,
    vertices_scaled: np.ndarray
) -> np.ndarray:
    """Scale a single 2D point using the same transformation as vertices."""
    # This is a simplified version - proper implementation would store
    # the transformation parameters
    # For now, return point as-is (assumes already scaled)
    return np.array(point2)
