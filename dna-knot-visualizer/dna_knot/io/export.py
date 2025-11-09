"""
Export knot geometry to various formats (OBJ, JSON).
"""

import json
import numpy as np
from pathlib import Path
from dna_knot.core.types import Knot, PlanarDiagram


def export_obj(knot: Knot, filepath: str, tube_radius: float = 0.05):
    """
    Export knot as OBJ file (simple polyline representation).
    
    For proper tube geometry, use a mesh library like trimesh.
    
    Args:
        knot: Knot to export.
        filepath: Output OBJ file path.
        tube_radius: Tube radius (for future tube mesh generation).
    """
    vertices = knot.vertices[:-1] if knot.is_closed() else knot.vertices
    
    with open(filepath, 'w') as f:
        # Write header
        f.write("# DNA Knot Visualizer - Knot Export\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Edges: {len(knot.edges)}\n\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write edges as line segments
        f.write("\n")
        for i, j in knot.edges:
            if i < len(vertices) and j < len(vertices):
                f.write(f"l {i+1} {j+1}\n")
    
    print(f"Exported OBJ to {filepath}")


def export_json(knot: Knot, filepath: str):
    """
    Export knot as JSON (vertices and metadata).
    
    Args:
        knot: Knot to export.
        filepath: Output JSON file path.
    """
    data = {
        "vertices": knot.vertices.tolist(),
        "edges": knot.edges,
        "metadata": knot.metadata,
        "n_vertices": knot.n_vertices,
        "n_edges": knot.n_edges,
        "is_closed": knot.is_closed(),
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported JSON to {filepath}")


def export_diagram_json(diagram: PlanarDiagram, filepath: str):
    """
    Export planar diagram as JSON.
    
    Args:
        diagram: Planar diagram to export.
        filepath: Output JSON file path.
    """
    crossings_data = []
    for c in diagram.crossings:
        crossings_data.append({
            "a_idx": c.a_idx,
            "b_idx": c.b_idx,
            "point2": list(c.point2),
            "over_segment": c.over_segment,
            "sign": c.sign,
            "params": list(c.params),
        })
    
    data = {
        "vertices2": diagram.vertices2.tolist(),
        "crossings": crossings_data,
        "adjacency": {str(k): v for k, v in diagram.adjacency.items()},
        "projection_direction": diagram.projection_direction.tolist(),
        "n_crossings": diagram.n_crossings,
        "writhe": diagram.writhe(),
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported diagram JSON to {filepath}")
