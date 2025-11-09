"""
Session save/load for knots and computed results.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from dna_knot.core.types import Knot


def save_session(
    knot: Knot,
    filepath: str,
    invariants: Optional[Dict[str, Any]] = None,
    diagram_data: Optional[Dict[str, Any]] = None
):
    """
    Save knot session to JSON file.
    
    Args:
        knot: Knot to save.
        filepath: Output JSON file path.
        invariants: Optional dictionary of computed invariants.
        diagram_data: Optional planar diagram data.
    """
    session_data = {
        "vertices": knot.vertices.tolist(),
        "edges": knot.edges,
        "metadata": knot.metadata,
        "invariants": invariants or {},
        "diagram": diagram_data or {},
    }
    
    with open(filepath, 'w') as f:
        json.dump(session_data, f, indent=2)
    
    print(f"Saved session to {filepath}")


def load_session(filepath: str) -> tuple:
    """
    Load knot session from JSON file.
    
    Args:
        filepath: Input JSON file path.
    
    Returns:
        (knot, invariants, diagram_data) tuple.
    """
    with open(filepath, 'r') as f:
        session_data = json.load(f)
    
    # Reconstruct knot
    vertices = np.array(session_data["vertices"], dtype=np.float64)
    edges = [tuple(e) for e in session_data["edges"]]
    metadata = session_data["metadata"]
    
    knot = Knot(vertices=vertices, edges=edges, metadata=metadata)
    
    invariants = session_data.get("invariants", {})
    diagram_data = session_data.get("diagram", {})
    
    print(f"Loaded session from {filepath}")
    
    return knot, invariants, diagram_data


def save_knot_npz(knot: Knot, filepath: str):
    """
    Save knot to NPZ (NumPy compressed format) for efficient storage.
    
    Args:
        knot: Knot to save.
        filepath: Output NPZ file path.
    """
    np.savez_compressed(
        filepath,
        vertices=knot.vertices,
        edges=np.array(knot.edges),
        metadata=np.array([knot.metadata])  # Store as object array
    )
    print(f"Saved knot to {filepath}")


def load_knot_npz(filepath: str) -> Knot:
    """
    Load knot from NPZ file.
    
    Args:
        filepath: Input NPZ file path.
    
    Returns:
        Knot object.
    """
    data = np.load(filepath, allow_pickle=True)
    
    vertices = data['vertices']
    edges = [tuple(e) for e in data['edges']]
    metadata = data['metadata'].item() if 'metadata' in data else {}
    
    return Knot(vertices=vertices, edges=edges, metadata=metadata)
