"""
Create a simple spherical cell mesh for minimal example.
"""

import numpy as np
from pathlib import Path
from cc.geometry.mesh import Mesh

def create_cell_mesh(radius: float = 1e-6, refinement: int = 2):
    """
    Create a spherical cell mesh.
    
    Args:
        radius: Cell radius (meters), default 1 µm
        refinement: Mesh refinement level
        
    Returns:
        Mesh object
    """
    print(f"Creating spherical cell mesh: radius={radius*1e6:.2f} µm, refinement={refinement}")
    
    # Create sphere mesh
    mesh = Mesh.create_sphere(radius=radius, refinement=refinement)
    
    # Add cytosol compartment
    from cc.geometry.mesh import Compartment
    
    # Compute volume (4/3 π r³)
    volume = (4.0 / 3.0) * np.pi * (radius ** 3)
    
    cytosol = Compartment(
        name="cytosol",
        volume=volume,
        node_indices=np.arange(len(mesh.nodes)),
        properties={'description': 'Bacterial cytosol'}
    )
    
    mesh.add_compartment(cytosol)
    
    print(f"  Nodes: {len(mesh.nodes)}")
    print(f"  Elements: {len(mesh.elements)}")
    print(f"  Volume: {volume:.2e} m³ ({volume*1e18:.2f} µm³)")
    
    return mesh


if __name__ == '__main__':
    # Create output directory
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mesh
    mesh = create_cell_mesh(radius=1e-6, refinement=2)
    
    # Save to file
    output_file = output_dir / "cell_mesh.xdmf"
    
    try:
        mesh.to_file(str(output_file))
        print(f"\nMesh saved to {output_file}")
    except ImportError:
        print("\nWarning: meshio not installed, cannot save XDMF file")
        print("Install with: pip install meshio")
        print("Mesh object created successfully in memory")
    
    print("\nMesh creation complete!")
