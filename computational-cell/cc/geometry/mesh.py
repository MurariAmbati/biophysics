"""
Geometry and mesh management for spatial discretization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Compartment:
    """
    Represents a cellular compartment (cytosol, nucleus, organelles, etc).
    """
    
    name: str
    volume: float  # m³
    node_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    element_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    properties: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate compartment data."""
        if self.volume <= 0:
            raise ValueError(f"Compartment {self.name} must have positive volume")
    
    def get_center(self, mesh: 'Mesh') -> np.ndarray:
        """Compute geometric center of compartment."""
        if len(self.node_indices) == 0:
            return np.array([0.0, 0.0, 0.0])
        nodes = mesh.nodes[self.node_indices]
        return np.mean(nodes, axis=0)


class Mesh:
    """
    Computational mesh for spatial discretization.
    
    Supports volumetric (tetrahedral) and surface (triangular) meshes.
    """
    
    def __init__(
        self,
        nodes: np.ndarray,
        elements: np.ndarray,
        element_type: str = "tetra",
        compartments: Optional[List[Compartment]] = None,
        boundary_labels: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Initialize mesh.
        
        Args:
            nodes: Node coordinates, shape (N_nodes, 3), dtype float64
            elements: Element connectivity, shape (N_elems, nodes_per_elem), dtype int64
            element_type: "tetra" (tetrahedral) or "tri" (triangular)
            compartments: List of compartments
            boundary_labels: Dict mapping boundary labels to face indices
        """
        self.nodes = nodes.astype(np.float64)
        self.elements = elements.astype(np.int64)
        self.element_type = element_type
        self.compartments = compartments or []
        self.boundary_labels = boundary_labels or {}
        
        # Validate
        self._validate()
        
        # Compute derived quantities
        self._compute_quality_metrics()
        
        logger.info(f"Mesh initialized: {len(self.nodes)} nodes, {len(self.elements)} elements")
    
    def _validate(self) -> None:
        """Validate mesh data."""
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("Nodes must have shape (N, 3)")
        
        if self.elements.ndim != 2:
            raise ValueError("Elements must be 2D array")
        
        # Check element connectivity
        max_node_idx = np.max(self.elements)
        if max_node_idx >= len(self.nodes):
            raise ValueError(f"Element references non-existent node {max_node_idx}")
        
        # Validate element types
        if self.element_type == "tetra":
            if self.elements.shape[1] != 4:
                raise ValueError("Tetrahedral elements must have 4 nodes")
        elif self.element_type == "tri":
            if self.elements.shape[1] != 3:
                raise ValueError("Triangular elements must have 3 nodes")
        else:
            raise ValueError(f"Unsupported element type: {self.element_type}")
    
    def _compute_quality_metrics(self) -> None:
        """Compute mesh quality metrics."""
        if self.element_type == "tetra":
            self.element_volumes = self._compute_tetra_volumes()
            self.min_volume = np.min(self.element_volumes)
            self.max_volume = np.max(self.element_volumes)
            
            # Quality checks
            if self.min_volume <= 0:
                n_negative = np.sum(self.element_volumes <= 0)
                logger.warning(f"Found {n_negative} elements with non-positive volume")
            
            # Aspect ratio (simplified)
            self.aspect_ratios = self._compute_aspect_ratios()
            self.max_aspect_ratio = np.max(self.aspect_ratios)
            
            if self.max_aspect_ratio > 10:
                logger.warning(f"Max aspect ratio {self.max_aspect_ratio:.2f} exceeds recommended threshold (10)")
    
    def _compute_tetra_volumes(self) -> np.ndarray:
        """Compute volumes of tetrahedral elements."""
        # Get node coordinates for each element
        v0 = self.nodes[self.elements[:, 0]]
        v1 = self.nodes[self.elements[:, 1]]
        v2 = self.nodes[self.elements[:, 2]]
        v3 = self.nodes[self.elements[:, 3]]
        
        # Volume = |det(v1-v0, v2-v0, v3-v0)| / 6
        mat = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=2)
        det = np.linalg.det(mat)
        volumes = np.abs(det) / 6.0
        
        return volumes
    
    def _compute_aspect_ratios(self) -> np.ndarray:
        """Compute aspect ratios (max edge / min edge) for elements."""
        aspect_ratios = np.zeros(len(self.elements))
        
        for i, elem in enumerate(self.elements):
            nodes = self.nodes[elem]
            
            # Compute all edge lengths
            edges = []
            for j in range(len(elem)):
                for k in range(j + 1, len(elem)):
                    edge_length = np.linalg.norm(nodes[j] - nodes[k])
                    edges.append(edge_length)
            
            if len(edges) > 0:
                aspect_ratios[i] = max(edges) / (min(edges) + 1e-12)
        
        return aspect_ratios
    
    def get_total_volume(self) -> float:
        """Get total mesh volume."""
        if self.element_type == "tetra":
            return np.sum(self.element_volumes)
        else:
            raise NotImplementedError("Volume calculation only for tetrahedral meshes")
    
    def get_compartment(self, name: str) -> Optional[Compartment]:
        """Get compartment by name."""
        for comp in self.compartments:
            if comp.name == name:
                return comp
        return None
    
    def add_compartment(self, compartment: Compartment) -> None:
        """Add a compartment to the mesh."""
        if self.get_compartment(compartment.name) is not None:
            raise ValueError(f"Compartment {compartment.name} already exists")
        self.compartments.append(compartment)
        logger.info(f"Added compartment '{compartment.name}' with volume {compartment.volume:.2e} m³")
    
    def refine(self, region_label: Optional[str] = None, criterion: Optional[str] = None) -> None:
        """
        Refine mesh (adaptively or uniformly).
        
        Args:
            region_label: Label of region to refine (None = uniform refinement)
            criterion: Refinement criterion ("gradient", "aspect_ratio", etc.)
        """
        logger.warning("Mesh refinement not yet implemented")
        # TODO: Implement adaptive mesh refinement
        pass
    
    def coarsen(self, region_label: str) -> None:
        """Coarsen mesh in specified region."""
        logger.warning("Mesh coarsening not yet implemented")
        # TODO: Implement mesh coarsening
        pass
    
    def project_field_to_mesh(
        self,
        field: np.ndarray,
        target_mesh: 'Mesh',
        method: str = "linear"
    ) -> np.ndarray:
        """
        Project field from this mesh to target mesh with mass conservation.
        
        Args:
            field: Field values on this mesh (N_nodes,)
            target_mesh: Target mesh
            method: Interpolation method ("linear", "nearest")
            
        Returns:
            Projected field on target mesh
        """
        if len(field) != len(self.nodes):
            raise ValueError("Field size must match number of nodes")
        
        # Simple nearest-neighbor projection for now
        target_field = np.zeros(len(target_mesh.nodes))
        
        for i, target_node in enumerate(target_mesh.nodes):
            # Find nearest source node
            distances = np.linalg.norm(self.nodes - target_node, axis=1)
            nearest_idx = np.argmin(distances)
            target_field[i] = field[nearest_idx]
        
        logger.debug(f"Projected field from {len(field)} to {len(target_field)} nodes")
        
        return target_field
    
    @classmethod
    def from_file(cls, filepath: str) -> 'Mesh':
        """
        Load mesh from file (XDMF/HDF5 or other formats via meshio).
        
        Args:
            filepath: Path to mesh file
            
        Returns:
            Mesh instance
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Mesh file not found: {filepath}")
        
        # Try to import meshio
        try:
            import meshio
        except ImportError:
            raise ImportError("meshio is required for mesh I/O: pip install meshio")
        
        # Load mesh
        mesh_data = meshio.read(filepath)
        
        # Extract nodes
        nodes = mesh_data.points
        
        # Extract elements (prefer tetrahedra, then triangles)
        elements = None
        element_type = None
        
        for cell_block in mesh_data.cells:
            if cell_block.type == "tetra":
                elements = cell_block.data
                element_type = "tetra"
                break
            elif cell_block.type == "triangle":
                elements = cell_block.data
                element_type = "tri"
        
        if elements is None:
            raise ValueError("No supported element types found in mesh file")
        
        logger.info(f"Loaded mesh from {filepath}: {len(nodes)} nodes, {len(elements)} {element_type} elements")
        
        return cls(nodes=nodes, elements=elements, element_type=element_type)
    
    def to_file(self, filepath: str) -> None:
        """
        Save mesh to file.
        
        Args:
            filepath: Output path
        """
        try:
            import meshio
        except ImportError:
            raise ImportError("meshio is required for mesh I/O")
        
        # Create meshio mesh object
        cells = [(self.element_type, self.elements)]
        mesh = meshio.Mesh(points=self.nodes, cells=cells)
        
        # Write
        meshio.write(filepath, mesh)
        logger.info(f"Mesh saved to {filepath}")
    
    @classmethod
    def create_unit_cube(cls, n: int = 10) -> 'Mesh':
        """
        Create a simple unit cube mesh for testing.
        
        Args:
            n: Number of nodes per dimension
            
        Returns:
            Mesh instance
        """
        # Create structured grid
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        z = np.linspace(0, 1, n)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        nodes = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        
        # Create tetrahedral elements (simplified - 5 tets per cube)
        elements = []
        nx, ny, nz = n, n, n
        
        def node_index(i, j, k):
            return i * ny * nz + j * nz + k
        
        for i in range(n - 1):
            for j in range(n - 1):
                for k in range(n - 1):
                    # Define 8 corners of cube
                    n0 = node_index(i, j, k)
                    n1 = node_index(i + 1, j, k)
                    n2 = node_index(i + 1, j + 1, k)
                    n3 = node_index(i, j + 1, k)
                    n4 = node_index(i, j, k + 1)
                    n5 = node_index(i + 1, j, k + 1)
                    n6 = node_index(i + 1, j + 1, k + 1)
                    n7 = node_index(i, j + 1, k + 1)
                    
                    # Split cube into 5 tetrahedra (Kuhn triangulation)
                    elements.extend([
                        [n0, n1, n2, n5],
                        [n0, n2, n3, n7],
                        [n0, n5, n2, n7],
                        [n5, n2, n7, n6],
                        [n0, n4, n5, n7]
                    ])
        
        elements = np.array(elements, dtype=np.int64)
        
        logger.info(f"Created unit cube mesh: {len(nodes)} nodes, {len(elements)} tetrahedra")
        
        return cls(nodes=nodes, elements=elements, element_type="tetra")
    
    @classmethod
    def create_sphere(cls, radius: float = 1.0, refinement: int = 2) -> 'Mesh':
        """
        Create a spherical mesh (icosphere subdivision).
        
        Args:
            radius: Sphere radius (meters)
            refinement: Refinement level (0-3)
            
        Returns:
            Mesh instance
        """
        # Start with icosahedron
        t = (1.0 + np.sqrt(5.0)) / 2.0
        
        vertices = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ], dtype=np.float64)
        
        # Normalize to sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices *= radius
        
        # Icosahedron faces
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ], dtype=np.int64)
        
        # Refine by subdivision
        for _ in range(refinement):
            vertices, faces = cls._subdivide_sphere(vertices, faces, radius)
        
        logger.info(f"Created spherical mesh: radius={radius}m, {len(vertices)} vertices, {len(faces)} faces")
        
        return cls(nodes=vertices, elements=faces, element_type="tri")
    
    @staticmethod
    def _subdivide_sphere(vertices: np.ndarray, faces: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Subdivide sphere faces."""
        new_vertices = list(vertices)
        new_faces = []
        edge_cache = {}
        
        def get_midpoint(v1_idx, v2_idx):
            key = tuple(sorted([v1_idx, v2_idx]))
            if key in edge_cache:
                return edge_cache[key]
            
            v1 = vertices[v1_idx]
            v2 = vertices[v2_idx]
            mid = (v1 + v2) / 2.0
            mid = mid / np.linalg.norm(mid) * radius  # Project to sphere
            
            new_idx = len(new_vertices)
            new_vertices.append(mid)
            edge_cache[key] = new_idx
            return new_idx
        
        for face in faces:
            v1, v2, v3 = face
            
            a = get_midpoint(v1, v2)
            b = get_midpoint(v2, v3)
            c = get_midpoint(v3, v1)
            
            new_faces.extend([
                [v1, a, c],
                [v2, b, a],
                [v3, c, b],
                [a, b, c]
            ])
        
        return np.array(new_vertices), np.array(new_faces, dtype=np.int64)
