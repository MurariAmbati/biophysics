"""
Diffusion-reaction PDE solver using finite element method.
"""

from typing import Dict, Optional, Callable, List
import numpy as np
import logging
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg, gmres

logger = logging.getLogger(__name__)


class DiffusionSolver:
    """
    Finite element diffusion-reaction solver.
    
    Solves: ∂c/∂t = D∇²c - R(c) + S(x,t)
    
    Implements the ModuleInterface protocol.
    """
    
    def __init__(
        self,
        mesh,
        config,
        method: str = "FEM",
        dt: float = 1e-3,
        time_scheme: str = "crank-nicolson"
    ):
        """
        Initialize diffusion solver.
        
        Args:
            mesh: Mesh object
            config: Simulation configuration
            method: "FEM" (finite element method)
            dt: Internal timestep for diffusion
            time_scheme: "backward-euler", "crank-nicolson"
        """
        self.name = "diffusion"
        self.mesh = mesh
        self.config = config
        self.method = method
        self.dt_diffusion = dt
        self.time_scheme = time_scheme
        
        # Species and their properties
        self.species: Dict[str, Dict] = {}  # name -> {D, field, bc}
        
        # FEM matrices (built on demand)
        self._mass_matrix: Optional[sparse.csr_matrix] = None
        self._stiffness_matrix: Optional[sparse.csr_matrix] = None
        self._matrices_built = False
        
        # State
        self._time = 0.0
        self._error_estimate = 0.0
        self._requested_dt = dt
        
        # Boundary conditions
        self.boundary_conditions: Dict[str, Callable] = {}
        
        logger.info(f"DiffusionSolver initialized: method={method}, dt={dt}s, scheme={time_scheme}")
    
    def add_species(
        self,
        name: str,
        diffusion_coefficient: float,
        initial_condition: Optional[np.ndarray] = None
    ) -> None:
        """
        Add a diffusing species.
        
        Args:
            name: Species name
            diffusion_coefficient: D (m²/s)
            initial_condition: Initial concentration field (N_nodes,)
        """
        if self.mesh is None:
            raise RuntimeError("Mesh must be set before adding species")
        
        n_nodes = len(self.mesh.nodes)
        
        if initial_condition is None:
            initial_condition = np.zeros(n_nodes, dtype=np.float64)
        
        if len(initial_condition) != n_nodes:
            raise ValueError(f"Initial condition size {len(initial_condition)} != n_nodes {n_nodes}")
        
        self.species[name] = {
            'D': diffusion_coefficient,
            'field': initial_condition.copy(),
            'reaction_rate': None,  # Optional reaction term
            'source': None  # Optional source term
        }
        
        logger.info(f"Added species '{name}': D={diffusion_coefficient:.2e} m²/s")
    
    def set_diffusion_coefficient(self, species: str, D: float) -> None:
        """Set diffusion coefficient for a species."""
        if species not in self.species:
            raise ValueError(f"Species {species} not found")
        self.species[species]['D'] = D
        # Invalidate matrices (will need rebuild with new D)
        self._matrices_built = False
    
    def apply_flux_boundary(self, label: str, flux_fn: Callable) -> None:
        """
        Apply flux boundary condition.
        
        Args:
            label: Boundary label
            flux_fn: Function(t, x) -> flux (mol·m⁻²·s⁻¹)
        """
        self.boundary_conditions[label] = flux_fn
        logger.debug(f"Applied flux boundary condition to '{label}'")
    
    def get_field(self, species: str) -> np.ndarray:
        """Get concentration field for species."""
        if species not in self.species:
            raise ValueError(f"Species {species} not found")
        return self.species[species]['field'].copy()
    
    def step(self, t: float, dt: float) -> None:
        """
        Advance diffusion equations by dt.
        
        Args:
            t: Current time
            dt: Timestep
        """
        # Build FEM matrices if needed
        if not self._matrices_built:
            self._build_fem_matrices()
        
        # Sub-step with internal dt if needed
        n_substeps = max(1, int(np.ceil(dt / self.dt_diffusion)))
        dt_sub = dt / n_substeps
        
        for _ in range(n_substeps):
            self._step_internal(t, dt_sub)
            t += dt_sub
        
        self._time = t
    
    def _step_internal(self, t: float, dt: float) -> None:
        """Internal timestep for diffusion."""
        for species_name, species_data in self.species.items():
            c_old = species_data['field']
            D = species_data['D']
            
            # Assemble system based on time scheme
            if self.time_scheme == "backward-euler":
                # (M + dt*K) c_new = M c_old
                A = self._mass_matrix + dt * D * self._stiffness_matrix
                b = self._mass_matrix @ c_old
                
            elif self.time_scheme == "crank-nicolson":
                # (M + 0.5*dt*K) c_new = (M - 0.5*dt*K) c_old
                A = self._mass_matrix + 0.5 * dt * D * self._stiffness_matrix
                b = (self._mass_matrix - 0.5 * dt * D * self._stiffness_matrix) @ c_old
            
            else:
                raise ValueError(f"Unknown time scheme: {self.time_scheme}")
            
            # Add reaction term if present
            if species_data['reaction_rate'] is not None:
                reaction = species_data['reaction_rate'](c_old)
                b -= dt * self._mass_matrix @ reaction
            
            # Add source term if present
            if species_data['source'] is not None:
                source = species_data['source'](t, self.mesh.nodes)
                b += dt * self._mass_matrix @ source
            
            # Solve linear system
            try:
                c_new = spsolve(A, b)
                
                # Clamp to non-negative
                c_new = np.maximum(c_new, 0.0)
                
                # Update field
                species_data['field'] = c_new
                
                # Estimate error
                error = np.linalg.norm(c_new - c_old) / (np.linalg.norm(c_old) + 1e-12)
                self._error_estimate = max(self._error_estimate, error)
                
            except Exception as e:
                logger.error(f"Linear solver failed for {species_name}: {e}")
                self._error_estimate = 1.0
    
    def _build_fem_matrices(self) -> None:
        """Build mass and stiffness matrices for FEM."""
        if self.mesh is None:
            raise RuntimeError("Cannot build FEM matrices without mesh")
        
        if self.mesh.element_type != "tetra":
            raise NotImplementedError("FEM currently only supports tetrahedral meshes")
        
        logger.info("Building FEM matrices...")
        
        n_nodes = len(self.mesh.nodes)
        n_elements = len(self.mesh.elements)
        
        # Preallocate matrix storage (COO format for assembly)
        max_entries = n_elements * 16  # 4x4 per element
        I_mass = np.zeros(max_entries, dtype=np.int32)
        J_mass = np.zeros(max_entries, dtype=np.int32)
        V_mass = np.zeros(max_entries, dtype=np.float64)
        
        I_stiff = np.zeros(max_entries, dtype=np.int32)
        J_stiff = np.zeros(max_entries, dtype=np.int32)
        V_stiff = np.zeros(max_entries, dtype=np.float64)
        
        entry_idx = 0
        
        # Assemble element-by-element
        for elem_idx, elem_nodes in enumerate(self.mesh.elements):
            # Get element node coordinates
            coords = self.mesh.nodes[elem_nodes]  # (4, 3)
            
            # Compute element volume
            v0, v1, v2, v3 = coords
            mat = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=1)
            vol = abs(np.linalg.det(mat)) / 6.0
            
            if vol <= 0:
                logger.warning(f"Element {elem_idx} has non-positive volume {vol}")
                continue
            
            # Mass matrix (linear elements): M_ij = vol/20 if i==j else vol/60
            M_elem = np.full((4, 4), vol / 60.0)
            np.fill_diagonal(M_elem, vol / 20.0)
            
            # Stiffness matrix: K_ij = ∫ ∇φ_i · ∇φ_j dV
            # For linear tetrahedra, gradients are constant
            K_elem = self._compute_element_stiffness(coords, vol)
            
            # Add to global matrices
            for i in range(4):
                for j in range(4):
                    I_mass[entry_idx] = elem_nodes[i]
                    J_mass[entry_idx] = elem_nodes[j]
                    V_mass[entry_idx] = M_elem[i, j]
                    
                    I_stiff[entry_idx] = elem_nodes[i]
                    J_stiff[entry_idx] = elem_nodes[j]
                    V_stiff[entry_idx] = K_elem[i, j]
                    
                    entry_idx += 1
        
        # Trim unused entries
        I_mass = I_mass[:entry_idx]
        J_mass = J_mass[:entry_idx]
        V_mass = V_mass[:entry_idx]
        
        I_stiff = I_stiff[:entry_idx]
        J_stiff = J_stiff[:entry_idx]
        V_stiff = V_stiff[:entry_idx]
        
        # Create sparse matrices
        self._mass_matrix = sparse.coo_matrix(
            (V_mass, (I_mass, J_mass)),
            shape=(n_nodes, n_nodes)
        ).tocsr()
        
        self._stiffness_matrix = sparse.coo_matrix(
            (V_stiff, (I_stiff, J_stiff)),
            shape=(n_nodes, n_nodes)
        ).tocsr()
        
        self._matrices_built = True
        
        logger.info(f"FEM matrices built: {n_nodes} nodes, {n_elements} elements")
        logger.info(f"Mass matrix: {self._mass_matrix.nnz} nonzeros")
        logger.info(f"Stiffness matrix: {self._stiffness_matrix.nnz} nonzeros")
    
    def _compute_element_stiffness(self, coords: np.ndarray, vol: float) -> np.ndarray:
        """
        Compute element stiffness matrix for linear tetrahedron.
        
        Args:
            coords: Element node coordinates (4, 3)
            vol: Element volume
            
        Returns:
            Element stiffness matrix (4, 4)
        """
        # Compute shape function gradients (constant over element)
        # For linear tet: ∇φ_i = (1/(6V)) * cross(edge_j, edge_k)
        
        v0, v1, v2, v3 = coords
        
        # Jacobian matrix
        J = np.array([
            v1 - v0,
            v2 - v0,
            v3 - v0
        ]).T  # (3, 3)
        
        J_inv = np.linalg.inv(J)
        
        # Shape function gradients in reference element
        grad_ref = np.array([
            [-1, -1, -1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])  # (4, 3)
        
        # Transform to physical element
        grad_phys = grad_ref @ J_inv.T  # (4, 3)
        
        # K_ij = vol * ∇φ_i · ∇φ_j
        K_elem = vol * (grad_phys @ grad_phys.T)
        
        return K_elem
    
    def get_state(self) -> bytes:
        """Serialize state."""
        import pickle
        state = {
            'time': self._time,
            'species_fields': {name: data['field'].copy() for name, data in self.species.items()}
        }
        return pickle.dumps(state)
    
    def set_state(self, state: bytes) -> None:
        """Restore state."""
        import pickle
        state_dict = pickle.loads(state)
        self._time = state_dict['time']
        for name, field in state_dict['species_fields'].items():
            if name in self.species:
                self.species[name]['field'] = field
    
    def get_requested_dt(self) -> float:
        """Return requested timestep."""
        return self._requested_dt
    
    def get_error_estimate(self) -> float:
        """Return error estimate."""
        return self._error_estimate
    
    def validate_state(self) -> bool:
        """Validate state."""
        for name, data in self.species.items():
            field = data['field']
            if np.any(np.isnan(field)) or np.any(np.isinf(field)):
                logger.error(f"Invalid values in {name} field")
                return False
            if np.any(field < -1e-6):
                logger.error(f"Negative concentrations in {name} field")
                return False
        return True
    
    def compute_mass(self, species: str) -> float:
        """
        Compute total mass of species in domain.
        
        Args:
            species: Species name
            
        Returns:
            Total mass (integrated concentration * volume)
        """
        if species not in self.species:
            raise ValueError(f"Species {species} not found")
        
        field = self.species[species]['field']
        
        # Mass = ∫ c dV ≈ Σ M_ij c_j (using mass matrix)
        mass_vector = self._mass_matrix @ field
        total_mass = np.sum(mass_vector)
        
        return total_mass
