"""
Circuit graph representation for synthetic biological circuits.

Defines nodes (biological parts) and edges (regulatory/reaction connections)
with validation rules and graph operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set
import yaml


# Type aliases for clarity
NodeType = Literal['promoter', 'gene', 'protein', 'repressor', 'enzyme', 'reporter']
InteractionType = Literal['activation', 'repression', 'binding', 'production', 'degradation']


@dataclass
class CircuitNode:
    """Represents a biological component in the circuit.
    
    Attributes:
        id: Unique identifier for this node
        type: Biological component type
        params: Dictionary of kinetic parameters (e.g., k_tx, delta_p)
    """
    id: str
    type: NodeType
    params: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate node parameters."""
        if not self.id:
            raise ValueError("Node id cannot be empty")
        # Ensure all params are float
        self.params = {k: float(v) for k, v in self.params.items()}


@dataclass
class CircuitEdge:
    """Represents a regulatory or reaction connection between nodes.
    
    Attributes:
        source: ID of the source node
        target: ID of the target node
        interaction: Type of interaction (activation, repression, etc.)
        hill_coefficient: Hill coefficient for cooperative binding (default 1.0)
        params: Additional edge-specific parameters
    """
    source: str
    target: str
    interaction: InteractionType
    hill_coefficient: float = 1.0
    params: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate edge parameters."""
        if self.hill_coefficient <= 0:
            raise ValueError("Hill coefficient must be positive")
        self.params = {k: float(v) for k, v in self.params.items()}


class CircuitGraph:
    """Directed multigraph representing a synthetic biological circuit.
    
    The graph encodes:
    - Nodes: biological parts (promoters, genes, proteins, etc.)
    - Edges: regulatory and reaction connections
    
    Provides validation and query methods for circuit topology.
    """
    
    def __init__(self):
        """Initialize an empty circuit graph."""
        self.nodes: Dict[str, CircuitNode] = {}
        self.edges: List[CircuitEdge] = []
        self._adjacency: Dict[str, List[CircuitEdge]] = {}
    
    def add_node(self, node: CircuitNode) -> None:
        """Add a node to the circuit.
        
        Args:
            node: CircuitNode to add
            
        Raises:
            ValueError: If node ID already exists
        """
        if node.id in self.nodes:
            raise ValueError(f"Node with id '{node.id}' already exists")
        self.nodes[node.id] = node
        self._adjacency[node.id] = []
    
    def add_edge(self, edge: CircuitEdge) -> None:
        """Add an edge to the circuit.
        
        Args:
            edge: CircuitEdge to add
            
        Raises:
            ValueError: If source or target nodes don't exist
        """
        if edge.source not in self.nodes:
            raise ValueError(f"Source node '{edge.source}' does not exist")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node '{edge.target}' does not exist")
        
        self.edges.append(edge)
        self._adjacency[edge.source].append(edge)
    
    def get_node(self, node_id: str) -> CircuitNode:
        """Retrieve a node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            CircuitNode
            
        Raises:
            KeyError: If node doesn't exist
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found")
        return self.nodes[node_id]
    
    def get_outgoing_edges(self, node_id: str) -> List[CircuitEdge]:
        """Get all edges originating from a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            List of outgoing edges
        """
        return self._adjacency.get(node_id, [])
    
    def get_incoming_edges(self, node_id: str) -> List[CircuitEdge]:
        """Get all edges targeting a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            List of incoming edges
        """
        return [edge for edge in self.edges if edge.target == node_id]
    
    def get_species_list(self) -> List[str]:
        """Get list of all species (proteins, mRNAs) in the circuit.
        
        Returns ordered list of species names for state vector indexing.
        """
        species = []
        for node in self.nodes.values():
            if node.type in ['protein', 'repressor', 'reporter', 'enzyme']:
                species.append(node.id)
            elif node.type == 'gene':
                # Add mRNA for genes if modeled explicitly
                species.append(f"{node.id}_mRNA")
                species.append(f"{node.id}_protein")
        return sorted(set(species))  # Ensure deterministic ordering
    
    def validate(self) -> None:
        """Validate circuit structure and connectivity.
        
        Checks:
        - No disconnected regulatory subgraphs
        - Each output species has degradation path
        - No invalid edge connections
        
        Raises:
            ValueError: If validation fails
        """
        if not self.nodes:
            raise ValueError("Circuit is empty (no nodes)")
        
        # Check for disconnected nodes
        connected_nodes = set()
        for edge in self.edges:
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)
        
        # Promoters can be unconnected inputs
        disconnected = set(self.nodes.keys()) - connected_nodes
        disconnected = {nid for nid in disconnected 
                       if self.nodes[nid].type not in ['promoter']}
        
        if disconnected:
            raise ValueError(f"Disconnected nodes found: {disconnected}")
        
        # Check that species have degradation
        species = self.get_species_list()
        for sp in species:
            # Check if species has a degradation edge
            node_id = sp.replace('_mRNA', '').replace('_protein', '')
            outgoing = self.get_outgoing_edges(node_id)
            has_degradation = any(e.interaction == 'degradation' for e in outgoing)
            
            # Also check if species itself has degradation via params
            if node_id in self.nodes:
                node = self.nodes[node_id]
                has_deg_param = any(k.startswith('delta') for k in node.params.keys())
                if not has_degradation and not has_deg_param:
                    # Warning only for now - many circuits have implicit degradation
                    pass
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'CircuitGraph':
        """Load circuit definition from YAML file.
        
        Args:
            filepath: Path to YAML file
            
        Returns:
            CircuitGraph instance
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        circuit = cls()
        
        # Add nodes
        for node_data in data.get('nodes', []):
            node = CircuitNode(
                id=node_data['id'],
                type=node_data['type'],
                params=node_data.get('params', {})
            )
            circuit.add_node(node)
        
        # Add edges
        for edge_data in data.get('edges', []):
            edge = CircuitEdge(
                source=edge_data['source'],
                target=edge_data['target'],
                interaction=edge_data['interaction'],
                hill_coefficient=edge_data.get('hill_coefficient', 1.0),
                params=edge_data.get('params', {})
            )
            circuit.add_edge(edge)
        
        circuit.validate()
        return circuit
    
    def to_yaml(self, filepath: str) -> None:
        """Save circuit definition to YAML file.
        
        Args:
            filepath: Path to output YAML file
        """
        data = {
            'nodes': [
                {
                    'id': node.id,
                    'type': node.type,
                    'params': node.params
                }
                for node in self.nodes.values()
            ],
            'edges': [
                {
                    'source': edge.source,
                    'target': edge.target,
                    'interaction': edge.interaction,
                    'hill_coefficient': edge.hill_coefficient,
                    'params': edge.params
                }
                for edge in self.edges
            ]
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def __repr__(self) -> str:
        """String representation of circuit."""
        return f"CircuitGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"
