"""
Advanced circuit visualization with detailed diagrams.

Creates publication-quality circuit diagrams showing:
- Node types with custom symbols (promoters, genes, proteins, etc.)
- Regulatory interactions (activation/repression arrows)
- Real-time state coloring
- Network topology analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
import networkx as nx
from typing import Dict, List, Tuple, Optional

from ..circuit_graph import CircuitGraph, CircuitNode, CircuitEdge
from ..config import SimulationResult


class CircuitVisualizer:
    """Advanced circuit diagram visualization."""
    
    # Color scheme for node types
    NODE_COLORS = {
        'promoter': '#FFE5B4',      # Peach
        'gene': '#87CEEB',          # Sky blue
        'protein': '#90EE90',       # Light green
        'repressor': '#FFB6C1',     # Light pink
        'enzyme': '#DDA0DD',        # Plum
        'reporter': '#FFD700'       # Gold
    }
    
    # Symbols for node types
    NODE_SHAPES = {
        'promoter': 'triangle',
        'gene': 'rectangle',
        'protein': 'ellipse',
        'repressor': 'octagon',
        'enzyme': 'hexagon',
        'reporter': 'star'
    }
    
    # Edge styles
    EDGE_STYLES = {
        'activation': {'color': 'green', 'arrowstyle': '->', 'linewidth': 2},
        'repression': {'color': 'red', 'arrowstyle': '-|>', 'linewidth': 2},
        'production': {'color': 'blue', 'arrowstyle': '->', 'linewidth': 1.5},
        'degradation': {'color': 'gray', 'arrowstyle': '->', 'linewidth': 1, 'linestyle': 'dashed'},
        'binding': {'color': 'purple', 'arrowstyle': '<->', 'linewidth': 1.5}
    }
    
    def __init__(self, circuit: CircuitGraph):
        """Initialize visualizer with circuit.
        
        Args:
            circuit: CircuitGraph to visualize
        """
        self.circuit = circuit
        self._build_network()
    
    def _build_network(self):
        """Build NetworkX graph from circuit."""
        self.G = nx.DiGraph()
        
        # Add nodes
        for node_id, node in self.circuit.nodes.items():
            self.G.add_node(node_id, **{'node': node})
        
        # Add edges
        for edge in self.circuit.edges:
            self.G.add_edge(edge.source, edge.target, **{'edge': edge})
    
    def draw_circuit_diagram(
        self,
        figsize: Tuple[float, float] = (14, 10),
        layout: str = 'spring',
        show_labels: bool = True,
        show_parameters: bool = False,
        state: Optional[Dict[str, float]] = None
    ) -> plt.Figure:
        """Draw detailed circuit diagram.
        
        Args:
            figsize: Figure size
            layout: Layout algorithm ('spring', 'circular', 'hierarchical', 'shell')
            show_labels: Show node labels
            show_parameters: Show parameter values
            state: Optional dict of current concentrations for coloring
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        
        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(self.G, k=2, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(self.G)
        elif layout == 'hierarchical':
            pos = self._hierarchical_layout()
        elif layout == 'shell':
            pos = nx.shell_layout(self.G)
        else:
            pos = nx.spring_layout(self.G, seed=42)
        
        # Scale positions for better visualization
        pos = {k: (v[0] * 3, v[1] * 3) for k, v in pos.items()}
        
        # Draw edges first (behind nodes)
        self._draw_edges(ax, pos)
        
        # Draw nodes
        self._draw_nodes(ax, pos, state)
        
        # Draw labels
        if show_labels:
            self._draw_labels(ax, pos, show_parameters)
        
        # Add legend
        self._add_legend(ax)
        
        ax.axis('off')
        ax.set_title('Circuit Diagram', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def _draw_nodes(self, ax, pos, state=None):
        """Draw nodes with custom shapes and colors."""
        for node_id, (x, y) in pos.items():
            node = self.circuit.nodes[node_id]
            
            # Base color
            color = self.NODE_COLORS.get(node.type, '#CCCCCC')
            
            # Modulate color by concentration if state provided
            if state and node_id in state:
                conc = state[node_id]
                # Intensity based on concentration (log scale)
                if conc > 0:
                    intensity = min(np.log10(conc / 1e-10) / 5, 1.0)
                    intensity = max(intensity, 0.2)
                else:
                    intensity = 0.2
                # Blend with white based on intensity
                color = self._blend_color(color, intensity)
            
            # Draw shape based on node type
            shape = self.NODE_SHAPES.get(node.type, 'ellipse')
            
            if shape == 'triangle':
                points = np.array([[x, y+0.15], [x-0.13, y-0.15], [x+0.13, y-0.15]])
                patch = Polygon(points, facecolor=color, edgecolor='black', linewidth=2)
            elif shape == 'rectangle':
                patch = FancyBboxPatch((x-0.15, y-0.1), 0.3, 0.2, 
                                      boxstyle="round,pad=0.02", 
                                      facecolor=color, edgecolor='black', linewidth=2)
            elif shape == 'octagon':
                patch = mpatches.RegularPolygon((x, y), 8, radius=0.15, 
                                               facecolor=color, edgecolor='black', linewidth=2)
            elif shape == 'hexagon':
                patch = mpatches.RegularPolygon((x, y), 6, radius=0.15, 
                                               facecolor=color, edgecolor='black', linewidth=2)
            elif shape == 'star':
                # Star shape (simplified)
                patch = mpatches.RegularPolygon((x, y), 5, radius=0.15, 
                                               facecolor=color, edgecolor='black', linewidth=2)
            else:  # ellipse
                patch = Circle((x, y), 0.15, facecolor=color, edgecolor='black', linewidth=2)
            
            ax.add_patch(patch)
    
    def _draw_edges(self, ax, pos):
        """Draw edges with interaction-specific styles."""
        for edge in self.circuit.edges:
            if edge.source not in pos or edge.target not in pos:
                continue
            
            x1, y1 = pos[edge.source]
            x2, y2 = pos[edge.target]
            
            style = self.EDGE_STYLES.get(edge.interaction, 
                                        {'color': 'black', 'arrowstyle': '->', 'linewidth': 1})
            
            arrow = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle=style['arrowstyle'],
                color=style['color'],
                linewidth=style['linewidth'],
                linestyle=style.get('linestyle', 'solid'),
                mutation_scale=20,
                alpha=0.7,
                connectionstyle="arc3,rad=0.1",
                zorder=1
            )
            ax.add_patch(arrow)
            
            # Add Hill coefficient label if significant
            if edge.hill_coefficient > 1.5:
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x, mid_y, f'n={edge.hill_coefficient:.1f}', 
                       fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    def _draw_labels(self, ax, pos, show_params):
        """Draw node labels and optional parameters."""
        for node_id, (x, y) in pos.items():
            node = self.circuit.nodes[node_id]
            
            # Main label
            ax.text(x, y, node_id, ha='center', va='center', 
                   fontsize=9, fontweight='bold', zorder=10)
            
            # Parameter annotations
            if show_params and node.params:
                param_str = '\n'.join([f"{k}={v:.2e}" for k, v in list(node.params.items())[:2]])
                ax.text(x, y - 0.25, param_str, ha='center', va='top',
                       fontsize=6, style='italic', color='gray')
    
    def _add_legend(self, ax):
        """Add legend explaining symbols and interactions."""
        legend_elements = []
        
        # Node types
        for ntype, color in self.NODE_COLORS.items():
            legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black', 
                                                 label=ntype.capitalize()))
        
        legend_elements.append(mpatches.Patch(visible=False, label=''))  # Spacer
        
        # Edge types
        for etype, style in self.EDGE_STYLES.items():
            legend_elements.append(mpatches.Patch(visible=False, label=f"{etype.capitalize()} →",
                                                 color=style['color']))
        
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(1.02, 1), fontsize=9)
    
    def _hierarchical_layout(self) -> Dict:
        """Compute hierarchical layout based on circuit topology."""
        # Try to arrange in layers based on topology
        try:
            layers = list(nx.topological_generations(self.G))
            pos = {}
            for layer_idx, layer in enumerate(layers):
                n_nodes = len(layer)
                for node_idx, node_id in enumerate(layer):
                    x = (node_idx - n_nodes / 2) * 1.5
                    y = -layer_idx * 1.5
                    pos[node_id] = np.array([x, y])
            return pos
        except:
            # Fall back to spring if not DAG
            return nx.spring_layout(self.G, seed=42)
    
    def _blend_color(self, hex_color: str, intensity: float) -> str:
        """Blend color with white based on intensity."""
        # Convert hex to RGB
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Blend with white (increase brightness)
        r = int(r + (255 - r) * (1 - intensity))
        g = int(g + (255 - g) * (1 - intensity))
        b = int(b + (255 - b) * (1 - intensity))
        
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def animate_circuit(
        self,
        result: SimulationResult,
        output_file: str = 'circuit_animation.gif',
        fps: int = 10,
        layout: str = 'spring'
    ):
        """Create animated circuit with concentrations over time.
        
        Args:
            result: SimulationResult with trajectory
            output_file: Output filename
            fps: Frames per second
            layout: Layout algorithm
        """
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            print("Animation requires pillow: pip install pillow")
            return
        
        # Compute static layout
        if layout == 'spring':
            pos = nx.spring_layout(self.G, k=2, iterations=50, seed=42)
        else:
            pos = nx.circular_layout(self.G)
        
        pos = {k: (v[0] * 3, v[1] * 3) for k, v in pos.items()}
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Subsample time points for animation
        n_frames = min(len(result.times), fps * 10)
        indices = np.linspace(0, len(result.times) - 1, n_frames, dtype=int)
        
        def update(frame_idx):
            ax.clear()
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Get current state
            idx = indices[frame_idx]
            t = result.times[idx]
            state = {name: result.concentrations[idx, i] 
                    for i, name in enumerate(result.species_names)}
            
            # Draw circuit with current state
            self._draw_edges(ax, pos)
            self._draw_nodes(ax, pos, state)
            self._draw_labels(ax, pos, False)
            
            ax.set_title(f'Circuit Dynamics - t = {t:.1f}s', 
                        fontsize=14, fontweight='bold')
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-3.5, 3.5)
        
        anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps)
        
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
        print(f"Animation saved to: {output_file}")
        plt.close()


def plot_circuit_with_trajectories(
    circuit: CircuitGraph,
    result: SimulationResult,
    figsize: Tuple[float, float] = (16, 8)
) -> plt.Figure:
    """Combined circuit diagram and trajectory plot.
    
    Args:
        circuit: CircuitGraph
        result: SimulationResult
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], hspace=0.05, wspace=0.3)
    
    # Left: Circuit diagram
    ax_circuit = fig.add_subplot(gs[0])
    visualizer = CircuitVisualizer(circuit)
    
    # Get final state for coloring
    final_state = {name: result.concentrations[-1, i] 
                  for i, name in enumerate(result.species_names)}
    
    # Draw circuit (without creating new figure)
    pos = nx.spring_layout(visualizer.G, k=2, iterations=50, seed=42)
    pos = {k: (v[0] * 3, v[1] * 3) for k, v in pos.items()}
    
    visualizer._draw_edges(ax_circuit, pos)
    visualizer._draw_nodes(ax_circuit, pos, final_state)
    visualizer._draw_labels(ax_circuit, pos, False)
    visualizer._add_legend(ax_circuit)
    
    ax_circuit.axis('off')
    ax_circuit.set_aspect('equal')
    ax_circuit.set_title('Circuit Topology', fontsize=14, fontweight='bold')
    
    # Right: Time series
    ax_traces = fig.add_subplot(gs[1])
    
    for species in result.species_names:
        conc = result.get_species(species)
        ax_traces.plot(result.times, conc, label=species, linewidth=2)
    
    ax_traces.set_xlabel('Time (s)', fontsize=11)
    ax_traces.set_ylabel('Concentration (mol/L)', fontsize=11)
    ax_traces.legend(fontsize=9)
    ax_traces.grid(True, alpha=0.3)
    ax_traces.set_title('Concentration Dynamics', fontsize=14, fontweight='bold')
    
    fig.suptitle('Integrated Circuit Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    return fig
