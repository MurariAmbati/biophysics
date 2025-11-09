"""
Export utilities for simulation results.

Supports:
- SBML (Systems Biology Markup Language)
- JSON (JavaScript Object Notation)
- HDF5 (Hierarchical Data Format)
- CSV (Comma-Separated Values)
"""

import json
import numpy as np
from typing import Dict, Optional
import warnings

from ..config import SimulationResult
from ..circuit_graph import CircuitGraph


class ResultExporter:
    """Export simulation results to various formats."""
    
    @staticmethod
    def to_json(result: SimulationResult, filepath: str, indent: int = 2):
        """Export result to JSON format.
        
        Args:
            result: SimulationResult to export
            filepath: Output file path
            indent: JSON indentation
        """
        data = result.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        
        print(f"✓ Exported to JSON: {filepath}")
    
    @staticmethod
    def to_csv(result: SimulationResult, filepath: str):
        """Export result to CSV format.
        
        Args:
            result: SimulationResult to export
            filepath: Output file path
        """
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['time'] + result.species_names)
            
            # Data rows
            for i in range(len(result.times)):
                row = [result.times[i]] + list(result.concentrations[i, :])
                writer.writerow(row)
        
        print(f"✓ Exported to CSV: {filepath}")
    
    @staticmethod
    def to_hdf5(result: SimulationResult, filepath: str):
        """Export result to HDF5 format.
        
        Args:
            result: SimulationResult to export
            filepath: Output file path
        """
        try:
            import h5py
        except ImportError:
            print("HDF5 export requires h5py: pip install h5py")
            return
        
        with h5py.File(filepath, 'w') as f:
            # Create datasets
            f.create_dataset('times', data=result.times)
            f.create_dataset('concentrations', data=result.concentrations)
            
            # Add metadata as attributes
            f.attrs['species_names'] = result.species_names
            f.attrs['t_start'] = result.config.t_start
            f.attrs['t_end'] = result.config.t_end
            f.attrs['method'] = result.config.method
            f.attrs['seed'] = result.config.seed
            
            # Add metadata group
            meta = f.create_group('metadata')
            for key, value in result.metadata.items():
                if isinstance(value, (int, float, str)):
                    meta.attrs[key] = value
        
        print(f"✓ Exported to HDF5: {filepath}")
    
    @staticmethod
    def to_sbml(circuit: CircuitGraph, filepath: str, model_id: str = "synthetic_circuit"):
        """Export circuit to SBML format.
        
        Args:
            circuit: CircuitGraph to export
            filepath: Output file path
            model_id: SBML model identifier
        """
        # Simplified SBML export (Level 3 Version 2)
        sbml = f'''<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="{model_id}" name="Synthetic Biological Circuit">
    
    <!-- List of compartments -->
    <listOfCompartments>
      <compartment id="cell" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    
    <!-- List of species -->
    <listOfSpecies>
'''
        
        # Add species
        for species_name in circuit.get_species_list():
            sbml += f'      <species id="{species_name}" compartment="cell" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>\n'
        
        sbml += '''    </listOfSpecies>
    
    <!-- List of parameters -->
    <listOfParameters>
'''
        
        # Add parameters from nodes
        for node_id, node in circuit.nodes.items():
            for param_name, param_value in node.params.items():
                sbml += f'      <parameter id="{node_id}_{param_name}" value="{param_value}" constant="true"/>\n'
        
        sbml += '''    </listOfParameters>
    
    <!-- List of reactions -->
    <listOfReactions>
'''
        
        # Add reactions from edges
        for i, edge in enumerate(circuit.edges):
            reaction_id = f"reaction_{i}"
            sbml += f'      <reaction id="{reaction_id}" reversible="false">\n'
            sbml += f'        <annotation>\n'
            sbml += f'          <interaction type="{edge.interaction}" source="{edge.source}" target="{edge.target}"/>\n'
            sbml += f'        </annotation>\n'
            sbml += f'      </reaction>\n'
        
        sbml += '''    </listOfReactions>
    
  </model>
</sbml>'''
        
        with open(filepath, 'w') as f:
            f.write(sbml)
        
        print(f"✓ Exported to SBML: {filepath}")
        print(f"  Note: This is a simplified SBML export. Full kinetic laws require manual editing.")


def export_results_bundle(
    result: SimulationResult,
    circuit: CircuitGraph,
    output_dir: str,
    base_name: str = "simulation"
):
    """Export results in multiple formats.
    
    Args:
        result: SimulationResult to export
        circuit: CircuitGraph used
        output_dir: Output directory
        base_name: Base filename
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    exporter = ResultExporter()
    
    # Export in multiple formats
    exporter.to_json(result, os.path.join(output_dir, f"{base_name}.json"))
    exporter.to_csv(result, os.path.join(output_dir, f"{base_name}.csv"))
    
    # Optional HDF5
    try:
        exporter.to_hdf5(result, os.path.join(output_dir, f"{base_name}.h5"))
    except:
        print("  (HDF5 export skipped - h5py not available)")
    
    # SBML for circuit
    exporter.to_sbml(circuit, os.path.join(output_dir, f"{base_name}_circuit.sbml"))
    
    print(f"\n✓ Results bundle exported to: {output_dir}")
