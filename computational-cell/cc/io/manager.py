"""
IO manager for checkpointing, output, and metadata tracking.
"""

import h5py
import yaml
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class IOManager:
    """
    Manages input/output operations, checkpointing, and metadata.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize IO manager.
        
        Args:
            output_dir: Output directory path
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"IOManager initialized: output_dir={self.output_dir}")
    
    def save_checkpoint(self, state, config) -> None:
        """
        Save checkpoint during simulation.
        
        Args:
            state: SimulationState object
            config: SimulationConfig object
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_t{state.t:.3f}_{timestamp}.h5"
        filepath = self.output_dir / filename
        
        self.save_checkpoint_to_file(state, config, str(filepath))
    
    def save_checkpoint_to_file(self, state, config, filepath: str) -> None:
        """
        Save checkpoint to specific file.
        
        Args:
            state: SimulationState object
            config: SimulationConfig object
            filepath: Target file path
        """
        with h5py.File(filepath, 'w') as f:
            # Metadata group
            meta = f.create_group('metadata')
            meta.attrs['version'] = '0.1.0'
            meta.attrs['timestamp'] = datetime.now().isoformat()
            meta.attrs['config_hash'] = config.hash()
            
            # Store config as JSON
            config_str = json.dumps(config.to_dict(), indent=2)
            meta.create_dataset('config', data=config_str)
            
            # Store additional metadata
            for key, value in state.metadata.items():
                if isinstance(value, (int, float, str)):
                    meta.attrs[key] = value
            
            # RNG state
            rng_group = f.create_group('rng_state')
            rng_group.create_dataset('global_rng', data=np.frombuffer(state.global_rng_state, dtype=np.uint8))
            
            # Simulation state
            state_group = f.create_group('state')
            state_group.attrs['t'] = state.t
            state_group.attrs['step'] = state.step
            
            # Module states
            modules_group = f.create_group('modules')
            for module_name, module_state in state.module_states.items():
                module_data = np.frombuffer(module_state, dtype=np.uint8)
                modules_group.create_dataset(module_name, data=module_data)
        
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint_from_file(self, filepath: str):
        """
        Load checkpoint from file.
        
        Args:
            filepath: Checkpoint file path
            
        Returns:
            SimulationState object
        """
        from cc.core.simulation import SimulationState
        
        with h5py.File(filepath, 'r') as f:
            # Load state
            state_group = f['state']
            t = state_group.attrs['t']
            step = state_group.attrs['step']
            
            # Load RNG state
            rng_data = f['rng_state/global_rng'][:]
            global_rng_state = rng_data.tobytes()
            
            # Load module states
            module_states = {}
            if 'modules' in f:
                for module_name in f['modules'].keys():
                    module_data = f[f'modules/{module_name}'][:]
                    module_states[module_name] = module_data.tobytes()
            
            # Load metadata
            metadata = {}
            if 'metadata' in f:
                for key in f['metadata'].attrs.keys():
                    metadata[key] = f['metadata'].attrs[key]
            
            state = SimulationState(
                t=t,
                step=step,
                global_rng_state=global_rng_state,
                module_states=module_states,
                metadata=metadata
            )
        
        logger.info(f"Checkpoint loaded from {filepath}: t={t:.3f}s, step={step}")
        return state
    
    def save_timeseries(
        self,
        filename: str,
        times: np.ndarray,
        data: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save time series data.
        
        Args:
            filename: Output filename
            times: Time points array
            data: Dict mapping variable names to arrays (n_times, ...)
            metadata: Optional metadata dict
        """
        filepath = self.output_dir / filename
        
        with h5py.File(filepath, 'w') as f:
            # Times
            f.create_dataset('times', data=times)
            
            # Data
            data_group = f.create_group('data')
            for name, values in data.items():
                data_group.create_dataset(name, data=values, compression='gzip')
            
            # Metadata
            if metadata:
                meta = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, (int, float, str)):
                        meta.attrs[key] = value
                    elif isinstance(value, dict):
                        meta.attrs[key] = json.dumps(value)
        
        logger.info(f"Time series saved to {filepath}")
    
    def load_timeseries(self, filename: str) -> Dict[str, Any]:
        """
        Load time series data.
        
        Args:
            filename: Input filename
            
        Returns:
            Dict with 'times', 'data', and 'metadata'
        """
        filepath = self.output_dir / filename
        
        with h5py.File(filepath, 'r') as f:
            times = f['times'][:]
            
            data = {}
            if 'data' in f:
                for name in f['data'].keys():
                    data[name] = f[f'data/{name}'][:]
            
            metadata = {}
            if 'metadata' in f:
                for key in f['metadata'].attrs.keys():
                    metadata[key] = f['metadata'].attrs[key]
        
        return {'times': times, 'data': data, 'metadata': metadata}
    
    def save_config(self, config, filename: str = "config.yaml") -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: SimulationConfig object
            filename: Output filename
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {filepath}")
    
    def load_config(self, filename: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            filename: Input filename
            
        Returns:
            Configuration dict
        """
        filepath = Path(filename)
        
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {filepath}")
        return config
