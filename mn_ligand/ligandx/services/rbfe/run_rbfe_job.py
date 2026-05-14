#!/usr/bin/env python
"""
RBFE Calculation Service Entrypoint

This script runs RBFE (Relative Binding Free Energy) calculation jobs.
It accepts JSON input and returns JSON output.

Usage:
    python run_rbfe_job.py < input.json > output.json
    python run_rbfe_job.py --input input.json --output output.json
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Configure logging to write to file
log_file = Path('/tmp/rbfe.log')
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),  # Log to stderr (won't interfere with JSON output)
        logging.FileHandler(str(log_file), mode='a')  # Also log to file
    ]
)

from mn_ligand.ligandx.services.rbfe.service import RBFEService


def validate_input_data(input_data: dict) -> tuple[bool, str]:
    """
    Validate input data before processing.
    
    Args:
        input_data: Input dictionary from JSON
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not input_data:
        return False, "Input data is empty"
    
    if not isinstance(input_data, dict):
        return False, "Input data must be a dictionary"
    
    if not input_data.get('protein_pdb_data'):
        return False, "Missing required field: protein_pdb_data"
    
    if not input_data.get('protein_pdb_data', '').strip():
        return False, "protein_pdb_data cannot be empty"
    
    if not input_data.get('ligands'):
        return False, "Missing required field: ligands"
    
    if not isinstance(input_data.get('ligands'), list) or len(input_data.get('ligands', [])) == 0:
        return False, "ligands must be a non-empty list"
    
    return True, ""


def main():
    parser = argparse.ArgumentParser(description='Run RBFE calculation job')
    parser.add_argument('--input', type=str, help='Input JSON file')
    parser.add_argument('--output', type=str, help='Output JSON file')
    args = parser.parse_args()
    
    # Read input
    if args.input:
        with open(args.input, 'r') as f:
            input_data = json.load(f)
    else:
        # Read from stdin
        input_data = json.load(sys.stdin)
    
    try:
        # Validate input data
        valid, error = validate_input_data(input_data)
        if not valid:
            output = {
                'success': False,
                'error': error
            }
        else:
            # Initialize service
            service = RBFEService()
            
            # Extract parameters
            job_id = input_data.get('job_id', 'rbfe_job')
            protein_pdb = input_data.get('protein_pdb_data', '')
            ligands_data = input_data.get('ligands', [])
            network_topology = input_data.get('network_topology') or input_data.get('network', {}).get('topology', 'mst')
            central_ligand = input_data.get('central_ligand') or input_data.get('network', {}).get('central_ligand')
            atom_mapper = input_data.get('atom_mapper', 'kartograf')  # Atom mapper for alignment
            simulation_settings = input_data.get('protocol_settings') or input_data.get('simulation_settings')
            protein_id = input_data.get('protein_id', 'protein')
            atom_map_hydrogens = input_data.get('atom_map_hydrogens', True)
            lomap_max3d = input_data.get('lomap_max3d', 1.0)

            # Run RBFE calculation using atom mapper-based alignment
            result = service.run_rbfe_calculation(
                protein_pdb=protein_pdb,
                ligands_data=ligands_data,
                job_id=job_id,
                network_topology=network_topology,
                central_ligand_name=central_ligand,
                atom_mapper=atom_mapper,
                atom_map_hydrogens=atom_map_hydrogens,
                lomap_max3d=lomap_max3d,
                simulation_settings=simulation_settings,
                protein_id=protein_id
            )
            
            # Prepare output
            output = {
                'success': result.get('status') == 'completed',
                'result': result
            }
        
    except Exception as e:
        import traceback
        output = {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
    # Helper to make objects JSON-serializable (handles numpy types, etc.)
    def make_serializable(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'm'):  # OpenFF/OpenMM unit quantities
            return float(obj.m)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    # Write output - wrap in try/except to catch serialization errors
    try:
        serializable_output = make_serializable(output)
        output_json = json.dumps(serializable_output, indent=2)
    except Exception as json_err:
        output_json = json.dumps({
            'success': False,
            'error': f'Failed to serialize output: {str(json_err)}',
            'original_status': str(output.get('result', {}).get('status', 'unknown'))
        }, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
    else:
        print(output_json, flush=True)


if __name__ == '__main__':
    main()
