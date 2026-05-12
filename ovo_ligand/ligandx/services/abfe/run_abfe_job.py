#!/usr/bin/env python
"""
ABFE Calculation Service Entrypoint

This script runs ABFE (Absolute Binding Free Energy) calculation jobs.
It accepts JSON input and returns JSON output.

Usage:
    python run_abfe_job.py < input.json > output.json
    python run_abfe_job.py --input input.json --output output.json
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Configure logging to write to file
log_file = Path('/tmp/abfe.log')
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),  # Log to stderr (won't interfere with JSON output)
        logging.FileHandler(str(log_file), mode='a')  # Also log to file
    ]
)

from ovo_ligand.ligandx.services.abfe.service import ABFEService


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
    
    if not input_data.get('ligand_sdf_data'):
        return False, "Missing required field: ligand_sdf_data"
    
    if not input_data.get('ligand_sdf_data', '').strip():
        return False, "ligand_sdf_data cannot be empty"
    
    return True, ""


def main():
    parser = argparse.ArgumentParser(description='Run ABFE calculation job')
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
            service = ABFEService()
            
            # Extract parameters
            job_id = input_data.get('job_id', 'abfe_job')
            protein_pdb = input_data.get('protein_pdb_data', '')
            ligand_sdf = input_data.get('ligand_sdf_data', '')
            ligand_id = input_data.get('ligand_id', input_data.get('ligand_name', 'ligand'))
            protein_id = input_data.get('protein_id', 'protein')
            simulation_settings = input_data.get('protocol_settings')
            
            # Run ABFE calculation
            result = service.run_abfe_calculation(
                protein_pdb=protein_pdb,
                ligand_sdf=ligand_sdf,
                job_id=job_id,
                simulation_settings=simulation_settings,
                ligand_id=ligand_id,
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
            # Fallback: convert to string
            return str(obj)
    
    # Write output - wrap in try/except to catch serialization errors
    try:
        serializable_output = make_serializable(output)
        output_json = json.dumps(serializable_output, indent=2)
    except Exception as json_err:
        # If serialization fails, return a simple error message
        output_json = json.dumps({
            'success': False,
            'error': f'Failed to serialize output: {str(json_err)}',
            'original_status': str(output.get('result', {}).get('status', 'unknown'))
        }, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
    else:
        print(output_json, flush=True)  # flush=True ensures output is written immediately


if __name__ == '__main__':
    main()
