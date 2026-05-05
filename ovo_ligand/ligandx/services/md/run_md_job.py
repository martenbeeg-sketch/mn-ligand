#!/usr/bin/env python
"""
MD Optimization Service Entrypoint

This script runs MD optimization jobs in the biochem-md environment.
It accepts JSON input and returns JSON output.

Usage:
    python run_md_job.py < input.json > output.json
    python run_md_job.py --input input.json --output output.json
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Configure logging to write to file
log_file = Path('/tmp/md.log')
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),  # Log to stderr (won't interfere with JSON output)
        logging.FileHandler(str(log_file), mode='a')  # Also log to file
    ]
)

from ovo_ligand.ligandx.services.md.service import MDOptimizationService
from ovo_ligand.ligandx.services.md.config import MDOptimizationConfig


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
    
    return True, ""


def main():
    parser = argparse.ArgumentParser(description='Run MD optimization job')
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
            # Create config from input data
            config = MDOptimizationConfig.from_dict(input_data)
            
            # Validate config
            valid, error = config.validate()
            if not valid:
                output = {
                    'success': False,
                    'error': error
                }
            else:
                # Initialize service and run optimization
                service = MDOptimizationService(job_id=config.job_id)
                result = service.optimize(config)
                
                # Prepare output
                # Success if status is not 'error' (includes 'success', 'preview_ready', 'minimized_ready')
                output = {
                    'success': result.get('status') != 'error',
                    'result': result
                }
        
    except Exception as e:
        import traceback
        output = {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
    # Write output
    output_json = json.dumps(output, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
    else:
        print(output_json)


if __name__ == '__main__':
    main()

