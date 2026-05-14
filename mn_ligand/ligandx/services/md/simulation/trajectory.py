"""
Trajectory processing module for MD optimization.

All functions return JSON-serializable data (dicts/strings/lists).
No unserialized objects are passed between functions.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TrajectoryProcessor:
    """Handles trajectory processing for MD simulations with proper serialization."""
    
    @staticmethod
    def validate_trajectory_files(dcd_path: str, pdb_path: str) -> Dict[str, Any]:
        """
        Validate trajectory files exist and are accessible.
        
        Args:
            dcd_path: Path to DCD trajectory file
            pdb_path: Path to PDB topology file
            
        Returns:
            Dict with keys:
                - 'valid': bool
                - 'dcd_exists': bool
                - 'pdb_exists': bool
                - 'issues': list of str
        """
        import os
        
        issues = []
        dcd_exists = os.path.exists(dcd_path)
        pdb_exists = os.path.exists(pdb_path)
        
        if not dcd_exists:
            issues.append(f"DCD file not found: {dcd_path}")
        if not pdb_exists:
            issues.append(f"PDB file not found: {pdb_path}")
        
        return {
            'valid': len(issues) == 0,
            'dcd_exists': dcd_exists,
            'pdb_exists': pdb_exists,
            'issues': issues
        }
    
    @staticmethod
    def validate_processing_parameters(
        stride: int = 20,
        align: bool = True,
        remove_solvent: bool = True
    ) -> Dict[str, Any]:
        """
        Validate trajectory processing parameters.
        
        Args:
            stride: Frame sampling interval
            align: Whether to align frames
            remove_solvent: Whether to remove solvent
            
        Returns:
            Dict with keys:
                - 'valid': bool
                - 'parameters': dict
                - 'issues': list of str
        """
        issues = []
        
        if stride < 1:
            issues.append("stride must be >= 1")
        if stride > 10000:
            issues.append("stride is very large (> 10000)")
        
        return {
            'valid': len(issues) == 0,
            'parameters': {
                'stride': stride,
                'align': align,
                'remove_solvent': remove_solvent
            },
            'issues': issues
        }
    
    @staticmethod
    def get_trajectory_info(
        dcd_path: str,
        pdb_path: str
    ) -> Dict[str, Any]:
        """
        Get information about trajectory files.
        
        Args:
            dcd_path: Path to DCD file
            pdb_path: Path to PDB file
            
        Returns:
            Dict with trajectory information (JSON-serializable)
        """
        import os
        
        info = {
            'dcd_path': dcd_path,
            'pdb_path': pdb_path,
            'dcd_size_mb': 0,
            'pdb_size_mb': 0,
            'dcd_exists': False,
            'pdb_exists': False
        }
        
        try:
            if os.path.exists(dcd_path):
                info['dcd_size_mb'] = os.path.getsize(dcd_path) / (1024 * 1024)
                info['dcd_exists'] = True
            
            if os.path.exists(pdb_path):
                info['pdb_size_mb'] = os.path.getsize(pdb_path) / (1024 * 1024)
                info['pdb_exists'] = True
        except Exception as e:
            logger.warning(f"Could not get file sizes: {e}")
        
        return info
    
    @staticmethod
    def estimate_processing_time(
        frame_count: int,
        stride: int = 20,
        align: bool = True
    ) -> Dict[str, Any]:
        """
        Estimate trajectory processing time.
        
        Args:
            frame_count: Total frames in trajectory
            stride: Frame sampling interval
            align: Whether alignment is enabled
            
        Returns:
            Dict with time estimates (JSON-serializable)
        """
        sampled_frames = max(1, frame_count // stride)
        
        # Rough estimates (ms per frame)
        base_time_ms = sampled_frames * 10  # 10ms per frame for loading
        align_time_ms = sampled_frames * 50 if align else 0  # 50ms per frame for alignment
        total_ms = base_time_ms + align_time_ms
        
        return {
            'total_frames': frame_count,
            'sampled_frames': sampled_frames,
            'stride': stride,
            'estimated_ms': total_ms,
            'estimated_seconds': total_ms / 1000,
            'note': 'Rough estimate based on frame count'
        }
