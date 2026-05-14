"""
Environment validation module for MD optimization.

Validates availability of OpenMM, OpenFF, RDKit, and other dependencies.
All functions return JSON-serializable data.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class EnvironmentValidator:
    """Validates MD simulation environment and dependencies."""
    
    # Valid element symbols for PDB sanitization
    VALID_ELEMENTS = {
        "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG", "AL", "SI", "P", "S", "CL", "AR",
        "K", "CA", "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "KR",
        "RB", "SR", "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN", "SB", "TE", "I", "XE",
        "CS", "BA", "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU",
        "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG", "TL", "PB", "BI", "PO", "AT", "RN",
        "FR", "RA", "AC", "TH", "PA", "U", "NP", "PU", "AM", "CM", "BK", "CF", "ES", "FM", "MD", "NO", "LR",
        "RF", "DB", "SG", "BH", "HS", "MT", "DS", "RG", "CN", "FL", "LV", "TS", "OG",
        "D", "T"  # common hydrogen isotopes
    }
    
    @staticmethod
    def validate_environment() -> Dict[str, Any]:
        """
        Validate environment and provide clear diagnostics.
        
        Returns:
            Dict with availability status for each dependency (JSON-serializable)
        """
        logger.info("=== ENVIRONMENT VALIDATION ===")
        status = {}
        
        # Check OpenFF Toolkit
        try:
            from openff.toolkit import Molecule, ForceField
            logger.info("[COMPLETE] OpenFF Toolkit available")
            status['openff'] = True
        except ImportError as e:
            logger.error(f"[ERROR] OpenFF Toolkit not available: {e}")
            status['openff'] = False
        
        # Check AmberTools
        try:
            from openff.toolkit.utils.toolkits import AmberToolsToolkitWrapper
            if AmberToolsToolkitWrapper.is_available():
                logger.info("[COMPLETE] AmberToolsToolkitWrapper available")
                logger.info("  AM1-BCC charges available via OpenFE charge_generation")
                status['ambertools'] = True
            else:
                logger.info("[COMPLETE] AmberToolsToolkitWrapper not available")
                logger.info("  AM1-BCC charges will not be available")
                status['ambertools'] = False
        except ImportError:
            logger.info("[COMPLETE] AmberTools not available")
            logger.info("  AM1-BCC charges will not be available")
            status['ambertools'] = False
        
        # Check OpenMM
        try:
            import openmm
            from openmm import Platform
            logger.info("[COMPLETE] OpenMM available")
            status['openmm'] = True
            
            # Check available platforms
            platforms = []
            try:
                Platform.getPlatformByName('CUDA')
                platforms.append('CUDA')
                logger.info("[COMPLETE] CUDA platform available")
            except Exception as e:
                logger.info(f"⚠ CUDA platform not available: {e}")
            
            try:
                Platform.getPlatformByName('OpenCL')
                platforms.append('OpenCL')
                logger.info("[COMPLETE] OpenCL platform available")
            except Exception as e:
                logger.info(f"⚠ OpenCL platform not available: {e}")
            
            # CPU is always available
            platforms.append('CPU')
            logger.info(f"[COMPLETE] Available OpenMM platforms: {', '.join(platforms)}")
            status['openmm_platforms'] = platforms
            
        except ImportError:
            logger.error("[ERROR] OpenMM not available")
            status['openmm'] = False
            status['openmm_platforms'] = []
        
        # Check PDBFixer
        try:
            from pdbfixer import PDBFixer
            logger.info("[COMPLETE] PDBFixer available")
            status['pdbfixer'] = True
        except ImportError:
            logger.error("[ERROR] PDBFixer not available")
            status['pdbfixer'] = False
        
        # Check RDKit
        try:
            from rdkit import Chem
            logger.info("[COMPLETE] RDKit available")
            status['rdkit'] = True
        except ImportError:
            logger.error("[ERROR] RDKit not available")
            status['rdkit'] = False
        
        # Check MDTraj
        try:
            import mdtraj
            logger.info("[COMPLETE] MDTraj available")
            status['mdtraj'] = True
        except ImportError:
            logger.warning("⚠ MDTraj not available (trajectory processing disabled)")
            status['mdtraj'] = False
        
        # Check openmmforcefields
        try:
            from openmmforcefields.generators import SMIRNOFFTemplateGenerator
            logger.info("[COMPLETE] openmmforcefields available")
            status['openmmforcefields'] = True
        except ImportError:
            logger.warning("⚠ openmmforcefields not available")
            status['openmmforcefields'] = False
        
        logger.info("=== VALIDATION COMPLETE ===")
        
        return status
    
    @staticmethod
    def get_best_platform() -> str:
        """
        Get the best available OpenMM platform.
        
        Returns:
            Platform name ('CUDA', 'OpenCL', or 'CPU')
        """
        try:
            from openmm import Platform
            
            # Try CUDA first
            try:
                Platform.getPlatformByName('CUDA')
                return 'CUDA'
            except Exception:
                pass
            
            # Try OpenCL
            try:
                Platform.getPlatformByName('OpenCL')
                return 'OpenCL'
            except Exception:
                pass
            
            # Fallback to CPU
            return 'CPU'
            
        except ImportError:
            return 'CPU'
    
    @staticmethod
    def check_minimum_requirements() -> Dict[str, Any]:
        """
        Check if minimum requirements for MD simulation are met.
        
        Returns:
            Dict with 'met': bool and 'missing': list of missing dependencies
        """
        status = EnvironmentValidator.validate_environment()
        
        required = ['openff', 'openmm', 'rdkit', 'pdbfixer']
        missing = [dep for dep in required if not status.get(dep, False)]
        
        return {
            'met': len(missing) == 0,
            'missing': missing,
            'status': status
        }
