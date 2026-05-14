"""
Chemistry utilities package.

This package provides molecular structure parsing, analysis, and preparation tools.

Submodules:
- parsers: PDB and mmCIF format parsing
- analysis: Structure component identification and analysis
- preparation: Protein and ligand preparation utilities
"""

from mn_ligand.ligandx.lib.chemistry.parsers import PDBParserUtils, MMCIFParserUtils
from mn_ligand.ligandx.lib.chemistry.parsers.pdb import get_pdb_parser, ResidueSelector
from mn_ligand.ligandx.lib.chemistry.parsers.mmcif import get_mmcif_parser
from mn_ligand.ligandx.lib.chemistry.analysis import ComponentAnalyzer, RESIDUE_CLASSIFICATIONS
from mn_ligand.ligandx.lib.chemistry.analysis.components import get_component_analyzer, AMINO_ACID_MAP, ATOMIC_MASSES
from mn_ligand.ligandx.lib.chemistry.preparation import ProteinPreparer, LigandPreparer
from mn_ligand.ligandx.lib.chemistry.preparation.protein import get_protein_preparer
from mn_ligand.ligandx.lib.chemistry.preparation.ligand import get_ligand_preparer

__all__ = [
    # Parsers
    'PDBParserUtils',
    'MMCIFParserUtils',
    'get_pdb_parser',
    'get_mmcif_parser',
    'ResidueSelector',
    # Analysis
    'ComponentAnalyzer',
    'get_component_analyzer',
    'RESIDUE_CLASSIFICATIONS',
    'AMINO_ACID_MAP',
    'ATOMIC_MASSES',
    # Preparation
    'ProteinPreparer',
    'LigandPreparer',
    'get_protein_preparer',
    'get_ligand_preparer',
]
