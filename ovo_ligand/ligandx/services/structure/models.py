"""Pydantic models for Structure Service."""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class SMILESRequest(BaseModel):
    """SMILES conversion request."""
    smiles: str = Field(..., description="SMILES string")


class SMILES3DResponse(BaseModel):
    """SMILES to 3D response."""
    sdf_data: str
    pdb_data: str
    format: str = "sdf"


class SMILESMolResponse(BaseModel):
    """SMILES to Molfile response."""
    molfile: str


class UploadSMILESRequest(BaseModel):
    """Upload SMILES request."""
    smiles: str
    name: Optional[str] = None


class FetchPDBRequest(BaseModel):
    """Fetch PDB request."""
    pdb_id: str


class ProcessPDBRequest(BaseModel):
    """Process PDB request."""
    pdb_id: Optional[str] = None
    pdb_data: Optional[str] = None
    structure_id: Optional[str] = None
    clean_protein: bool = True
    include_2d_images: bool = True


class UploadStructureRequest(BaseModel):
    """Upload structure file request (handled as multipart)."""
    pass  # File upload handled separately


class DownloadSDFRequest(BaseModel):
    """Download SDF request."""
    pdb_data: str
    generate_conformers: bool = False
    num_conformers: int = 10


class MoleculeModel(BaseModel):
    """Molecule data model."""
    id: Optional[int] = None
    name: str
    original_name: Optional[str] = None  # Original ligand name (e.g. residue name from PDB)
    smiles: Optional[str] = None
    canonical_smiles: Optional[str] = None
    molfile: Optional[str] = None
    inchi: Optional[str] = None
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    num_atoms: Optional[int] = None
    num_bonds: Optional[int] = None
    source: Optional[str] = None


class SaveMoleculeRequest(BaseModel):
    """Save molecule to library request."""
    name: str = "Untitled Molecule"
    original_name: Optional[str] = None  # Original ligand name (e.g. residue name from PDB)
    smiles: Optional[str] = None
    molfile: Optional[str] = None
    inchi: Optional[str] = None
    canonical_smiles: Optional[str] = None
    source: str = "editor"


class SaveStructureRequest(BaseModel):
    """Save structure to library request."""
    pdb_data: str
    name: str = "Untitled Structure"


class ExtractLigandRequest(BaseModel):
    """Extract ligand from complex request."""
    pdb_data: str


class GetLigandStructureRequest(BaseModel):
    """Get ligand structure request."""
    structure_id: str
    ligand_id: str
    pdb_data: Optional[str] = None


class SaveEditedMoleculeRequest(BaseModel):
    """Save edited molecule request."""
    molfile: str
    name: str
    smiles: Optional[str] = None
    original_ligand_id: Optional[str] = None
    structure_id: Optional[str] = None


class CombineProteinLigandRequest(BaseModel):
    """Combine protein and ligand request."""
    protein_data: Optional[str] = None
    protein_pdb: Optional[str] = None
    ligand_data: Optional[str] = None
    ligand_pdb: Optional[str] = None


class CleanProteinStagedRequest(BaseModel):
    """Clean protein with staged control request."""
    pdb_data: str
    remove_heterogens: bool = True
    remove_water: bool = True
    add_missing_residues: bool = True
    add_missing_atoms: bool = True
    add_missing_hydrogens: bool = True
    ph: float = 7.4
    add_solvation: bool = False
    solvation_box_size: float = 10.0
    solvation_box_shape: str = 'cubic'
    keep_ligands: bool = False


class CleanProteinStagedResponse(BaseModel):
    """Clean protein staged response."""
    stages: Dict[str, str]  # stage_name -> pdb_data
    stage_info: Dict[str, Any]  # Metadata for each stage
    ligands: Optional[Dict[str, Any]] = None  # Preserved ligands when keep_ligands=True


class FetchHETIDRequest(BaseModel):
    """Fetch structure from PDB database containing a specific HET ID."""
    het_id: str


class ExtractLigandByHETIDRequest(BaseModel):
    """Extract ligand by HET ID from protein structure."""
    pdb_data: str
    het_id: str
    ligand_name: Optional[str] = None



