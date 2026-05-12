"""
RBFE Network Planner
Handles ligand network topology planning for relative binding free energy calculations.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Optional, List, Tuple  # noqa: F401 – Tuple used in generate_edge_svg
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# OpenFE and dependencies
try:
    import openfe
    from openfe import LomapAtomMapper
    try:
        from kartograf import KartografAtomMapper
        KARTOGRAF_AVAILABLE = True
    except ImportError:
        KARTOGRAF_AVAILABLE = False
        KartografAtomMapper = None
    from openfe.setup.ligand_network_planning import (
        generate_minimal_spanning_network,
        generate_radial_network,
        generate_maximal_network,
    )
    from openfe.setup import lomap_scorers
    OPENFE_AVAILABLE = True
except ImportError:
    OPENFE_AVAILABLE = False
    openfe = None
    LomapAtomMapper = None
    logger.warning("OpenFE not available. RBFE network planning will not work.")


@dataclass
class NetworkEdge:
    """Represents an edge in the ligand network."""
    ligand_a: str
    ligand_b: str
    score: float = 0.0
    mapping_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LigandNetworkData:
    """Data structure for ligand network information."""
    nodes: List[str]  # Ligand names
    edges: List[NetworkEdge]
    topology: str  # 'mst', 'radial', 'maximal'
    central_ligand: Optional[str] = None  # For radial networks


class NetworkPlanner:
    """Plans ligand networks for RBFE calculations using OpenFE atom mappers."""

    def __init__(
        self,
        atom_mapper: str = 'kartograf',
        atom_map_hydrogens: bool = True,
        lomap_max3d: float = 1.0
    ):
        """
        Initialize network planner with user-selected atom mapper.

        Following OpenFE best practices, the atom mapper is used as the PRIMARY
        strategy for network creation. The mapper handles both atom mapping AND
        alignment simultaneously - no pre-alignment is needed.

        Args:
            atom_mapper: Primary atom mapper to use ('kartograf', 'lomap', 'lomap_relaxed')
                - 'kartograf': Geometry-based, preserves 3D binding mode (RECOMMENDED for docked poses)
                - 'lomap': 2D MCS-based, strict settings (max3d=1.0, no element change)
                - 'lomap_relaxed': 2D MCS-based, relaxed settings (max3d=5.0, allow element change)
            atom_map_hydrogens: For Kartograf - include hydrogens in mapping (default: True)
            lomap_max3d: For LOMAP - maximum 3D distance for mapping (default: 1.0)

        References:
            - Kartograf paper: https://pubs.acs.org/doi/10.1021/acs.jctc.3c01206
            - OpenFE RBFE tutorial: https://docs.openfree.energy/en/latest/tutorials/rbfe_cli_tutorial.html
        """
        if not OPENFE_AVAILABLE:
            raise ImportError("OpenFE is not available. Please install openfe package.")

        self.atom_mapper_type = atom_mapper
        self.atom_map_hydrogens = atom_map_hydrogens
        self.lomap_max3d = lomap_max3d

        # Create PRIMARY mapper based on user selection
        if atom_mapper == 'kartograf':
            if not KARTOGRAF_AVAILABLE:
                raise ImportError(
                    "Kartograf selected but not available. "
                    "Please install kartograf package or choose 'lomap' mapper."
                )
            self.primary_mapper = KartografAtomMapper(atom_map_hydrogens=atom_map_hydrogens)
            logger.info(f"Primary mapper: Kartograf (atom_map_hydrogens={atom_map_hydrogens})")

        elif atom_mapper == 'lomap':
            self.primary_mapper = LomapAtomMapper(
                max3d=lomap_max3d,
                element_change=False,
                time=20,
            )
            logger.info(f"Primary mapper: LOMAP strict (max3d={lomap_max3d})")

        elif atom_mapper == 'lomap_relaxed':
            self.primary_mapper = LomapAtomMapper(
                max3d=5.0,
                element_change=True,
                time=30,
            )
            logger.info("Primary mapper: LOMAP relaxed (max3d=5.0, element_change=True)")

        else:
            raise ValueError(
                f"Unknown atom_mapper: {atom_mapper}. "
                "Choose 'kartograf', 'lomap', or 'lomap_relaxed'."
            )

        # Setup FALLBACK mapper (opposite of primary for robustness)
        if atom_mapper == 'kartograf':
            # Fallback to LOMAP if Kartograf fails
            self.fallback_mapper = LomapAtomMapper(
                max3d=5.0,
                element_change=True,
                time=30,
            )
            logger.info("Fallback mapper: LOMAP relaxed (max3d=5.0, element_change=True)")
        else:
            # Fallback to Kartograf if LOMAP fails (if available)
            if KARTOGRAF_AVAILABLE:
                self.fallback_mapper = KartografAtomMapper(atom_map_hydrogens=True)
                logger.info("Fallback mapper: Kartograf")
            else:
                self.fallback_mapper = None
                logger.info("Fallback mapper: None (Kartograf not available)")

        # Default scorer for network quality
        self.scorer = lomap_scorers.default_lomap_score

        logger.info(
            f"Network planner initialized with {atom_mapper} mapper "
            f"(following OpenFE best practices)"
        )
    
    def create_network(
        self,
        ligands: List[openfe.SmallMoleculeComponent],
        topology: str = 'mst',
        central_ligand_name: Optional[str] = None
    ) -> Tuple[Any, LigandNetworkData]:
        """
        Create a ligand network using user-selected atom mapper.

        Following OpenFE best practices, the atom mapper creates the network AND
        handles alignment simultaneously. No pre-alignment step is needed.

        The primary mapper (user's choice) is attempted first. If it fails, a
        fallback mapper is tried for robustness.

        Args:
            ligands: List of OpenFE SmallMoleculeComponent objects with 3D coordinates
            topology: Network topology ('mst', 'radial', 'maximal')
            central_ligand_name: Name of central ligand for radial networks

        Returns:
            Tuple of (OpenFE LigandNetwork, LigandNetworkData for serialization)

        Raises:
            ValueError: If fewer than 2 ligands provided
            RuntimeError: If network creation fails with both primary and fallback mappers
        """
        if len(ligands) < 2:
            raise ValueError("At least 2 ligands are required for RBFE calculations")

        logger.info(
            f"Creating {topology} network with {len(ligands)} ligands "
            f"using {self.atom_mapper_type} mapper"
        )

        # Identify central ligand if needed (shared for all attempts)
        central_ligand = None
        if topology == 'radial':
            if central_ligand_name:
                for lig in ligands:
                    if lig.name == central_ligand_name:
                        central_ligand = lig
                        break

            if central_ligand is None:
                # Default to first ligand
                central_ligand = ligands[0]
                logger.warning(f"Central ligand not found, using {central_ligand.name}")

        def generate_with_mapper(mappers_list, mapper_name):
            """Helper to generate network with a specific mapper."""
            logger.info(f"Attempting network generation with {mapper_name}...")
            if topology == 'mst':
                return generate_minimal_spanning_network(
                    ligands=ligands,
                    mappers=mappers_list,
                    scorer=self.scorer
                )
            elif topology == 'radial':
                return generate_radial_network(
                    ligands=ligands,
                    central_ligand=central_ligand,
                    mappers=mappers_list,
                    scorer=self.scorer
                )
            elif topology == 'maximal':
                return generate_maximal_network(
                    ligands=ligands,
                    mappers=mappers_list,
                    scorer=self.scorer
                )
            else:
                raise ValueError(f"Unknown topology: {topology}")

        MIN_MAPPED_ATOMS = 3

        def validate_network_mappings(net):
            """Check that all edges have sufficient atom mappings."""
            for edge in net.edges:
                if hasattr(edge, 'componentA_to_componentB'):
                    if len(edge.componentA_to_componentB) < MIN_MAPPED_ATOMS:
                        return False
            return True

        try:
            # STEP 1: Try PRIMARY mapper (user's choice)
            network = generate_with_mapper([self.primary_mapper], self.atom_mapper_type)

            # Validate mappings are non-empty (Kartograf can return empty mappings
            # when ligands aren't spatially aligned without raising an exception)
            if not validate_network_mappings(network):
                mapped_counts = []
                for edge in network.edges:
                    if hasattr(edge, 'componentA_to_componentB'):
                        mapped_counts.append(len(edge.componentA_to_componentB))
                raise RuntimeError(
                    f"{self.atom_mapper_type} produced insufficient atom mappings "
                    f"(found {mapped_counts}, need >= {MIN_MAPPED_ATOMS} per edge). "
                    "Ligands may not be spatially aligned or too dissimilar."
                )

            logger.info(
                f"✓ Network created successfully with {self.atom_mapper_type} mapper "
                f"(user-selected)"
            )

        except RuntimeError as e:
            # Primary mapper failed - log and try fallback
            logger.warning(
                f"Primary mapper ({self.atom_mapper_type}) failed: {e}. "
                "This may indicate incompatible ligands or coordinate issues."
            )

            # STEP 2: Try FALLBACK mapper (if available)
            if self.fallback_mapper:
                fallback_name = "LOMAP" if self.atom_mapper_type == 'kartograf' else "Kartograf"
                logger.info(f"Attempting fallback with {fallback_name} mapper...")
                try:
                    network = generate_with_mapper([self.fallback_mapper], fallback_name)

                    if not validate_network_mappings(network):
                        mapped_counts = []
                        for edge in network.edges:
                            if hasattr(edge, 'componentA_to_componentB'):
                                mapped_counts.append(len(edge.componentA_to_componentB))
                        raise RuntimeError(
                            f"{fallback_name} also produced insufficient atom mappings "
                            f"(found {mapped_counts}, need >= {MIN_MAPPED_ATOMS} per edge)."
                        )

                    logger.info(
                        f"✓ Network created successfully with fallback {fallback_name} mapper"
                    )
                except RuntimeError as e2:
                    logger.error(
                        f"Both primary ({self.atom_mapper_type}) and fallback "
                        f"({fallback_name}) mappers failed."
                    )
                    raise RuntimeError(
                        f"Network creation failed with both mappers. "
                        f"Primary error: {e}. Fallback error: {e2}"
                    ) from e2
            else:
                logger.error(
                    f"Primary mapper ({self.atom_mapper_type}) failed and no fallback available."
                )
                raise RuntimeError(
                    f"Network creation failed: {e}. No fallback mapper available."
                ) from e

        # Convert to serializable data
        network_data = self._network_to_data(network, topology, central_ligand_name)

        # Log mapping details for diagnostics
        for edge in network.edges:
            if hasattr(edge, 'componentA_to_componentB'):
                n_mapped = len(edge.componentA_to_componentB)
                logger.info(
                    f"  Edge {edge.componentA.name} → {edge.componentB.name}: "
                    f"{n_mapped} mapped atoms"
                )

        logger.info(
            f"Network created: {len(network_data.nodes)} nodes, "
            f"{len(network_data.edges)} edges"
        )

        return network, network_data
    
    def _network_to_data(
        self,
        network: Any,
        topology: str,
        central_ligand: Optional[str]
    ) -> LigandNetworkData:
        """Convert OpenFE LigandNetwork to serializable data structure."""
        nodes = [node.name for node in network.nodes]
        
        edges = []
        for edge in network.edges:
            # Get mapping score if available
            score = 0.0
            mapping_info = {}
            
            if hasattr(edge, 'annotations'):
                score = edge.annotations.get('score', 0.0)
                mapping_info = edge.annotations
            
            edges.append(NetworkEdge(
                ligand_a=edge.componentA.name,
                ligand_b=edge.componentB.name,
                score=score,
                mapping_info=mapping_info
            ))
        
        return LigandNetworkData(
            nodes=nodes,
            edges=edges,
            topology=topology,
            central_ligand=central_ligand
        )
    
    def estimate_network_quality(
        self,
        network_data: LigandNetworkData
    ) -> Dict[str, Any]:
        """
        Estimate the quality of a network based on edge scores.
        
        Returns metrics about network connectivity and expected reliability.
        """
        if not network_data.edges:
            return {
                'num_nodes': len(network_data.nodes),
                'num_edges': 0,
                'avg_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'quality': 'poor'
            }
        
        scores = [e.score for e in network_data.edges]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        # Quality assessment based on Lomap scores
        # Higher scores are better (closer to 1.0)
        if avg_score >= 0.7:
            quality = 'excellent'
        elif avg_score >= 0.5:
            quality = 'good'
        elif avg_score >= 0.3:
            quality = 'moderate'
        else:
            quality = 'poor'
        
        return {
            'num_nodes': len(network_data.nodes),
            'num_edges': len(network_data.edges),
            'avg_score': avg_score,
            'min_score': min_score,
            'max_score': max_score,
            'quality': quality
        }
    
    def generate_edge_svg(self, edge) -> List[str]:
        """Generate [svg_a, svg_b] SVG strings for one atom mapping edge.

        Atoms in the shared mapping are highlighted green; atoms unique to each
        molecule are highlighted red.

        Returns:
            List of 2 SVG strings [svg_mol_a, svg_mol_b], or empty list on failure.
        """
        try:
            from rdkit.Chem.Draw import rdMolDraw2D
            from rdkit.Chem import Draw, AllChem
            import copy

            mol_a = edge.componentA.to_rdkit()
            mol_b = edge.componentB.to_rdkit()
            mapping: Dict[int, int] = edge.componentA_to_componentB if hasattr(edge, 'componentA_to_componentB') else {}

            mapped_a = list(mapping.keys())
            unique_a = [i for i in range(mol_a.GetNumAtoms()) if i not in mapping]
            mapped_b = list(mapping.values())
            unique_b = [i for i in range(mol_b.GetNumAtoms()) if i not in set(mapped_b)]

            GREEN = (0.2, 0.8, 0.2)
            RED = (0.8, 0.2, 0.2)

            svgs = []
            for mol, mapped, unique in [(mol_a, mapped_a, unique_a), (mol_b, mapped_b, unique_b)]:
                # Work on a copy so we don't mutate the original
                mol2d = copy.copy(mol)
                # Remove any existing conformers (3D) and generate fresh 2D coords
                mol2d.RemoveAllConformers()
                AllChem.Compute2DCoords(mol2d)

                # DrawMoleculeWithHighlights expects dict[int, list[color_tuple]]
                hl_atoms: Dict[int, List] = {a: [GREEN] for a in mapped}
                hl_atoms.update({a: [RED] for a in unique})

                drawer = rdMolDraw2D.MolDraw2DSVG(250, 250)
                Draw.PrepareMolForDrawing(mol2d)
                drawer.DrawMoleculeWithHighlights(mol2d, '', hl_atoms, {}, {}, {})
                drawer.FinishDrawing()
                svgs.append(drawer.GetDrawingText())

            return svgs
        except Exception as e:
            logger.warning(f"Failed to generate mapping SVG: {e}", exc_info=True)
            return []

    def compute_all_pairwise_mappings(self, ligands: List) -> List[Dict[str, Any]]:
        """Compute all pairwise atom mappings and return per-pair data with SVGs.

        Uses the primary mapper to generate a maximal network, then extracts
        per-edge mapping data and renders highlight SVGs.

        Args:
            ligands: List of OpenFE SmallMoleculeComponent objects (at least 2).

        Returns:
            List of dicts with keys:
                ligand_a, ligand_b, score, num_mapped,
                num_unique_a, num_unique_b, svgs
        """
        if len(ligands) < 2:
            raise ValueError("At least 2 ligands are required for pairwise mapping")

        logger.info(f"Computing all pairwise mappings for {len(ligands)} ligands using {self.atom_mapper_type}")

        network = generate_maximal_network(
            ligands=ligands,
            mappers=[self.primary_mapper],
            scorer=self.scorer,
        )

        pairs: List[Dict[str, Any]] = []
        for edge in network.edges:
            mapping: Dict[int, int] = {}
            if hasattr(edge, 'componentA_to_componentB'):
                mapping = edge.componentA_to_componentB

            score = 0.0
            if hasattr(edge, 'annotations'):
                score = edge.annotations.get('score', 0.0)

            mapped_a = list(mapping.keys())
            mapped_b = list(mapping.values())

            try:
                mol_a = edge.componentA.to_rdkit()
                mol_b = edge.componentB.to_rdkit()
                num_unique_a = mol_a.GetNumAtoms() - len(mapped_a)
                num_unique_b = mol_b.GetNumAtoms() - len(mapped_b)
            except Exception:
                num_unique_a = 0
                num_unique_b = 0

            svgs = self.generate_edge_svg(edge)

            pairs.append({
                'ligand_a': edge.componentA.name,
                'ligand_b': edge.componentB.name,
                'score': float(score),
                'num_mapped': len(mapped_a),
                'num_unique_a': num_unique_a,
                'num_unique_b': num_unique_b,
                'svgs': svgs,
            })

        logger.info(f"Computed {len(pairs)} pairwise mappings")
        return pairs

    def network_data_to_dict(self, network_data: LigandNetworkData) -> Dict[str, Any]:
        """Convert LigandNetworkData to JSON-serializable dict."""
        return {
            'nodes': network_data.nodes,
            'edges': [
                {
                    'ligand_a': e.ligand_a,
                    'ligand_b': e.ligand_b,
                    'score': e.score,
                    'mapping_info': e.mapping_info
                }
                for e in network_data.edges
            ],
            'topology': network_data.topology,
            'central_ligand': network_data.central_ligand
        }

