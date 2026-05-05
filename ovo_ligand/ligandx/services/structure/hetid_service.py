"""HET ID service for finding PDB structures by ligand chemical component ID."""
import httpx
import logging

logger = logging.getLogger(__name__)


class HETIDService:
    """Service for finding PDB structures by HET ID (chemical component ID)."""

    RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

    def get_best_structure_for_hetid(self, het_id: str) -> str:
        """Find the best PDB structure containing the given HET ID.

        Queries RCSB PDB for entries containing the chemical component,
        sorted by resolution (ascending) to return the highest-quality structure.

        Args:
            het_id: PDB chemical component ID (e.g. 'ATP', 'HEM', 'LIG')

        Returns:
            Lowercase PDB ID of the best structure containing the ligand.

        Raises:
            ValueError: If no structures are found for the given HET ID.
        """
        query = {
            "query": {
                "type": "terminal",
                "service": "text_chem",
                "parameters": {
                    "attribute": "rcsb_chem_comp_container_identifiers.comp_id",
                    "operator": "exact_match",
                    "value": het_id,
                },
            },
            "return_type": "entry",
            "request_options": {
                "sort": [
                    {
                        "sort_by": "rcsb_entry_info.resolution_combined",
                        "direction": "asc",
                    }
                ],
                "paginate": {"start": 0, "rows": 10},
            },
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(self.RCSB_SEARCH_URL, json=query)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to query RCSB PDB for HET ID '{het_id}': {e}")
            raise ValueError(f"Failed to search PDB database for ligand '{het_id}': {e}")

        results = data.get("result_set", [])
        if not results:
            raise ValueError(f"No PDB structures found containing ligand '{het_id}'")

        best_pdb_id = results[0]["identifier"].lower()
        logger.info(
            f"Found {len(results)} structures for HET ID '{het_id}', using: {best_pdb_id}"
        )
        return best_pdb_id
