from __future__ import annotations

import json
import csv
import io
import re
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import (
    JOB_SCHEMA_VERSION,
    JobRecord,
    display_job_code,
    short_job_code,
)
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.compound_preparation import (
    compound_component_records,
    normalize_compound_smiles,
)
from rdkit import rdBase


COMPOUND_REVIEW_TASK_GROUP = "compound-review"
PUBCHEM_PROPERTIES = (
    "ConnectivitySMILES,SMILES,MolecularFormula,InChIKey"
)
PUBCHEM_MAX_NAME_CANDIDATES = 50
FORMULA_ELEMENT_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _formula_composition(formula: str) -> dict[str, Fraction]:
    composition: dict[str, Fraction] = {}
    for raw_part in re.split(r"[.·]", str(formula or "").replace(" ", "")):
        part = raw_part.strip()
        if not part:
            continue
        coefficient = Fraction(1)
        coefficient_match = re.match(
            r"^((?:\d+/\d+)|(?:\d+(?:\.\d+)?))(?=[A-Z])",
            part,
        )
        if coefficient_match:
            coefficient = Fraction(coefficient_match.group(1))
            part = part[coefficient_match.end() :]
        matches = list(FORMULA_ELEMENT_PATTERN.finditer(part))
        if not matches or "".join(match.group(0) for match in matches) != part:
            raise ValueError(f"Unsupported molecular formula: {formula}")
        for match in matches:
            count = Fraction(int(match.group(2) or "1"))
            element = match.group(1)
            composition[element] = (
                composition.get(element, Fraction(0)) + coefficient * count
            )
    if not composition:
        raise ValueError("Molecular formula cannot be empty")
    return composition


def compare_molecular_formulas(
    vendor_formula: str,
    candidate_formula: str,
) -> dict[str, Any]:
    """Compare formulae while allowing an integer-scaled formulation unit."""
    try:
        vendor = _formula_composition(vendor_formula)
        candidate = _formula_composition(candidate_formula)
    except ValueError as exc:
        return {
            "compatible": False,
            "status": "not comparable",
            "message": str(exc),
            "scale": "",
        }
    if set(vendor) != set(candidate):
        return {
            "compatible": False,
            "status": "different elemental composition",
            "message": "Vendor and PubChem formulas contain different elements.",
            "scale": "",
        }
    scales = {
        candidate[element] / vendor[element]
        for element in vendor
        if vendor[element] != 0
    }
    if len(scales) != 1:
        return {
            "compatible": False,
            "status": "stoichiometry differs",
            "message": "Element ratios do not describe the same formula unit.",
            "scale": "",
        }
    scale = next(iter(scales))
    if scale <= 0:
        return {
            "compatible": False,
            "status": "stoichiometry differs",
            "message": "Formula scale is not positive.",
            "scale": "",
        }
    scale_label = (
        str(scale.numerator)
        if scale.denominator == 1
        else f"{scale.numerator}/{scale.denominator}"
    )
    status = "exact formula" if scale == 1 else "scaled formula unit"
    message = (
        "PubChem and vendor formulas are identical."
        if scale == 1
        else (
            f"PubChem equals {scale_label} × the vendor formula unit; "
            "this is stoichiometrically compatible."
        )
    )
    return {
        "compatible": True,
        "status": status,
        "message": message,
        "scale": scale_label,
    }


def pubchem_component_options(smiles: str) -> list[dict[str, Any]]:
    """Group identical PubChem fragments into selectable single components."""
    records = compound_component_records(
        smiles,
        compound_id="pubchem-candidate",
    )
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        component_smiles = str(record.get("smiles") or "")
        if component_smiles not in grouped:
            grouped[component_smiles] = {
                "smiles": component_smiles,
                "formula": str(record.get("formula") or ""),
                "molecular_weight": float(
                    record.get("molecular_weight") or 0
                ),
                "heavy_atoms": int(record.get("heavy_atoms") or 0),
                "formal_charge": int(record.get("formal_charge") or 0),
                "occurrences": 0,
            }
        grouped[component_smiles]["occurrences"] += 1
    options = sorted(
        grouped.values(),
        key=lambda row: (
            -int(row["heavy_atoms"]),
            -float(row["molecular_weight"]),
            str(row["smiles"]),
        ),
    )
    if not options:
        raise ValueError("PubChem candidate contains no components")
    largest_heavy_atoms = int(options[0]["heavy_atoms"])
    largest_count = sum(
        int(option["heavy_atoms"]) == largest_heavy_atoms
        for option in options
    )
    for index, option in enumerate(options):
        option["automatic_parent_candidate"] = bool(
            index == 0 and largest_count == 1
        )
    return options


def selected_parent_candidate(
    candidate: dict[str, Any],
    component: dict[str, Any],
) -> dict[str, Any]:
    """Create a single-component candidate while retaining the full record."""
    full_smiles = str(candidate.get("smiles") or "")
    full_formula = str(candidate.get("molecular_formula") or "")
    return {
        **candidate,
        "pubchem_record_smiles": full_smiles,
        "pubchem_record_formula": full_formula,
        "pubchem_record_fragment_count": len(
            compound_component_records(
                full_smiles,
                compound_id="pubchem-candidate",
            )
        ),
        "smiles": str(component.get("smiles") or ""),
        "molecular_formula": str(component.get("formula") or ""),
        "selected_component_smiles": str(component.get("smiles") or ""),
        "selected_component_formula": str(component.get("formula") or ""),
        "selected_component_occurrences": int(
            component.get("occurrences") or 1
        ),
        "component_selection": (
            "automatic unique-largest component"
            if component.get("automatic_parent_candidate")
            else "user-selected component"
        ),
    }


def search_pubchem_candidates(
    query: str,
    *,
    query_type: str,
    timeout_seconds: float = 20.0,
) -> list[dict[str, Any]]:
    """Look up PubChem candidates by CAS or a multi-result name search."""
    value = str(query or "").strip()
    if not value:
        raise ValueError("PubChem search query cannot be empty")
    if query_type not in {"cas", "name"}:
        raise ValueError("PubChem query type must be 'cas' or 'name'")
    search_terms = [value]
    if query_type == "name":
        base_name = re.sub(r"\s*\([^()]+\)\s*$", "", value).strip()
        if base_name and base_name.casefold() != value.casefold():
            search_terms.append(base_name)

    cid_order: list[int] = []
    matched_terms: dict[int, list[str]] = {}
    for term in search_terms:
        encoded = quote(term, safe="")
        identifier_url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{encoded}/cids/JSON"
            + ("?name_type=word" if query_type == "name" else "")
        )
        request = Request(
            identifier_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "mn-ligand/0.0.1 compound-review",
            },
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                identifier_payload = json.loads(
                    response.read().decode("utf-8")
                )
        except HTTPError as exc:
            if exc.code == 404:
                continue
            raise ValueError(f"PubChem returned HTTP {exc.code}") from exc
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise ValueError(f"PubChem lookup failed: {exc}") from exc
        identifiers = (
            identifier_payload.get("IdentifierList", {}).get("CID", [])
            if isinstance(identifier_payload, dict)
            else []
        )
        for raw_cid in identifiers:
            cid = int(raw_cid or 0)
            if not cid:
                continue
            matched_terms.setdefault(cid, []).append(term)
            if cid not in cid_order:
                cid_order.append(cid)
            if len(cid_order) >= PUBCHEM_MAX_NAME_CANDIDATES:
                break
    if not cid_order:
        return []

    cid_text = ",".join(str(cid) for cid in cid_order)
    property_url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{cid_text}/property/{PUBCHEM_PROPERTIES}/JSON"
    )
    request = Request(
        property_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "mn-ligand/0.0.1 compound-review",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            property_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ValueError(f"PubChem returned HTTP {exc.code}") from exc
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise ValueError(f"PubChem lookup failed: {exc}") from exc

    properties = (
        property_payload.get("PropertyTable", {}).get("Properties", [])
        if isinstance(property_payload, dict)
        else []
    )
    properties_by_cid = {
        int(item.get("CID") or 0): item
        for item in properties
        if isinstance(item, dict)
    }
    candidates: list[dict[str, Any]] = []
    for rank, cid in enumerate(cid_order, start=1):
        item = properties_by_cid.get(cid)
        if not item:
            continue
        smiles = str(item.get("SMILES") or "").strip()
        if not smiles:
            continue
        terms = list(dict.fromkeys(matched_terms.get(cid) or []))
        candidates.append(
            {
                "cid": cid,
                "smiles": smiles,
                "connectivity_smiles": str(
                    item.get("ConnectivitySMILES") or ""
                ),
                "molecular_formula": str(item.get("MolecularFormula") or ""),
                "inchi_key": str(item.get("InChIKey") or ""),
                "query": value,
                "query_type": query_type,
                "matched_queries": terms,
                "match_scope": (
                    "full vendor name"
                    if value in terms
                    else "base name without formulation qualifier"
                ),
                "candidate_rank": rank,
                "pubchem_url": (
                    f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
                    if cid
                    else ""
                ),
                "retrieved_at": _utc_now_iso(),
            }
        )
    return candidates


def create_compound_review_job(
    source_job: JobRecord,
    *,
    source_row: int,
    compound_id: str,
    decision: str,
    query: str = "",
    query_type: str = "",
    candidate: dict[str, Any] | None = None,
) -> JobRecord:
    if source_job.task_group != "compound-import":
        raise ValueError("Compound reviews require a compound-import source job")
    if decision not in {"accepted", "rejected"}:
        raise ValueError("Compound review decision must be accepted or rejected")
    if decision == "accepted" and not candidate:
        raise ValueError("Accepted compound reviews require a PubChem candidate")
    normalized_candidate: dict[str, Any] = {}
    if decision == "accepted":
        normalized_candidate = normalize_compound_smiles(
            str((candidate or {}).get("smiles") or "")
        )
        if int(normalized_candidate.get("fragment_count") or 0) != 1:
            raise ValueError(
                "Confirmed PubChem review must contain exactly one selected "
                "parent component"
            )

    run_id = str(uuid4())
    run_dir = runs_root() / COMPOUND_REVIEW_TASK_GROUP / run_id
    report_dir = run_dir / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=False)
    now = _utc_now_iso()
    review = {
        "schema_version": 1,
        "source_task_group": source_job.task_group,
        "source_run_id": source_job.run_id,
        "source_row": int(source_row),
        "compound_id": str(compound_id),
        "decision": decision,
        "query": str(query),
        "query_type": str(query_type),
        "candidate": {
            **dict(candidate or {}),
            **normalized_candidate,
            "structure_origin": "PubChem confirmed import",
            "validation_engine": "RDKit",
            "rdkit_version": rdBase.rdkitVersion,
        }
        if candidate
        else {},
        "reviewed_at": now,
    }
    review_path = report_dir / "pubchem_review.json"
    review_path.write_text(json.dumps(review, indent=2) + "\n")
    reviewed_compound_path: Path | None = None
    if decision == "accepted":
        reviewed_compound_path = report_dir / "reviewed_compound.csv"
        reviewed_row = {
            "compound_id": str(compound_id),
            **review["candidate"],
            "pubchem_cid": int((candidate or {}).get("cid") or 0),
            "structure_origin": "PubChem confirmed import",
            "source_compound_run_id": source_job.run_id,
            "source_row": int(source_row),
        }
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=list(reviewed_row),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerow(reviewed_row)
        reviewed_compound_path.write_text(buffer.getvalue())
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "compound_pubchem_review",
        "status": "completed",
        "source": "PubChem PUG REST",
        "parent_run_id": source_job.run_id,
        "source_compound_run_id": source_job.run_id,
        "compound_id": str(compound_id),
        "source_row": int(source_row),
        "decision": decision,
        "structure_origin": (
            "PubChem confirmed import" if decision == "accepted" else ""
        ),
        "pubchem_cid": int((candidate or {}).get("cid") or 0),
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (run_dir / "input.json").write_text(
        json.dumps(
            {
                "source_task_group": source_job.task_group,
                "source_run_id": source_job.run_id,
                "source_row": int(source_row),
                "compound_id": str(compound_id),
                "query": str(query),
                "query_type": str(query_type),
            },
            indent=2,
        )
        + "\n"
    )
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "success": True,
                "decision": decision,
                "compound_id": str(compound_id),
                "pubchem_cid": int((candidate or {}).get("cid") or 0),
            },
            indent=2,
        )
        + "\n"
    )
    artifacts = [
            ArtifactRef.from_path(
                run_dir,
                review_path,
                "compound_review",
                role=f"pubchem_{decision}",
                metadata={
                    "compound_id": str(compound_id),
                    "source_run_id": source_job.run_id,
                    "decision": decision,
                },
            )
        ]
    if reviewed_compound_path is not None:
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                reviewed_compound_path,
                "reviewed_compound",
                role="pubchem_confirmed_import",
                metadata={
                    "compound_id": str(compound_id),
                    "pubchem_cid": int((candidate or {}).get("cid") or 0),
                    "structure_origin": "PubChem confirmed import",
                },
            )
        )
    write_artifact_manifest(run_dir, artifacts)
    return JobRecord.load(run_dir, task_group=COMPOUND_REVIEW_TASK_GROUP)


def list_compound_reviews(source_run_id: str) -> list[JobRecord]:
    root = runs_root() / COMPOUND_REVIEW_TASK_GROUP
    if not root.is_dir():
        return []
    reviews = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        job = JobRecord.load(path, task_group=COMPOUND_REVIEW_TASK_GROUP)
        if str(job.metadata.get("source_compound_run_id") or "") == source_run_id:
            reviews.append(job)
    return sorted(
        reviews,
        key=lambda job: (job.created_at, job.run_dir.stat().st_mtime),
        reverse=True,
    )


def latest_compound_review_map(source_run_id: str) -> dict[str, JobRecord]:
    latest: dict[str, JobRecord] = {}
    for job in list_compound_reviews(source_run_id):
        compound_id = str(job.metadata.get("compound_id") or "")
        if compound_id and compound_id not in latest:
            latest[compound_id] = job
    return latest


def accepted_reviewed_compound_rows(
    source_run_id: str,
) -> list[dict[str, Any]]:
    """Load the effective accepted additions from the latest review decisions."""
    rows: list[dict[str, Any]] = []
    for compound_id, job in latest_compound_review_map(source_run_id).items():
        if str(job.metadata.get("decision") or "") != "accepted":
            continue
        if job.artifact_manifest is None:
            continue
        refs = job.artifact_manifest.by_type("reviewed_compound")
        if not refs:
            continue
        path = refs[0].resolve(job.run_dir, must_exist=True)
        with path.open(newline="") as handle:
            reviewed_rows = list(csv.DictReader(handle))
        for row in reviewed_rows:
            rows.append(
                {
                    **row,
                    "compound_id": str(
                        row.get("compound_id") or compound_id
                    ),
                    "review_decision": "accepted",
                    "review_job": display_job_code(
                        job.metadata.get("job_code"), job.run_id
                    ),
                    "review_run_id": job.run_id,
                    "structure_origin": "PubChem confirmed import",
                }
            )
    return rows
