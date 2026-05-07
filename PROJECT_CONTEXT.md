# PROJECT_CONTEXT

## 1) Project Overview
`ovo-ligand` is a ligand-centric computational chemistry app built as an OVO plugin package with its own Streamlit app UX and Docker-backed execution model. The project began as “OVO-inspired” but is intended to remain self-contained and focused on small-molecule workflows rather than OVO protein-design workflows.

Primary local repo path:
- `/home/user/programs/git-projects/ovo-ligand`

Runtime workspace path:
- `/home/user/programs/git-projects/ovo-ligand/.ovo-home/workdir/runs`

## 2) Product Goals
1. Provide a practical, wizard-like user experience for structure preparation, MD setup, MD production, and analysis.
2. Keep chemistry handling strict and reproducible (especially ligand refinement/use).
3. Decouple workflow stages into reusable assets:
   - Structure prep artifacts
   - MD system prep artifacts
   - Production runs from prepared systems
4. Allow job revisiting/inspection and downstream reuse (OpenFE/analysis).

## 3) Current Features
- **Structure Preparation task** (PDB-driven):
  - Download complex
  - Select chain/ligand
  - Produce refined artifacts (protein/complex and ligand files)
  - Show preparation reports and ligand preview panels
- **MD System Preparation task**:
  - Import prepared structure
  - Build and equilibrate system through NPT
  - Output NPT/system artifacts for downstream production
- **MD Production task**:
  - Select prepared MD system
  - Select restart mode:
    - Checkpoint continuation
    - NPT-final PDB coordinate restart
  - Run production in Docker
- **Jobs pages**:
  - Structure jobs
  - MD system prep jobs
  - MD production jobs (plus legacy/transition pages)
- **Visualization**:
  - Mol* viewer integration (vendored component)
  - py3Dmol-based views in results contexts
- **Result rendering**:
  - Run status
  - Output files
  - Thermodynamics/RMSD sections (varies by run type/stage)

Unknown/To be confirmed:
- Final completeness of OpenFE pages and exact downstream integration status.

## 4) User Flows
### A) Structure → MD system prep → MD production (intended primary flow)
1. User runs Structure Preparation from PDB.
2. User selects chains and ligand; runs preparation.
3. User opens MD System Preparation and imports prepared structure job.
4. System prep runs through NPT and writes artifacts.
5. User opens MD Production, selects MD system prep job.
6. User chooses restart mode and runs production.

### B) Production restart modes
1. **Checkpoint (exact continuation)**:
   - Requires NPT checkpoint (`*_npt_final.chk`) and system definition compatibility.
2. **NPT-final PDB (coordinate restart)**:
   - Uses final coordinates only; less exact and more fragile.

## 5) Tech Stack
- Python package (`pyproject.toml`, requires Python `>=3.11`)
- Streamlit pages
- Docker runtime orchestration from Python subprocess calls
- Ligand-X-derived backend modules vendored in-tree:
  - `ovo_ligand/ligandx/...`
- MD/chem dependencies include (project-level): `ovo`, `mdtraj`, `pandas`, `py3Dmol`
- Container builds via:
  - `docker-compose.yml`
  - `containers/*`

Reference/inspiration folders:
- `/home/user/programs/ovo/original/` (OVO source patterns)
- `/home/user/programs/ligand-x` (workflow and environment reference)
- `/home/user/programs/HiQBind/workflow/` (discussed as reference; parity scope To be confirmed)

## 6) Codebase Structure
Top-level:
- `/home/user/programs/git-projects/ovo-ligand/README.md`
- `/home/user/programs/git-projects/ovo-ligand/pyproject.toml`
- `/home/user/programs/git-projects/ovo-ligand/docker-compose.yml`
- `/home/user/programs/git-projects/ovo-ligand/containers/`
- `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/`

Key package paths:
- App shell/navigation:
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/run_app.py`
- UI pages:
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/app/pages/`
- Components:
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/app/components/molstar_viewer/`
- Workflow wrappers:
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/workflows/`
- Ligand-X vendored logic:
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/ligandx/services/md/`
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/ligandx/services/structure/`

## 7) Data Model
Primary run artifacts are file-based JSON + generated outputs.

Common run folder pattern:
- `.ovo-home/workdir/runs/<workflow>/<run_id>/`

Important files:
- `input.json` (run input contract)
- `result.json` (workflow output contract)
- `metadata.json` (UI/job metadata)
- additional workflow outputs in `md_outputs/` (for MD jobs)

Common metadata keys observed:
- `workflow`, `status`, `created_at`, `completed_at`
- `run_id`, `job_code`
- `pdb_id`, `ligand_key`, `ligand_label`
- `structure_run_id`, `md_system_prep_run_id`
- runtime info (`docker_image`, `use_gpu`, `host_run_dir`)

## 8) API / Backend Logic
No external web API contract is primary; orchestration is in-process Python + Docker execution.

Important backend classes/functions:
- `MDOptimizationConfig`:
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/ligandx/services/md/config.py`
- `MDOptimizationService.optimize` and preparation/system creation:
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/ligandx/services/md/service.py`
- Equilibration/production runner:
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/ligandx/services/md/workflow/equilibration_runner.py`
- Wrapper building input config and invoking service:
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/workflows/bound_ligand_md.py`

Recent key implementation direction:
- Resume/checkpoint fields propagated through config and wrapper.
- MD production page sets restart mode and passes resume paths.
- Service adds resume-mode branch to bypass fragile protein-cleaning path for checkpoint continuation.

To be confirmed:
- Full backward-compatibility behavior for all legacy input modes.

## 9) UI / Design System
- Streamlit app with multiple task/job pages.
- UX has been evolving from one large wizard page to separated task pages.
- Visual style and controls are custom Streamlit + vendored Mol* component.
- Some legacy/transition pages still exist and can confuse navigation.

Known UX principle from user requests:
- Keep user-facing flow explicit and modular.
- Avoid silently falling back to old routes/modes.
- Show exactly which files are used for each run stage.

## 10) State Management
- Streamlit session state is used extensively for page-level interactions.
- Job persistence is filesystem-based rather than DB-backed.
- Cross-page continuity typically uses run metadata and artifacts under `.ovo-home/workdir/runs`.

Risk area:
- Streamlit key collisions and session-state mutation order (seen in earlier trajectory controls work).

## 11) Environment Variables and Setup
Setup commands (known):
- `source /home/user/mambaforge/etc/profile.d/conda.sh`
- `conda activate ovo-ligand`
- `ovo-ligand init`
- `ovo-ligand app`

Build containers:
- `docker compose build`

Runtime/environment notes:
- App runtime root:
  - `OVO_HOME=/home/user/programs/git-projects/ovo-ligand/.ovo-home`
- Temp path:
  - `TMPDIR=/home/user/programs/git-projects/ovo-ligand/.tmp`

Container path constraints (important):
- In container, repo is mounted at `/ovo-ligand`
- Run dir mounted at `/output`
- Host absolute paths are not directly readable unless translated to mounted paths.

## 12) Known Bugs and Issues
1. Legacy/new routing overlap can still show old pages unexpectedly in some navigation contexts.
2. Restart-mode behavior has been fragile due to path mapping and system rebuild assumptions.
3. Coordinate restart can fail in protein prep/system recreation paths.
4. Inconsistent run/status presentation across job pages during migration.
5. Missing logs in some failed runs (only `result.json` available), reducing observability.
6. Some analysis/UI areas still carry transitional placeholders.

## 13) Decisions Already Made
1. Keep `ovo-ligand` self-contained and not coupled to OVO design pages.
2. Structure preparation, MD system prep, and MD production are distinct tasks.
3. Production restart mode selector is explicit in UI.
4. Checkpoint continuation is treated as preferred path.
5. Do not invent MM/GBSA implementation when absent in original Ligand-X core.
6. Preserve provenance through `input.json`, `result.json`, `metadata.json`.

## 14) Open Questions
1. Should coordinate restart remain available long-term or be hidden behind advanced mode?
2. Exact UI scope for non-MD tasks (Docking/OpenFE/ADMET/QC) in the modular task model.
3. Final contract of required artifacts between structure-prep and system-prep runs.
4. How much of legacy pages should remain accessible.
5. Whether to enforce strict no-fallback globally for all production-related workflows.

## 15) Immediate Next Steps
1. Validate checkpoint continuation across several fresh MD system prep runs.
2. Add automated checks/tests for:
   - resume path mapping
   - required artifact presence
   - deterministic checkpoint flow
3. Improve failure reporting by ensuring stderr/console logs are always captured to run folder.
4. Continue UI cleanup:
   - remove confusing duplicate sections
   - make structure source and run source explicit
5. Confirm and stabilize job table columns and links for all job types.

## 16) Instructions for Future LLMs
1. Do not guess chemistry/MD behavior; verify in code and run artifacts.
2. For any failure report, inspect the exact run folder first:
   - `/home/user/programs/git-projects/ovo-ligand/.ovo-home/workdir/runs/.../<run_id>/`
3. Preserve strict provenance and explicit mode behavior in UI and JSON payloads.
4. Avoid hidden fallback paths for production continuation.
5. Treat checkpoint continuation as default recommendation unless user explicitly requests otherwise.
6. Keep edits scoped; this codebase has many in-flight refactors.
7. If something is unknown, mark `Unknown` or `To be confirmed`.
8. When editing path handling for Docker runs, validate host→container path visibility.

---

## Useful Commands
- App:
  - `ovo-ligand init`
  - `ovo-ligand app`
- Build:
  - `docker compose build`
- Inspect latest runs:
  - `find /home/user/programs/git-projects/ovo-ligand/.ovo-home/workdir/runs -maxdepth 4 -type f | sort`

Last Updated: 2026-05-07
