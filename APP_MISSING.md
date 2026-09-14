# mn-ligand Missing Work and Implementation Roadmap

For current development commands, testing policy, code landmarks, and the
handoff reading order, start with `APP_DEVELOPMENT.md`.

Status: code audit updated on 2026-08-03. This document distinguishes implemented,
partial, unvalidated, and proposed behavior.

## Current checkpoint (2026-08-03)

MD worker concurrency update (2026-09-11): per-workflow advancement and
per-child MD finalization are inter-process locked, successful finalized MD
children are idempotent, and shared JSON writers use unique temporary files.
Superseded queued workflow children are cancelled rather than left runnable.
This closes the observed race where concurrent workers both finalized the same
100 ns trajectory, one writer lost the shared `.artifacts.json.tmp` rename, and
the worker overwrote a successful native result with a false failure. The same
orchestration race had created redundant queued endpoint jobs. The affected
Eprotirome trajectories were retained and are repaired through a guarded,
audited analytics reconstruction rather than rerunning MD. The local CPU pool
is now 20 slots while AmberTools MM/PBSA retains a 16-thread request so it can
coexist with a four-thread GPU simulation. Queue discovery now scans only the
canonical run-record depth, caches unchanged metadata, and coordinates pending
workflow scans across workers; idle GPU workers no longer traverse nested MD
output and backup trees on every poll while endpoint analysis is CPU-bound.

The campaign-results and aggregate-comparison redesign is implemented. Physical
campaigns remain immutable execution records; compatible campaigns can be saved
as first-class Analysis Sets with an explicit pre-save membership table.
Comparison scope is centralized, target sorting no longer defines membership,
and prepared-target selection uses stable row identities. The adaptive views,
target × compound overview, input-recovery matrices, per-target and pooled
pose-agreement matrices, GNINA dual ranking, linked target 3D panels, local 3D
engine filtering, pose-validation and interaction integration, normalized
repetition-column export, and selected-plot PNG ZIP export are present.

The principal remaining comparison work is scientific validation rather than
basic UI construction:

- verify symmetry-aware T3 atom mapping against independently calculated
  controls, especially equivalent iodine assignments;
- validate fixed RMSD/centroid/shape-overlap color thresholds on chemically
  diverse ligands and document the accepted ranges in the UI;
- spot-check that every linked 3D panel and matrix cell resolves the identical
  immutable prediction selection after sorting/filtering;
- profile and reduce Streamlit rerun latency for large multi-target 3D scopes;
- validate Analysis Set compatibility and target naming on non-PDB and
  multi-chain sources.

Shared Scrub modeling-state handoff is implemented for Docking/Cofolding.
Canonical parent identity and engine modeling SMILES are stored separately;
RosettaLigand can preserve the prepared protonation/formal-charge state while
still performing its native conformer and partial-charge preparation. Native
scientific validation is in progress through campaign
`0c3f2e1a-91ea-4e9a-8501-1a8778254ef7`. Remaining work is to compare the
actual charged-state and pose artifacts across all seven engines and expose a
compact per-engine modeling-state audit on results pages.

The MD settings-template API is implemented and regression-tested. It creates
fresh system preparation and independent replicas for a new target, excludes
all source/checkpoint/trajectory state from the template, and converts an old
cumulative target duration into one full new production segment. Workflow
`9d240e37-b83e-4ab4-b352-6b0ce220d677` is queued as the first native validation:
prepared target `7C4A9`, OpenMM, three independent 100 ns replicas, automatic
AmberTools MM/PBSA per replica, and aggregate analysis. Both GPU workers were
occupied at creation time, so native completion and scientific review remain
pending; the durable workers will dispatch it when a GPU becomes available.

## Current checkpoint (2026-07-28)

Later checkpoint notes and the current code/tests supersede the historical
inventory below. Structure Import, Sequence Modification, Compound Datasets,
Pocket Detection, Docking / Cofolding, Redocking Benchmark, typed AF3/Boltz-2/
Nesso campaigns, and the durable two-GPU worker are implemented. The current
full local pytest baseline is **340 passed**.

Benchmark Dataset import and the unified Benchmark Redocking / Refolding page is
implemented with `Dataset → Engines → Run → Results`. The importer has one generic
manifest/archive contract plus PoseBench-aware profiles for Astex Diverse,
PoseBusters Benchmark, DockGen, and CASP15. Installed structural/scoring
adapters are exposed according to their real output capability; Nesso is not
assigned a ligand RMSD because it has no pose. Focused unit/AppTest and related
workflow regression coverage passes. The requested native PoseBusters smoke is
complete on immutable dataset `B9C86`, case `8AEU_M0L`: Vina, GNINA, Uni-Dock
Pro, RosettaLigand, Boltz-2, AlphaFold 3, Nesso-1, GNINA score-only, and
Boltzina all completed. Aggregate results contain canonical fixed-frame or
protein-aligned ligand RMSD as applicable, explicit campaign coverage, and six
exact score-to-source-pose joins. The remaining scientific work is a
production-setting multi-case campaign and interpretation of recovery and
ranking statistics; the one-case smoke is not a reproduced PoseBench result.
All four official PoseBench v1.1.0 collections are also registered locally:
Astex Diverse (85 cases), PoseBusters Benchmark (428), DockGen (260), and
CASP15 (15 ligand-evaluable cases plus one explicitly rejected ion-only
reference). These registrations have not yet been run as full native campaigns;
the remaining scientific work is still bounded multi-case and then
production-scale recovery validation.

The demanding HY-L126 Excel import is retained as compound-import run
`fc0deeb8-0b96-49e0-b95c-162598c5dd4e` (job `16259`, worksheet
`Compound Information`). Its effective result contains 766 usable records,
724 unique stereochemistry-aware docking parents, 42 redundant source records,
60 multi-component sources, and no unresolved parent ambiguity. The result page
shows both full duplicate membership and a one-row-per-parent
`Docking-ready parents` table without modifying the source workbook.

Docking / Cofolding now lists only completed compound imports, defaults to all
unique parents, supports manual multi-row selection, and records exact
membership in an immutable typed `compound-selection` job shared by all checked
engines. Automated unit and Streamlit AppTest coverage passes. A native docking
campaign using a manually selected subset remains the next scientific
validation gate.

For generative design, minimized 4LNW target job `6CFD2` now has exact PLIP
analysis `3DD68`, bound-T3 pocket `1585D`, and final immutable
SER277-side-chain-directed pharmacophore `F73C5`. Its active feature set
replaces the reference backbone hydrogen-bond point; the original observation
remains in immutable PLIP provenance. The next missing gate is a small native
generation campaign and downstream pose validation that requires an author
`A:SER277` side-chain contact.

## 1. Current implementation inventory

### Implemented in the ligand app

- Streamlit navigation for structure, MD system preparation, MD production,
  OpenFE, ADMET, QC, jobs, and result pages.
- PDB import, chain/ligand selection, protein cleanup, ligand extraction, reference
  SMILES lookup, bond-order correction, and standardized downstream filenames.
- Docking from a prepared structure through an external
  `avgu-docking-suite-cuda:latest` image with `udp`, Vina, and GNINA choices.
- Registration of successful docking poses as new structure jobs.
- Boltz-2 protein-ligand structure prediction, optional affinity prediction, and
  normalization into a structure job.
- OpenMM/OpenFF molecular-system preparation, equilibration, checkpoint restart,
  production MD, trajectory viewing, and result collection.
- OpenMM MM/GBSA plus an AmberTools MMPBSA.py path in the MD workflow.
- OpenFE ABFE/RBFE services, UI configuration, network planning, and result pages.
- ADMET-AI execution and normalized tabular output.
- File-backed run folders and a single-GPU queue/lock.
- Mol* and py3Dmol-based structure inspection.

### Present but incomplete or misleading

- Generic `docking.py` and `batch_docking.py` call the generic workflow page,
  whose fallback is a smoke wrapper rather than the integrated docking path used
  inside Structure Import.
- The custom protein/ligand upload tab records paths in metadata but does not
  create `*_protein_refined.pdb`, `*_ligand_refined.sdf`, and
  `*_complex_refined.pdb`; downstream selectors therefore ignore these jobs.
- QC writes mock values in `common.py`; ORCA is not bundled and no real QC command
  is executed by that path.
- ABFE/RBFE contain real OpenFE execution code, but production correctness,
  convergence reporting, retry/resume behavior, and current OpenFE-version
  compatibility are not established by automated tests in this repository.
- The GPU queue is implemented in Streamlit process code and executes queued jobs
  synchronously when a page dispatches them. It is not a durable worker service.
- Job contracts vary by page and task group; several collectors infer artifacts
  from filename globs.
- Some metadata stores host absolute paths. Legacy path remapping only recognizes
  selected old layouts.
- Boltz model/cache defaults are hard-coded to
  `/mnt/db/reference_files/boltz_models` in the page layer.
- `MN_LIGAND_APP_HOME` and `MN_LIGAND_RUN_DIR` exist, but there is no central
  runtime module, editable persisted Settings page, or reference/library path
  contract.
- Docker commands sometimes mount the repository source, coupling execution to a
  checkout rather than an installed package/image contract.
- Current docs still contain historical absolute paths from a different checkout.

### Source checkouts are not integrations

The untracked `tools_to_implement/` directory contains useful upstream source,
but none of these tools should be listed as an app feature until it has a pinned
image, manifest, adapter, normalized result parser, and tested UI/workflow route.

The local set includes HiQBind, OpenVS, OMTRA, Uni-Dock Pro, GNINA, fpocket,
PeSTo, Boltz, OpenMM, GROMACS, OpenFE, gmx_qk, PocketXMol, PocketFlow, GenMol,
DrugRPG, PFM, PGMG, and FlowR-related code.

## 2. Structure prediction tools already available

### In `mn-ligand`

**Boltz-2** is integrated as a user-facing protein-ligand complex prediction
workflow and as a typed batch adapter on Docking / Refolding. The typed path
consumes prepared target and compound-set artifacts, generates one native YAML
per compound, queues the CUDA image, and publishes predicted complexes,
confidence, affinity, and metrics artifacts through the worker.

**AlphaFold 3 / AlphaFast** now has a ligand-owned typed adapter on Docking /
Refolding. It consumes prepared target and compound-set artifacts, extracts
receptor chains, generates one native AF3 JSON input per compound, runs the
installed CUDA 12.8 AlphaFast image, and records predicted complexes and
confidence metrics. Reference resources use the shared layout beneath
`MN_LIGAND_REFERENCE_DIR`. Sequence-hashed cached A3Ms are reused; missing
chains are generated with AlphaFast MMseqs-GPU and written back to the shared
repository. Ordered data-pipeline and inference commands retain one worker claim
and GPU lease. Target-template injection, resume, large-campaign chunking, and
explicit in-app model-terms acknowledgement remain missing.

**Nesso-1** is integrated as an affinity-only cofolding engine on Docking /
Refolding. Its CUDA 12.8 Docker image is built from the pinned local source,
reuses the existing ESM-2 650M cache read-only, and publishes native and
normalized affinity artifacts without fabricating a predicted structure. Native
4WBK/STE inference, worker retry, failure handling, result display, and typed
affinity handoff are validated. Larger campaign throughput and calibration
against assay-specific endpoints remain future scientific validation work. The
full application suite passes at **162 tests** after this integration.

### Separately present in `mn-protein-design`

The separate protein-design application contains its own integrations for:

- Boltz-2
- AlphaFold 2/ColabFold and AF2 initial-guess variants
- AlphaFold 3/AlphaFast
- Protenix v0.5 and newer Protenix CLI variants
- OpenFold-3
- RF3/Foundry
- ESMFold/ESMFold2

These are not `mn-ligand` features and do not share its scientific workflow or
artifact model. Generic container, job-runner, settings, and Blackwell lessons
may be reviewed or copied into ligand-owned modules. Any predictor must otherwise
receive an independent ligand-app integration, license review, result parser,
and protein-small-molecule validation.

### Recommended prediction order

1. Harden the typed Boltz-2 batch mode with chunking, resume, and a worker-owned
   local-MSA generation path for sequences not yet present in the shared cache.
2. Port Protenix as the second openly deployable complex-prediction engine after
   confirming its selected version and model license.
3. Harden the AlphaFold 3 adapter with target-template injection, resumable
   chunks, and explicit in-app model-terms acknowledgement.
4. Independently integrate a protein-only predictor only when apo receptor
   ensemble generation is needed; do not import protein-design workflows or
   treat protein-only predictions as ligand poses.

## 3. Foundational changes required before adding more tools

### 3.1 Central runtime configuration

Create a ligand-owned `mn_ligand/runtime.py`. The neighboring runtime is only an
example of a configuration pattern:

- `app_home()`
- `runs_root()`
- `reference_root()`
- `library_root()`
- `tmp_root()`
- `config_path()`
- `ensure_runtime_home()`

Update the CLI with `--reference-dir`, `--library-dir`, and `--config`. Add a
Settings page that persists values under app home and shows effective precedence.
Replace direct `Path(__file__).../mn-ligand-workdir` calculations throughout.

Acceptance criteria:

- install the wheel in a fresh environment outside the source tree;
- point app home, runs, references, and libraries at four arbitrary directories;
- launch and complete a CPU smoke job and GPU smoke job;
- move the complete app home and reopen historical results without editing JSON.

### 3.2 One job store and typed artifacts

Create ligand-owned equivalents of jobs, artifacts, Docker execution, and tool
manifests. Generic infrastructure code may be copied and adapted when useful,
but do not import the neighboring package at runtime or modify that repository.
Protein-design entities, candidate stages, campaigns, and adapters must not be
copied into the ligand application.

Define each scientific operation as an independently launchable job. In
particular, protein import and protein cleaning/repair must be separate jobs.
Cleaning publishes a reusable `prepared_target` artifact that pocket prediction,
docking, virtual screening, receptor refolding, generative design, complex
preparation, and MD setup can all consume without repeating cleanup.

Required graph behavior:

- immutable artifact references using producer run ID and relative path;
- explicit parent workflow plus upstream/downstream job links;
- fan-out from one artifact into multiple independent jobs;
- fan-in when a campaign aggregates receptors, pockets, poses, or energy edges;
- clone/rerun from the same inputs with changed parameters;
- downstream compatibility determined by artifact type and schema version;
- deletion planning that detects consumers and prevents broken references;
- no hidden scientific work inside a page or result renderer.

Add:

```text
mn_ligand/core/runtime.py
mn_ligand/core/jobs.py
mn_ligand/core/artifacts.py
mn_ligand/core/docker_runner.py
mn_ligand/core/manifests.py
mn_ligand/core/workflows.py
mn_ligand/manifests/*.yaml
```

Migrate collectors from filename globs to `artifacts.json`. Keep compatibility
adapters that index existing run folders without rewriting them.

### 3.3 Durable local worker and resource scheduler

Replace page-triggered execution with a background worker. The UI should create a
queued job and return immediately. The worker should own subprocesses, logs,
heartbeats, cancellation, retries, and resource locks.

Resource requests should include GPU ID, minimum VRAM, CPU threads, RAM, scratch
space, and exclusivity. Multi-GPU support should use one lock per GPU rather than
one global file. Parent workflows should continue independent CPU stages while a
GPU stage waits.

Implemented status (2026-08-04): durable CPU and per-GPU workers now share an
atomic CPU-slot pool capped by the runtime `cpu_process_limit` (16 by default).
Every admitted job reserves at least one slot and normally reserves its declared
`cpu_threads`; GPU jobs simultaneously hold their CPU allocation and an exclusive
device lease. Exact per-slot files support heartbeat, release, rollback of partial
allocation, and stale dead-worker recovery. Oversized requests fail permanently,
while temporary pool exhaustion leaves jobs queued with an admission reason.
Settings reports leased CPU slots versus capacity. Activation of scheduler-code
updates requires a worker-service restart and should be deferred while scientific
jobs are active.

### 3.4 Manifest-driven Docker adapters

Each tool manifest should pin source commit, image digest, CUDA requirement,
license, model paths, input/output types, and a health-check command. The generic
runner should add consistent labels and mounts and write `command.json` before
launch.

Use separate images by dependency family. A single image containing every
chemistry tool will be difficult to reproduce and will make license boundaries
unclear.

### 3.5 UI organization refactor

Use the generic navigation pattern from the neighboring app without sharing its
protein-design pages or domain model:

- replace six task-specific Jobs sidebar pages with one default Jobs page;
- place Settings next to Jobs;
- group ligand-owned task pages under Prepare, Discover, and Simulate & Score;
- add asset pages for targets, pockets, compound sets, poses/complexes,
  trajectories, and energy results;
- use hidden job/workflow/asset detail routes rather than one sidebar result page
  per tool;
- give task pages a shared input-selection, configuration, preview, and queue
  layout;
- give result pages shared Overview, Artifacts, Metrics, Viewer, Lineage, and Logs
  tabs;
- expose downstream actions from typed artifacts instead of hard-coded page
  links;
- show workflow parent/child progress on both Jobs and workflow result pages.

Keep table state, filters, and selected rows stable when navigating to a result
and back. Large compound or pose sets require paginated/streamed summaries rather
than loading whole libraries into Streamlit dataframes.

## 4. Scientific work still missing

### 4.1 Structure import and repair

Priority: highest.

- Make custom upload produce the same normalized artifacts as PDB and Boltz jobs.
- Preserve mmCIF as the authoritative source and produce PDB only where a tool
  requires it.
- Port HiQBind protein and ligand repair into `ovolig-hiqbind` with a single-entry
  adapter instead of its dataset-only command shape.
- Expose biological assembly, alternate locations, missing atoms/residues,
  mutations, protonation, histidine states, disulfides, covalent ligands, metals,
  cofactors, and retained waters.
- Emit a machine-readable repair/validation report and atom/residue maps.
- Add tests for modified residues, multi-chain complexes, metal sites, covalent
  ligands, and ligands whose PDB coordinates lack reliable bond order.

Current implementation: PDB, mmCIF, manual-structure, and promoted AF3/Boltz
complex imports share a strict Gemmi/Ligand-X/PDBFixer/OpenMM/OpenFF/RDKit
repair path in the installation-managed MD image. It preserves mmCIF sequence
records, optionally builds a biological assembly, maps supported modified
residues including CAS/CAF, applies HiQBind's default terminal and ten-residue
gap safeguards, rebuilds missing atoms/residues, adds hydrogens, optionally
retains non-water metals/cofactors, and performs ligand-aware restrained local
minimization. Typed reports include every added/skipped segment, refinement
energy, force field, ligand context, Gemmi assembly policy, and lineage. It
fails instead of publishing an uncleaned fallback.

Remaining specialized work beyond this noncovalent HiQBind-parity core:
preserving arbitrary modified-residue chemistry through generated force-field
patches instead of parent-residue normalization; covalently linked ligands and
polymeric cofactors; explicit retained-water selection; user-selected
histidine/protonation microstates and disulfides; atom/residue mapping across
all assembly copies; and regression fixtures for each case.

### 4.2 Compound and library management

Priority: highest, because screening and RBFE require stable identity.

- [x] Add typed reusable compound-import jobs and compound-set artifacts.
- [x] Import SDF/SMILES/TXT/CSV and complete Excel workbooks with worksheet and
  configurable ID/SMILES column selection while retaining all source columns.
- [x] Validate and describe structures with RDKit without property thresholds;
  retain invalid rows and multi-component sources separately.
- [x] Review rejected structures one at a time against PubChem with immutable
  accept/skip provenance and a single explicitly chosen parent component.
- [x] Report parent-aware duplicates and publish a one-row-per-unique-parent
  docking-ready view while preserving all source aliases.
- Add explicit protonation/tautomer/stereo enumeration, conformer generation,
  and library sharding as derived jobs rather than changing imports.
- Preserve parent compound versus protonation/tautomer/stereo variant IDs.
- Add substructure, similarity, property, and text search.
- Export any selected subset without losing provenance.

### 4.3 Pocket and binding-site prediction

Priority: high.

- Build a CPU fpocket image first; normalize ranked pockets, residues, centers,
  boxes, volumes, and druggability descriptors.
- PeSTo CUDA 12.8 ligand-interface inference is implemented and presented as a
  ligand-interface indicator. Its score and spatial grouping controls remain
  explicit; a holo comparison is optional future evaluation rather than a
  prerequisite for using the adapter.
- [x] Add P2Rank as a complementary CPU detector. The pinned 2.6-alpha.5
  worker adapter preserves native pocket/residue tables and SAS points and
  publishes normalized, target-linked pocket artifacts without a GPU lease.
- Add mdpocket only after trajectory artifacts are normalized.
- Build an interactive pocket editor using Mol* selection and save pocket assets.
- Compare methods and merge overlapping pockets without discarding method scores.

### 4.4 Docking and virtual screening

Priority: high.

- Consolidate the working embedded docking route and the generic docking pages
  behind one adapter contract.
- Build/pin separate Vina, GNINA, and Uni-Dock Pro images.
- Verify GNINA and Uni-Dock Pro on RTX 5090 with actual docking fixtures.
- [x] Add redocking benchmark jobs with symmetry-aware ligand RMSD and pose
  recovery. The typed parent fans out Vina, GNINA, and Uni-Dock Pro replicates,
  retains native child poses, publishes top/best-of-N RMSD and sample statistics,
  and emits crystal/top-pose overlays.
- Add receptor/pocket/ligand matrix execution and library sharding.
- Add checkpointed shard status, failed-ligand reports, rerun-failed, and resume.
- Normalize scores without pretending they share the same physical units.
- Add consensus ranking, diversity clustering, and candidate promotion.
- Use OpenVS algorithms selectively after separating optional Rosetta and CSD
  dependencies; do not make them prerequisites for the core screen.

### 4.5 Generative molecule design

Priority: after compound sets, pockets, and screening are stable.

- Implement a generic generator contract: pocket/pharmacophore/scaffold inputs,
  generated compound set output, seeds, model metadata, validity report.
- Integrate OMTRA first, then PocketXMol and GenMol.
- Add DrugRPG as a comparison engine after rebuilding its mixed CUDA dependency
  stack for Blackwell.
- Treat PFM, PocketFlow, and PGMG as experimental until dependencies, weights,
  licenses, and custom-pocket inference pass acceptance tests.
- [x] Run generated compounds through a common SMILES-first chemical and 3D
  qualification gate: RDKit plausibility/SA/reactivity checks, deterministic
  ETKDGv3 plus MMFF94s/UFF, and PoseBusters molecule-only validation now publish
  the sole typed downstream compound set.
- Add novelty, diversity, broader medicinal-chemistry/property filters, and
  target-specific docking/cofolding filters after the common qualification
  gate.
- Record generated-to-parent/scaffold relationships and failed molecules.

### 4.6 MD and trajectory analysis

Priority: medium-high; OpenMM and GROMACS now share the workflow contract, but
native GROMACS qualification and advanced analysis remain.

- [x] Move the current MD Simulation orchestration behind the unified worker/job
  contract. Preparation and production finalize in the worker, pending children
  advance on worker startup/completion, and page rendering no longer advances
  workflow state. Hidden legacy MD routes remain compatibility surfaces.
- [x] Replace the two primary user-facing MD setup/production pages with one
  `MD Simulation` workflow page while retaining preparation, equilibration, and
  production as separate child-job types.
- [x] Provide `New MD simulation` and `Reuse prepared system` modes; default to new
  preparation for a new complex.
- [x] Add a prepared-system compatibility fingerprint covering source structure and
  ligand hashes, atom mapping, topology, force fields, charges, solvent, ions,
  box, constraints, engine, and preparation settings.
- [x] Permit reuse only for compatible replicas or checkpoint continuation. Route
  changed equilibration conditions through a new equilibration child job and
  system-defining changes through a new preparation child job.
- [x] Remove host/source-tree path assumptions from restart and MM/PBSA paths.
- [x] Add multiple independent production replicates and aggregate statistics.
- [x] Express production protocols in ns, preserve exact steps and timestep in
  provenance, and clearly separate technical smoke from scientific presets.
- [x] Replace generic duration labels with Smoke, Ligand MM/GBSA, Stability,
  and Manual. Stability defaults to three independent 100 ns trajectories;
  Ligand MM/GBSA defaults to three independent 50 ns trajectories and automatic
  immutable endpoint jobs. Explicitly avoid convergence claims.
- [x] Run endpoint MM/GBSA as one immutable child per completed production
  replica and aggregate available terms with sample SD.
- [x] Keep automatic MM/GBSA as an opt-in convenience while making post-run
  evaluation the recommended default. Hide backend/window/frame sampling under
  advanced controls and retain repeated immutable post-run evaluations.
- [x] Browse trajectories as on-demand static frames and provide a downloadable
  or explicitly launched local PyMOL trajectory script.
- [x] Add a Roe–Brooks 2020-inspired OpenMM preparation choice with staged
  protein/ligand heavy-atom restraint release, production-like unrestrained NPT
  density stabilization, the three published plateau criteria, preserved
  density CSV/JSON artifacts, and result-page reporting.
- [x] Default to Roe–Brooks preparation, hide ignored legacy step presets in
  that mode, and reveal the old preset/step controls only in explicit
  compatibility mode.
- Validate the Roe–Brooks-inspired path with a small native protein-ligand
  fixture on the installation-managed CUDA image. Confirm stage-by-stage
  restraint parameter values, zero ligand restraint before density sampling,
  restart artifacts, and density-gate pass/fail behavior. The implementation is
  intentionally not called an exact reproduction while OpenMM H-bond
  constraints and HMR remain active during staged minimization.
- [x] Add shared DCD/XTC stability reports for protein and ligand RMSD/RMSF,
  contacts, hydrogen bonds, secondary structure, radius of gyration, and
  persistent interaction networks.
- Add trajectory clustering and representative complex export.
- [x] Add a CUDA 12.8 GROMACS 2026.3 image definition and worker adapter with
  reproducible Roe–Brooks `.mdp` templates, molecule-local restraints,
  double-precision minimization, GPU dynamics, and typed native artifacts.
  The image and shortened 4LNW workflow smoke are verified on RTX 4090;
  qualifying the image on an RTX 5090 remains pending.
- [x] Add GROMACS as an explicit engine-specific preparation workflow from the
  source complex, never as an implicit OpenMM checkpoint continuation.
- [x] Normalize GROMACS PBC for shared MDTraj/MDAnalysis trajectory analytics.
  Add mdpocket only after native fixture validation.
- [x] Add campaign-level repetition extension that preserves existing children
  and queues only missing repetitions under the same campaign identity.
  Completion is validated from engine-native outputs rather than directory
  presence, and recovery queues exact missing IDs for Vina, GNINA, Uni-Dock
  Pro, RosettaLigand, Boltz-2, Nesso-1, and AlphaFold 3 seeds. Internal gaps are
  supported without overwriting successful repetitions, for both Docking /
  Cofolding and Redocking / Refolding campaigns.
- [x] Add opt-in cumulative MD-duration extension under the same workflow ID.
  OpenMM restores its serialized System/Integrator plus endpoint checkpoint and
  appends DCD/log output. GROMACS extends the TPR and resumes its complete native
  CPT/XTC/EDR/log set with `mdrun -cpi ... -append`. Both paths fail closed on
  incomplete restart artifacts and replace downstream endpoint/aggregate
  analysis children after successful continuation.
- Validate the GROMACS duration-extension adapter on a completed native
  production fixture, including checkpoint output checksums, full-trajectory
  timing, final-frame export, and recomputed endpoint analysis.

### 4.7 Energy calculations

Priority: medium-high, with validation before expansion.

- Establish fixture-based OpenMM MM/GBSA and AmberTools MM/PBSA regression tests.
- Validate topology/trajectory atom identity, frame selection, units, and receptor
  and ligand masks.
- Add replicate statistics and per-residue decomposition where supported.
- [x] Add immutable GROMACS `g_mmpbsa` endpoint jobs using real trajectory-time
  windows. Keep `gmx_qk` outside the runtime because its broader scripts are not
  needed for this contract.
- [x] Bridge GROMACS 2026 production to `g-mmpbsa` 3.0.13 with an isolated
  GROMACS 2025.4 endpoint TPR and reject its process-zero/empty-output failure
  mode.
- Validate GROMACS `g_mmpbsa` atom identity, masks, units, and energy values
  against a small native protein-ligand fixture.
- Pin an OpenFE version and create small ABFE/RBFE end-to-end test systems.
- Add overlap, convergence, cycle closure, failed-window, and uncertainty reports.
- Persist the RBFE network as an editable artifact and support safe partial resume.
- Separate smoke/quick settings from scientifically meaningful production
  presets in both labels and results.

### 4.8 ADMET and QC

Priority: medium.

- Add model version, applicability-domain warnings, and calibration metadata to
  ADMET-AI results.
- Add transparent RDKit descriptor/rule baselines.
- Remove the mock QC result path.
- Add an ORCA external-install adapter with license/readiness checks, or add Psi4
  and xTB open images for supported calculations.
- Normalize optimized geometry, charges, orbital energies, frequencies, method,
  basis, solvent model, convergence, and units.

## 5. Workflow jobs to implement

### Workflow A: Experimental complex to validated MD

```text
import -> HiQBind repair -> ligand chemistry validation -> prepared complex
       -> system preparation -> equilibration -> replicate production MD
       -> stability/contact analysis -> representative structures
       -> optional MM/GBSA or MM/PBSA
```

### Workflow B: Sequence/apo target to docking-ready receptor ensemble

```text
sequence or apo structure -> Boltz-2 / Protenix / optional AF3 children
-> confidence comparison -> selected models -> repair/protonation
-> pocket detection -> receptor ensemble
```

### Workflow C: Large virtual screen

```text
prepared receptor ensemble + pocket + compound library
-> library preparation/sharding -> Uni-Dock Pro fast screen
-> GNINA rescore/refine -> consensus/diversity/property selection
-> promoted prepared complexes -> short MD/MMGBSA
```

### Workflow D: Pocket-conditioned generation

```text
prepared pocket + optional pharmacophore/scaffold
-> OMTRA / PocketXMol / GenMol child jobs
-> normalize/deduplicate/filter -> dock/rescore
-> diversity selection -> complex prediction or MD promotion
```

### Workflow E: Lead-series free energy

```text
related prepared ligands + common receptor pose
-> mapping diagnostics -> RBFE network plan -> edge simulations
-> convergence/cycle closure -> relative ranking with uncertainty
```

## 6. Proposed implementation phases

### Phase 0: Make current claims honest

- Label or remove generic smoke-only docking and batch-docking routes.
- Replace QC mock execution with `not configured` until a real backend exists.
- Mark OpenFE presets as smoke/experimental until validated.
- Fix custom import normalization.
- Add a current-feature matrix to the UI/docs.

### Phase 1: Portable foundation

- Central runtime and persisted Settings.
- Relative artifact references and artifact manifest.
- Unified jobs, manifests, Docker runner, worker, GPU/resource queue.
- Typed module registry with declared inputs/outputs and reusable artifact links.
- Upstream/downstream graph, fan-out/fan-in, clone, and guarded deletion.
- Unified Jobs page, Settings page, and hidden lineage-aware result route.
- Legacy run indexer and migration tests.
- `mn-ligand doctor` and image verification.

### Phase 2: Reliable preparation and assets

- Target, compound, pocket, pose-set, and complex asset stores.
- Separate protein-import and protein-cleaning jobs with reusable prepared targets.
- Shared asset selectors and detail pages for targets, pockets, compounds, poses,
  complexes, trajectories, and energy results.
- HiQBind adapter.
- Compound-library preparation.
- fpocket/P2Rank and interactive pocket assets.

### Phase 3: Docking and screening

- Vina, GNINA, and Uni-Dock Pro adapters.
- Redocking benchmarks.
- Resumable sharded virtual-screening workflow.
- Ranking, clustering, comparison, and candidate promotion.

### Phase 4: Prediction and generative design

- Hardened batch Boltz-2 and affinity parsing.
- Protenix adapter and AlphaFold 3 production hardening.
- OMTRA, PocketXMol, and GenMol.
- Common generated-compound evaluation workflow.

### Phase 5: MD and energy maturation

- Replicate-aware OpenMM workflow and analysis.
- GROMACS adapter.
- Validated endpoint energy calculations.
- Production-tested OpenFE ABFE/RBFE with convergence reporting.

### Phase 6: Packaging and release qualification

- Pinned image digests and source commits.
- License/model registry and external-resource installer guidance.
- RTX 5090 CI/smoke suite plus a CPU-only suite.
- Fresh-machine installation test from built wheel and published images.
- Example datasets and end-to-end workflow fixtures.

## 7. First concrete engineering slice

The best first slice is deliberately foundational but user-visible:

1. Add `runtime.py` and a Settings page for app home, runs, references, libraries,
   and temp directories.
2. Add the unified job/artifact contract and index existing structure jobs.
3. Fix custom protein/ligand import to emit a valid prepared-complex artifact.
4. Convert the current working PDB, docking, and Boltz routes to typed artifacts.
5. Add `mn-ligand doctor` with Docker, path, image, GPU, CUDA, PyTorch, and OpenMM
   checks.

This slice removes the largest portability and workflow blockers without waiting
for new scientific engines.

## 8. Step-by-step safe implementation instructions

The migration must be additive. Keep current pages and historical run folders
working until the replacement path reaches parity.

### Step 0: Protect and record the baseline

1. Work on a dedicated feature branch.
2. Do not move or rewrite `mn-ligand-workdir` or historical JSON files.
3. Record one known successful run for PDB preparation, docking, Boltz-2, MD
   preparation, MD production, ADMET, and OpenFE where available.
4. Create compact regression fixtures; omit large trajectories and caches.
5. Record current image tags, package versions, and expected artifact names.

Exit gate: the existing app starts and all selected historical results still open.

Deployment update (2026-07-22): the user-authorized storage migration moved the
complete app home, without rewriting or renaming any historical run folder, to
`/mnt/data/RESULTS/mn-ligand-workdir`. The former checkout path is retained as a
compatibility symlink so old absolute diagnostic and provenance values continue
to resolve.

### Step 1: Add characterization and handoff tests

Add tests for:

- PDB import/preparation producing protein, ligand, and complex artifacts;
- docking-derived complex to MD handoff;
- Boltz-2 output registration;
- MD system preparation to production handoff;
- strict checkpoint continuation;
- current and legacy path resolution;
- current job collectors finding historical runs.

Exit gate: tests describe current behavior before structural refactoring begins.

### Step 2: Centralize runtime paths without moving data

1. Add ligand-owned runtime functions for app home, runs, references, libraries,
   temporary files, and persisted configuration.
2. Preserve current environment variables and exact defaults.
3. Replace page-local path calculations incrementally.
4. Add a read-only Settings page showing effective values and path readiness.
5. Add persistence only after read-only resolution tests pass.

Exit gate: launching without new options uses the existing runtime directories;
custom paths also work on a test app home.

### Step 3: Introduce typed artifacts alongside legacy files

Implementation status (2026-07-20): the versioned `ArtifactRef`,
`ArtifactManifest`, and normalized `JobRecord` foundation is implemented.
Structure jobs publish run-relative manifests, and historical structure files are
indexed in memory without rewriting old runs. Cross-run input selection and the
remaining task-group compatibility indexers are still pending.

1. Add `ArtifactRef`, `ArtifactManifest`, and schema versions.
2. New jobs write run-relative `artifacts.json` while preserving current native
   filenames.
3. Add a compatibility indexer that recognizes historical filename patterns.
4. Resolve cross-run inputs by run ID plus relative artifact path.
5. Never bulk-rewrite existing run metadata.

Exit gate: old and new structure jobs appear through one artifact API.

### Step 4: Add the unified Jobs page in parallel

Implementation status (2026-07-20): the read-only unified Jobs page is now the
default route and indexes every current task group through `JobRecord`. It has
status, task, tool, workflow, warning, date, GPU, and text filters. The hidden
common result route provides Overview, Artifacts, Metrics, Viewer, Lineage, and
Logs tabs, with links to specialized result viewers. Existing specialized Jobs
pages remain available; unified destructive actions and downstream artifact
actions are still pending validation.

1. Add one read-only Jobs page that indexes every existing task group.
2. Provide status, task, tool, workflow, warning, date, GPU, and text filters.
3. Keep every old Jobs page accessible during validation.
4. Add a hidden common result route with Overview, Artifacts, Metrics, Viewer,
   Lineage, and Logs tabs.
5. Add destructive actions only after indexing and result routing are proven.

Exit gate: every historical job opens correctly from the unified page.

### Step 5: Separate protein import and cleaning

Implementation status (2026-07-21): `Protein Import` and `Protein Cleaning /
Repair` are separate job groups with hidden direct routes. The visible
`Structure Import` workspace combines these steps as one guided path.
Imports publish immutable
run-relative `imported_target` and `import_report` artifacts. Cleaning consumes
the imported artifact by reference through a read-only container mount and
publishes `prepared_target` and `repair_report` artifacts with parent lineage.
The PDB Structure Import path creates import and cleaning
child jobs, and its custom path can reuse a prepared target for multiple ligand
jobs. Common results expose the corresponding downstream actions.

1. Create `Protein Import` producing an immutable `imported_target`.
2. Create `Protein Cleaning / Repair` consuming it and producing
   `prepared_target` plus `repair_report`.
3. Initially call the existing preparation functions rather than rewriting the
   chemistry implementation.
4. Keep the current Structure Import page as a compatibility workflow that
   launches import, cleaning, ligand preparation, and complex preparation jobs.
5. Add downstream `Used by` actions from the prepared target.

Exit gate: one cleaned target can be reused by at least two downstream jobs
without rerunning cleaning.

### Step 6: Introduce reusable workflows and parent/child jobs

Implementation status (2026-07-20): the versioned workflow contract is now
implemented in `mn_ligand/core/workflows.py`. Workflow parents are ordinary
discoverable jobs under `runs/workflows/<workflow-id>/`; they declare expected
steps, retain reusable run-relative artifact inputs, reference ordered child
jobs and dependencies, and aggregate child status and progress. Child metadata
retains its direct scientific `parent_run_id` and adds workflow ownership. The
PDB protein-complex preparation path is the first migrated workflow, composing
protein import, protein cleaning, and final complex preparation. Jobs and common
results show workflow progress and lineage.

1. Migrate additional compatibility pages to the same workflow API one at a
   time; do not duplicate their scientific runners.
2. Add queue-driven child transitions and dependency blocking when the queue is
   introduced.
3. Add resume, retry, cancellation propagation, and optional-child policies.
4. Store immutable workflow templates separately from instantiated workflow
   records so a configured workflow can be reused with new artifact inputs.

Exit gate: one workflow parent can compose reusable jobs, report their progress,
and retain both direct dependency and workflow lineage without copying outputs.

### Step 7: Build the combined MD Simulation workflow

Implementation status (updated 2026-07-28): the combined `MD Simulation` page
now creates an `md-simulation` parent workflow. New simulations consume a typed
prepared-complex artifact and default to fresh system preparation/equilibration;
reuse mode accepts a completed prepared system. Production replicas remain
separate children and are unlocked only after preparation succeeds. The page
uses ns-based named presets while recording exact steps, the engine timestep
(default 4 fs/HMR or selectable 2 fs/normal masses for either engine), frame
interval, seed, and replica Roe
density-revalidation window.
OpenMM, GROMACS, or both can be selected; paired launches share a comparison
group but retain separate preparation, checkpoints, and trajectories. Engine
selection uses checkboxes plus select/deselect actions, and each engine has an
expandable panel with independent protein, ligand, water, box, and condition
settings. Recommended defaults are Amber ff14SB-family/OpenFF 2.2.0/TIP3P for
OpenMM and ff14SB/GAFF2/TIP3P for GROMACS. Roe's truncated-octahedral TIP3P box
with 1.0 nm padding is selectable for both engines. Optional
endpoint analysis creates one
immutable `md-mmgbsa` child per completed replica; it is not embedded in
production. Optional replica analysis is a final child and publishes per-replica
CSV/JSON plus mean and sample SD. Prepared systems receive a compatibility
contract and fingerprint, while preparation, trajectory, checkpoint, endpoint
energy, and analysis outputs receive typed artifact manifests. The established
preparation and production pages remain available during parity testing.

Historical production trajectories are listed from the MD Simulation Results
tab. The result viewer now extracts one aligned DCD frame on demand instead of
retaining a full multi-model PDB in browser session state. Full trajectories can
be opened using a generated PyMOL script or an explicit local-PyMOL launch
button. Existing historical runs and relative artifact paths remain unchanged.

Restart semantics were tightened on 2026-07-21 after reviewing the former
coordinate-production path. Exact continuation now requires the checkpoint plus
the serialized OpenMM System and Integrator, performs no minimization or velocity
replacement, permits one trajectory, and fails instead of silently falling back.
Independent replicas restore the serialized prepared System and the NPT-final
positions and periodic box from a serialized OpenMM State, receive deterministic
recorded seeds, disable restraints, and run unrecorded production-like NPT until
the replica-specific Roe density plateau passes. The configured duration is a
minimum and can extend in increments up to a maximum before failing closed.
The PDB remains topology input but is not trusted as the box
state. The seed is applied to velocity generation, the stochastic integrator,
and stochastic forces such as the barostat. System-defining or equilibration-setting
changes still require a new preparation/equilibration job.

GPU smoke status (2026-07-21): validated in `ovolig-md-cu128:latest` with OpenMM
8.2 on host GPU 1 using the 58,758-atom 4LNW prepared system. A 10 ps exact
continuation loaded the bundle checkpoint with no minimization, velocity reset,
or fallback. The final NPT to first logged production volume changed from
589.338 to 589.517 nm3 (0.03%); first-frame protein and ligand heavy-atom RMSD
were 0.54 and 0.23 A. Two short State-based replicas started at exactly
589.337954 nm3, used distinct recorded seeds, remained unrestrained, and produced
distinct trajectories. A missing checkpoint hard-failed. The smoke also fixed
GPU precision preservation during context reconstruction, restart audit metadata,
final-PDB box vectors, State-based replica boxes, and unnecessary exact-restart
reparameterization. Production-length stability remains a scientific validation
task, not an engineering gate.

1. Keep the existing MD system-preparation and production code operational.
2. Add one `MD Simulation` page that creates a parent workflow.
3. For `New MD simulation`, create preparation, equilibration, production
   replica, and optional analysis child jobs.
4. For `Reuse prepared system`, require a prepared-system artifact and validate
   its compatibility fingerprint before queuing production.
5. Create a new equilibration child if equilibration conditions changed.
6. Create a new preparation child if any system-defining input changed.
7. Keep the old MD pages available until new and legacy result parity is tested.

Exit gate: new simulation, additional replica, and checkpoint continuation each
take the correct path without silently rebuilding or reusing a system.

### Step 8: Migrate downstream modules to typed inputs and outputs

Implementation status (updated 2026-07-24): Pocket Detection was the first migrated
module. Its Target, Tool, Run, and Results tabs consume reusable
`prepared_target` or `prepared_receptor` artifacts. Bound-ligand sites are
derived directly on the Docking page rather than submitted as detection jobs.
The page now only creates a typed queued job and returns; the durable worker owns
fpocket/PeSTo container execution, native logs, failure handling, normalized
artifact publication, and GPU leasing.
The Target tab reports prior pocket jobs for the selected target. The Tool tab
offers independent fpocket, P2Rank, and PeSTo checkboxes with grouped
engine-specific parameters, and one Run action can queue every checked engine.
Generic Results reconstruct the complete prepared target and highlight the
selected pocket residues and docking box instead of displaying the isolated
pocket fragment alone.
The pinned CPU image `ovolig-fpocket:latest` builds fpocket 4.0 commit
`4bb0d8447f62fee77e2c3c29f54b5fcaf5e2c066`. Its normalized `pocket_set` schema
contains ranked pockets, method and source identity, scores, druggability,
descriptors, lining residues, centers, docking boxes, and portable per-pocket
structure/point paths. Jobs publish typed `pocket_set`, `pocket`, and
`pocket_points` artifacts. The image and adapter passed the upstream 1UYD smoke
fixture with host-owned outputs; fpocket remains CPU-native and requires no CUDA
or GPU allocation. Its runner omits `--user` for a detected rootless Docker
socket and uses the host UID/GID for rootful Docker so result ownership remains
portable. The rootless/rootful ownership policy is now part of the shared runner
used by every current command builder; page-to-worker ownership remains
incremental outside Pocket Detection.

The PeSTo engine runs the installed or rebuildable `mnprot-pesto-cu128:latest`
image with `--interface ligand`, mounting the configured
`pesto/i_v4_1/model_ckpt.pt` checkpoint read-only. It publishes the raw residue
probability CSV, a score-encoded PDB, and compact ranked `pocket_set`, `pocket`,
and `pocket_points` artifacts produced by thresholding and complete-linkage
spatial clustering. A GPU 1 smoke run on target C9249 completed with two sites.

Recommended order:

1. Pocket prediction consumes `prepared_target`.
2. Docking consumes `prepared_target`, `pocket`, and `prepared_ligand_set`.
3. Structure prediction/refolding produces predicted target/complex artifacts.
4. Compound preparation creates stable compound and variant identities.
5. Virtual screening composes preparation, sharded docking, rescoring, and
   selection child jobs.
6. Generative design produces normalized compound sets.
7. Endpoint and alchemical energy jobs consume typed complex/trajectory inputs.

The Discover sidebar is now reserved for Pocket Detection, Docking / Cofolding,
Redocking Benchmark, Virtual Screening, and
Generative Design. These routes are artifact-only: target and ligand file/SMILES
ingestion remains exclusively under Structure Import. Prediction selects a
prepared target and optional prepared ligand; docking selects a prepared target,
pocket, and prepared ligand set; screening selects a prepared target, pocket,
and compound set; design selects a pocket and optional prepared reference assets.
Job submission remains disabled for adapters that have not yet been migrated to
these contracts. Classical docking and AlphaFold 3 refolding are executable on
this page. AutoDock Vina now queues a worker-owned campaign and returns
immediately; GNINA and Uni-Dock Pro now use the same worker lifecycle with
explicit GPU leases. This keeps navigation stable without creating fake
scientific jobs or parallel import paths.

Docking / Cofolding is one ordinary-campaign page. It selects one or more typed
targets, optional target-specific pockets for classical engines, and one
completed imported compound dataset. Its Compounds tab defaults to manual
selection with the first parent visibly checked; users change checkbox rows or
explicitly choose all unique parents.
Launch materializes the exact membership as
an immutable typed `compound-selection` job consumed by every queued
target/engine pair under one required campaign name and ID.
Uni-Dock Pro, Vina, GNINA, RosettaLigand, Boltz-2, AlphaFold 3, and Nesso-1
are independent checkboxes with grouped settings; a target-by-engine matrix
controls the pairs queued by one Run action. Pocket guidance and docking boxes are
consumed only by checked classical engines. AF3 reports ranking score, ipTM,
pTM, disorder, and clash confidence as structural evidence, not affinity.
The old one-complex Redocking Benchmark route is hidden for retained links. New
benchmark campaigns start from an immutable imported Benchmark Dataset and
cover Vina, GNINA, Uni-Dock Pro, RosettaLigand, Boltz-2, AlphaFold 3, Nesso
affinity-only, GNINA score-only, and case-compatible Boltzina. Dataset Results
aggregates ligand RMSD and recovery without inventing a structural result for
affinity-only engines.
Both ordinary classical docking and redocking use the shared search-region
control: fixed 20 × 20 × 20 Å is the default, while padding mode defaults to
15 Å on each side and derives each target's effective dimensions from its
selected source region. Multi-target campaigns use one alignment choice and
shared box dimensions, but derive an automatic center independently for each
target from its associated ligand or selected pocket; no per-target parameter
expanders are rendered. An on-page selector switches one 3D viewer among the
selected models. The effective
box and sizing provenance are stored with every job.
Docking / Cofolding repetitions now use one shared execution control: Vina,
GNINA, Uni-Dock Pro, all RosettaLigand protocols, Boltz-2, and AlphaFold 3
default to one but accept up to 100 explicitly seeded attempts after one frozen
preparation. For AF3 the shared value is passed as its native model-seed count.
Multi-run results publish per-run values and aggregate mean/sample-SD summaries.
Boltz-2 diffusion samples remain a separate within-run setting. Nesso-1 retains
its separate affinity-repetition control. Nesso multi-run summaries include log-space statistics plus arithmetic,
geometric, and sample-SD IC50 values in µM without claiming a structure output.
Structure prediction
belongs under Prepare, where sequence/template inputs and predicted structures
can be normalized before reuse. Uni-Dock Pro, Vina, GNINA, and AlphaFold 3 are
executable without reintroducing downstream upload controls. Boltz-2 now has the
same typed target/compound campaign contract and is executable from this page.

The Docking / Cofolding page now follows the workflow organization used by the
protein-design and earlier docking applications: `Target & Pocket`, `Compounds`,
`Engines`, `Run`, and `Results`. Runtime controls, missing-input navigation,
launch actions, live CPU/queue/lease metrics, and per-GPU availability are
contained in `Run`; active and result tables are reconstructed from the run
store. Result rows link to the separate common result page. The initial
Uni-Dock Pro command contract uses `udp`, batched `ligand_index.txt`, explicit
classic/hybrid prerequisites, fast/balance/detail search modes, Docker GPU
device selection, and workspace-relative container paths. The migrated setup
retains the earlier editable box, scrub pH, tautomer, search, pose, and advanced
engine controls. CSV, SMILES, and SDF compound sets are normalized to stable
IDs, prepared as one indexed campaign, and published as typed pose-set, score,
and log artifacts. A real 4WBK/STE UDP smoke run passed on GPU 1.

Target selection is now a shared inventory component across Pocket Detection,
Docking / Refolding, Virtual Screening, MD Simulation, Free Energy, and the
prepared-target handoff in Structure Import. It supports searchable and
categorical filtering, single-row selection, provenance and manipulation
columns, and an embedded PDB viewer. Known bound ligands and typed pocket
artifacts can supply the displayed binding-site box. Pocket lists are filtered
by target lineage, and RBFE uses the corresponding multi-row selector. Hidden
legacy task routes remain compatibility surfaces and should be retired after
their old deep links and job handoffs have migrated.

CSV compound imports now allow explicit ID and SMILES column mapping. The
original upload is retained as `source_compound_dataset`; a canonical
`compound_id,smiles,...` CSV is published as the reusable `compound_set`, with
extra source columns preserved for later filtering and analysis.

Add Targets, Pockets, Compound Sets, Pose Sets/Complexes, Trajectories, and
Energy Results as supporting asset views during these migrations. They show
immutable producer references, `Derived from` and `Used by` lineage, downstream
actions, clone behavior, and guarded deletion. Asset views are not a separate
gate before the combined MD workflow.

Keep filename compatibility fallbacks until each module passes handoff tests.

Exit gate: every migrated module consumes declared artifact types rather than
page-specific paths or filename globs.

### Step 9: Add the Docker tool registry, resource queue, and diagnostics

Implementation status (2026-07-22): the first versioned registry is bundled in
`mn_ligand/manifests/tools.json` and has typed validation in
`mn_ligand/core/manifests.py`. It declares images, artifact contracts, CPU/GPU
resources, references, license status, health checks, and integration readiness
for 16 app roles across 12 unique images. The scope is the 11 image families
already referenced by mn-ligand plus `openvs:local`; installed protein-design
images are deliberately excluded. Current statuses distinguish validated,
implemented, experimental, compatibility-only, disabled, and candidate paths.
The QC image remains disabled because its page path is mock-only, OpenFE remains
experimental, the generic docking image remains compatibility-only, and OpenVS
was initially kept as a candidate rather than an implementation claim.
`mn_ligand/core/docker_runner.py` builds shell-free commands with
explicit GPU selection, portable mounts, and rootless/rootful ownership and can
write command provenance. `mn-ligand doctor` and the Settings page now report
path, Docker, NVIDIA GPU, image, reference, and unpinned-digest readiness.

The next Step 9 slice added `mn_ligand/core/resources.py` and
`mn_ligand/core/worker.py`. `mn-ligand worker` now scans the portable run store,
claims a run atomically, acquires a lease for an explicit available GPU, rewrites
`--gpus all` to the selected device, maintains worker and lease heartbeats,
captures native logs, honors cancellation requests, distinguishes native-result
failure from process success, and recovers a stale lease only when its heartbeat
has expired and owning process is dead. The compatibility dispatcher used by
legacy pages now invokes the same worker implementation, while the old global
lock remains temporarily respected so direct legacy runs cannot collide with a
worker job.

All existing app Docker command builders now use the shared registry-backed
runner: protein cleaning, fpocket, PeSTo, Vina, GNINA, Uni-Dock Pro, AlphaFold
3, Boltz-2, OpenMM MD and MM/GBSA recomputation, ADMET, the disabled QC
compatibility route, and experimental OpenFE ABFE/RBFE. New direct and queued
runs write `command.json`; AF3 retains portable placeholder paths. OpenVS remains
registry-backed. As of 2026-07-23 it has a CPU-only executable adapter for the
Rosetta GALigandDock stage: Docker-contained MMFF94/generic-potential ligand
preparation, VSH and VSX protocols, native silent/score preservation, normalized
REU results, extracted complexes, failure handling, Results display, and typed
best-complex handoff to MD preparation. The iterative OpenVS ML selection loop
and optional CSD geometry analysis are still missing.
OpenVS VSH, VSX, and exhaustive convergence now share the ordinary repetition
contract. Its multi-run analysis uses symmetry-aware heavy-atom automorphisms
in the fixed receptor frame before clustering, with a named-atom fallback for
legacy/synthetic outputs.

The current MD Simulation workflow and post-run endpoint evaluation are now
worker-owned. A completed `bound-ligand-md` job can fan out to independent
`md-mmgbsa` jobs without modifying the source directory. The child input records
typed trajectory/topology references and the backend, percentage window, stride,
and GPU choice; the source is mounted read-only. The worker preserves the native
result, rejects native scientific failure even after process exit zero, and
publishes portable `endpoint_energy` plus `native_output` artifacts. OpenMM GPU
smoke run `b43f0147-2b76-4ba9-9156-8dd1817e764a` evaluated one frame from
source production `17b4bfca-629f-407e-9381-615bed7d381a` on GPU 0 and left the
source metadata/input/result hashes unchanged. AmberTools is selectable through
the same job contract but was not re-smoked in this slice.

GROMACS implementation status (2026-07-29): the worker adapter translates the
shared Roe–Brooks stage contract into reproducible GROMACS inputs, publishes
native restart artifacts, retains the native production XTC, creates a
PBC-normalized XTC for shared geometric analysis, and supports immutable
`g_mmpbsa` endpoint jobs. The container definition targets GROMACS 2026.3 and
CUDA 12.8 with AmberTools, MDTraj, MDAnalysis, and pip-distributed `g-mmpbsa`.
Because the wheel embeds a GROMACS 2025 core, endpoint jobs regenerate only a
GROMACS 2025.4 compatibility TPR while retaining the original production XTC
and TPR. Focused workflow/contract tests pass. The image built and passed CUDA
water, shortened 4LNW Roe preparation, 10-step production/shared-analysis, and
one-frame MM/PBSA artifact smoke on RTX 4090. Full density stabilization,
meaningful production sampling, energy regression, and RTX 5090 qualification
remain pending.

Migration smoke status (2026-07-22): fpocket run
`90cfaa74-cd12-45f1-be0c-8965e6ec3b35` produced three normalized pockets from
prepared receptor `54115f0a-d99b-4a8f-8130-c92493468da1`. PeSTo run
`19c32837-aea1-4c13-b8d5-c2fa4a97834c` used GPU 1 and produced two normalized
pockets plus native residue scores and the scored structure. Failed run
`04e489fa-8fbe-4f59-b8ff-498f4160875f` exposed an incorrect direct-API default
image and remains preserved as failure evidence; method-specific image selection
was fixed and regression-tested.

Pocket Detection is now the first end-to-end worker-owned scientific workflow.
Worker fpocket run `0d21cf53-3781-4c2c-ad0c-7df6fdb8d55c` produced eight
normalized pockets. Worker PeSTo run
`9330c259-3554-4dce-89af-e026aea48f24` acquired GPU 1 and produced two pockets,
native residue scores, a scored structure, and normalized pocket-point
artifacts. The page does not launch either container directly, and worker-side
finalization turns missing or invalid native output into an actionable failed
job with an empty typed artifact manifest.

P2Rank is also worker-owned and CPU-native. Image
`ovolig-p2rank:latest` builds local source commit
`d8c8e0d870f79a36b7dbb176edf8c019b6cc789c` (2.6-alpha.5, MIT) on
digest-pinned Java 17 bases. Upstream 1FBL produced four pockets. Worker run
`6fe5efa5-60b8-4225-9f46-4d1a0b68dac9` consumed the typed 4WBK receptor and
published the native pocket-score table, residue-score table, SAS points, and
one normalized pocket with profile-specific probability metadata. It used no
GPU or lease and passed common-result display and Docking artifact handoff.
Invalid-input run `fd765101-5034-4db5-8fbe-af6efd7885e2` caught native
scientific failure despite process exit zero. The full suite passes at **157
tests**.

AutoDock Vina is the next worker-owned path. Run
`a59d4cf8-c58f-4818-a565-9f6c49526cf3` reused typed 4WBK receptor and STE
compound artifacts, completed without a GPU allocation, and published native
Vina poses, normalized docking scores, a typed combined pose set, and portable
logs. Both the Docking page and common Results page rendered the completed run
through Streamlit AppTest. Missing poses are treated as a failed scientific job
even when the container exits zero.

GNINA and Uni-Dock Pro are also worker-owned. GNINA run
`2c2a2671-caf1-43ef-b0ff-482a8777ff30` acquired GPU 0 and published its native
CNN score and affinity, normalized docking scores, a combined typed pose set,
and logs. Uni-Dock Pro run `2f86e28e-6fc8-40f3-a207-e4b91da67c9f` acquired GPU
1 and published native affinity, normalized scores, a typed combined pose set,
and logs. Both reused the typed 4WBK/STE lineage, released their leases, and
passed Docking/Common Results AppTest display. Separate RTX 5090 hardware
validation remains required despite the CUDA 12.8 resource contract.

AlphaFold 3 and Boltz-2 typed refolding are now worker-owned. AF3 run
`f265dbd6-3ad7-4c38-ba85-de2b4594c087` acquired GPU 1, reused the cached target
MSA, and published a native predicted CIF, confidence JSON, and normalized
metrics. Boltz-2 run `4ee3dc88-48c1-4e31-a25f-2a335eeac03d` acquired GPU 0 and
published a native predicted CIF, confidence JSON, affinity JSON, and normalized
metrics. Both used the typed 4WBK/STE inputs, released their leases, and passed
common result display. Exit-zero runs without a predicted complex are rejected.
The older sequence-entry Boltz-2 page remains a compatibility path rather than
the typed Docking / Refolding campaign. Typed AF3 and Boltz-2 campaigns default
to all compounds (`Maximum compounds = 0`); a positive value applies an explicit
campaign cap without changing AF3's separate execution batch size.

Step 9 is not yet complete. Scientific pages outside Pocket Detection,
classical Docking / typed Refolding, and current MD Simulation can still execute
directly in Streamlit. The durable worker is now installed and supervised as
two enabled systemd user-service instances for GPUs 0 and 1, ordered after the
rootless Docker user service. Structured progress, parent-aware workflow-child retry,
parallel execution within one worker, and runtime CUDA/PyTorch/JAX/OpenMM health
checks remain unfinished.

Supervised-worker smoke status (2026-07-22): `mn-ligand-worker@0.service` and
`mn-ligand-worker@1.service` are enabled and active with stable worker IDs and
user lingering enabled. Runs `worker-service-gpu0-20260722` and
`worker-service-gpu1-20260722` were independently claimed by the intended
instances, executed the existing CUDA MD image, recorded native GPU output and
return code zero, and released their leases. Shutdown run
`worker-service-stop-cleanup-gpu0-20260722` interrupted a live Docker command,
preserved the run as failed with interruption provenance, removed its
exact CID-tracked container, released its claim/lease, and allowed both services
to return active. Focused worker/service tests, all 10 Streamlit AppTests, and
the full suite pass at **151 tests**.

Worker health is now visible from Settings without granting the web process
service-control authority. Each worker publishes a durable heartbeat containing
its stable identity, PID, GPU scope, idle/running state, current run, and selected
GPU. Settings combines this with read-only systemd enabled/active/PID state,
queued-job count, and active per-GPU leases; an unavailable user bus or stale
heartbeat produces an actionable non-fatal message. Live GPU 0/1 validation
reported both enabled and `active/running`, fresh idle heartbeats, zero queued
jobs, and zero leases. Focused health/lifecycle/Settings AppTest coverage passed
at 23 tests and the full suite passes at **153 tests**.

Standalone worker cancellation and retry controls are complete. Unified Jobs
and generic Results use one shared API; render/refresh no longer advances MD
workflow state. Cancellation is limited to records with valid worker commands,
queued cancellation is immediate, and the worker terminates running processes
without finalizing partial scientific output. Verified immutable retry profiles
cover Pocket Detection, classical docking, typed AF3/Boltz-2 refolding, and
post-run MD endpoint jobs. A retry creates a new run, copies only staged inputs,
rewrites run-local mounts, starts with empty artifacts, preserves the original,
and records root/previous lineage plus a monotonic attempt number. Workflow
children are explicitly ineligible until a retry can replace the failed child in
its parent workflow.

Real Docker cancellation smoke `worker-control-cancel-smoke-20260722` terminated
an attached `ovolig-md-cu128:latest` sleep container, escalated after the grace
period to return code `-9`, released the worker claim, and left no container.
Focused worker/control/AppTest coverage passes, and the full baseline is
**134 passed**.

CPU/RAM/VRAM/scratch admission is complete. The worker captures CPU capacity,
available/total RAM, free/total scratch filesystem space, and per-GPU free/total
VRAM before process launch. Impossible requests become failed jobs; temporary
shortages stay queued without blocking smaller later jobs. GPU candidates are
filtered by `min_vram_gb` before an exclusive lease is acquired. Admission
request, snapshot, eligible devices, status, and reasons are stored in metadata
and displayed in Unified Jobs and generic Results. Sparse legacy resource
records remain compatible.

Native smoke `worker-resource-admission-smoke-20260722` requested 4 GiB scratch
while the run filesystem had 3.58 GiB free. The worker returned idle, recorded a
waiting reason, released its temporary claim, created no process logs, and
launched no Docker container. The preserved smoke record was then cancelled.
Focused admission/worker/AppTest coverage and the full baseline pass at
**141 tests**.

1. Add a versioned Docker tool registry containing image tags and digests,
   commands, accepted and produced artifact types, licenses, and resource needs.
2. Add RTX 5090 readiness checks for driver, CUDA architecture, PyTorch, JAX,
   custom kernels, and required model weights.
3. Add a durable local worker with heartbeats and structured progress.
4. Move subprocess ownership, logs, cancellation, retries, and resource locks to
   the worker.
5. Use one lock per GPU and declare CPU/RAM/VRAM/scratch requirements.
6. Add installation diagnostics for Docker, NVIDIA Container Toolkit, writable
   runtime paths, image availability, reference data, disk space, and ports.
7. Keep workflow parents responsive while children queue or run and test
   stop/resume for checkpoint-capable tools.

Exit gate: leaving or refreshing a page does not interrupt a running job, and a
new installation can verify every configured tool before starting scientific
work.

### Step 10: Switch navigation only after parity

1. Make unified Jobs the default.
2. Expose independent pages under Prepare, Discover, Simulate & Score, and Assets.
3. Replace separate MD setup/production sidebar entries with MD Simulation and MD
   Analysis.
4. Hide legacy pages for one release while retaining direct routes.
5. Remove legacy pages only after migration documentation and regression tests
   cover all historical result types.

Exit gate: no supported current workflow or historical result requires a removed
page.

### Step 11: Add new tools behind validation gates

Implement HiQBind, mdpocket, additional structure predictors, generators,
GROMACS, and energy/QC tools only after the modular job,
artifact, worker, and UI contracts are stable. Each tool must pass the validation
gates below before being called implemented.

PoseBusters now satisfies the initial dock-mode gate for classical docking and
AF3/Boltz-2 cofolded complexes. Remaining optional extensions are a distinct
reference-aware redocking mode, explicit validation of retained cofactors and
waters when those become first-class prepared-target artifacts, and campaign
aggregation/filtering by failed physical check. These are extensions, not
requirements for the current physical pocket-plausibility workflow.

The current focused selection policy deliberately does not validate every
emitted search pose. It covers the best classical/RosettaLigand pose per
compound and repetition, AF3 sample 0, Boltz-2 model 0, and GNINA's CNN-best
plus Vina/minimized-affinity-best choices (deduplicated when identical). Older
all-pose runs are retained as historical data but require a focused-policy rerun
before entering current comparison statistics.

PLIP and PandaMap now satisfy the initial adapter, native-output, normalized
artifact, source-job coverage, combined launch, result display, and campaign
summary gates. Remaining scientific validation should use a multi-compound,
multi-engine campaign to quantify detector agreement and review disagreements
manually against the displayed complexes. Cofolded residue-number normalization
is implemented and verified for 4LNW (263/263 residues mapped from A:1–263 to
author IDs A:145–407); it still needs scientific spot checks on a multi-chain
target and a target with internal sequence gaps. PandaMap's empirical energy
remains an auxiliary estimate and must not be combined with docking scores or
free-energy results.

Prepared target complexes are now also eligible PLIP/PandaMap sources when an
immutable `prepared_complex` and coordinate-bearing `prepared_ligand_set` are
both present. Non-polymer ligand residues are separately inventoried and the
exact selected residue identity is carried through both native adapters and
normalized outputs. Completed interaction artifacts can initialize a
pharmacophore hypothesis for one exact complex/pose. The present implementation
uses interaction-class and contacted-residue evidence; exact ligand-atom to
pharmacophore-feature association remains missing until the normalized PLIP and
PandaMap schemas expose stable ligand atom identifiers and coordinates.

This prepared-target path still requires native scientific validation with a
complex containing the intended ligand plus at least one competing cofactor,
confirming exact membership, independent PLIP/PandaMap outputs, normalized
identity fields, display, and pharmacophore provenance.

## 9. Validation gates for every new tool

A tool is not “implemented” until all gates pass:

1. Source commit, image digest, code license, weight license, and citation recorded.
2. Image builds reproducibly without depending on an undeclared host path.
3. `doctor` passes on RTX 5090 or the tool is explicitly CPU-only.
4. A minimal fixture completes and outputs non-empty native files.
5. Adapter parses normalized artifacts and validates counts, identity, and units.
6. Failed input produces a failed job with actionable logs.
7. Stop/resume behavior is tested where the native tool supports it.
8. Parent workflow can consume the output by artifact type without filename globs.
9. Result page exposes provenance, warnings, and downloadable native outputs.
10. A regression test protects the tool-to-tool handoff.

## 10. Main risks

- Scientific tools use incompatible and sometimes old CUDA/PyTorch ecosystems;
  rebuilding can alter numerical behavior and requires per-tool validation.
- Model and dataset licenses may prevent image or weight redistribution.
- Absolute paths embedded in historical JSON complicate migration.
- Structure and ligand identity can be lost when converting PDB, mmCIF, SDF, and
  PDBQT without explicit atom/residue maps.
- Screening scale requires streaming and sharding; loading millions of compounds
  into Streamlit or pandas at once will not be viable.
- Free-energy workflows can finish technically while remaining scientifically
  unconverged; completion and convergence must be separate statuses.
- Learned affinity, docking scores, endpoint estimates, and alchemical free
  energies are not interchangeable and need method-specific displays.

## 11. Results discovery and provenance

Implementation status (updated 2026-07-27):

- [x] Add `Results > Results Explorer` as a cross-workflow entry point.
- [x] Browse completed predictions by dataset, biological target,
  prepared-target variant, campaign, compound, and engine.
- [x] Prefer explicit launch-campaign metadata and provide bounded,
  target-aware grouping for historical jobs that predate it.
- [x] Merge separately launched repetitions into one campaign/engine summary
  while retaining links to every immutable job.
- [x] Hide failed and incomplete jobs by default while retaining an opt-in
  troubleshooting view.
- [x] Link prediction, PoseBusters, PLIP, and PandaMap pages to the exact source
  job and requested compound.
- [x] Display source PDB/biological origin, prepared-target code, artifact
  filename, ordered modification history, and target-lineage link.
- [x] Version the cached index schema and tolerate stale rows during Streamlit
  hot reload.

Remaining validation:

- [ ] Spot-check historical grouping on campaigns with interleaved manual
  launches to ensure the time-window inference never joins unrelated work.
- [ ] Validate target-origin presentation for a non-PDB imported structure,
  multi-chain targets, and two targets that reuse the same artifact filename.
- [ ] Add a persistent explicit campaign-repair/membership mechanism if
  historical inference proves insufficient; never rewrite immutable prediction
  jobs merely to improve display grouping.
- [ ] Extend Results Explorer to MD and free-energy workflow families only after
  their parent/child aggregation contract is stable; current coverage focuses
  on docking, cofolding, rescoring, and their pose/interaction evaluations.

## 12. Generate implementation update (2026-07-27)

A dedicated Generate navigation group and shared campaign/conditioning UI
foundation are present. A canonical, editable, immutable pharmacophore job can
be created from a coordinate ligand with RDKit BaseFeatures, an exact
PLIP/PandaMap analysis, a previous hypothesis, or manually. PLIP sources can
retain atom-resolved observed features while adding a distinct mandatory
author-numbered side-chain constraint. It publishes engine-neutral JSON/CSV
plus Pharmit JSON, OMTRA XYZ, and compatible PGMG representations.

Registry and separate CUDA 12.8 image definitions cover OMTRA, PocketXMol,
FLOWR.root, conDitar, paOPT, DrugRPG, PFM, PocketFlow, and PGMG. Their model
files resolve below `generation/<engine>` under the configured reference root;
paOPT shares the permissioned conDitar Diff/PocketAE bundle.

All nine engines have experimental queue/native-command adapters, normalized
generated SDF/CSV/report artifacts, result display, and immutable
SMILES-first chemical/3D qualification children. Only accepted children publish
typed compound-set handoff. Focused native inference has completed for every
engine, and the version-2 historical 13-run backfill accepted 34/42 normalized
molecules: 32 strict PoseBusters passes plus two explicitly reviewable
non-aromatic-ring-flatness warnings. Remaining before validated production use:

- rebuild the nine images after shared runner changes and perform focused
  parameter-handshake checks for each affected native CLI;
- keep conDitar/paOPT source, weights, and images inside the authorized group
  deployment; PGMG remains explicitly non-commercial under CC BY-NC-SA 4.0;
- extend invalid/duplicate-generation accounting beyond the current normalized
  validity/deduplication report where an engine exposes rejected records, and
  add novelty/diversity/property policy layers after chemical/3D qualification;
- [x] Add a target-centric combined molecule-design result page with
  cross-engine metrics, QED/MW plots, property/status filters, standardized 3D
  preview, multi-row promotion, and typed Docking / Cofolding dataset handoff;
- calibrate warm/cold time estimates separately by GPU model as more typed runs
  become available;
- run the target-specific 4LNW/T3 campaign and downstream docking/cofolding plus
  exact PLIP/PandaMap acceptance for mandatory SER277 side-chain binding.
