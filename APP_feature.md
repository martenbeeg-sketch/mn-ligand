# mn-ligand Product and Feature Architecture

For current implementation landmarks and verification commands, read
`APP_DEVELOPMENT.md` first.

Status: product contract grounded in the repository state on 2026-08-03.

## 1. Product purpose

`mn-ligand` should become a local, installable workbench for structure-based
small-molecule discovery. It should connect experimental or predicted receptor
structures, compound libraries, docking, generative design, molecular dynamics,
and binding-energy calculations through reproducible Docker jobs.

The application is not intended to hide scientific choices behind one score.
Every workflow must retain inputs, parameters, tool and image versions, logs,
intermediate artifacts, quality checks, and uncertainty or confidence fields.

The primary deployment target is Linux/x86_64 with Docker, the NVIDIA Container
Toolkit, and an RTX 5090. CPU-only workflows should remain usable without a GPU.

## 2. Product principles

1. **Workflows are first-class jobs.** A workflow owns an ordered graph of child
   jobs and can be stopped, resumed, inspected, or cloned with changed settings.
2. **Artifacts connect tools.** Pages must exchange typed artifact references,
   not infer compatibility from filenames or host-specific absolute paths.
3. **Native outputs are preserved.** Normalized outputs support downstream use,
   while each tool's original files remain available for audit and debugging.
4. **Scientific provenance is mandatory.** Results record source structures,
   ligand identity, preparation decisions, random seeds, image digests, model
   weights, software versions, and warnings.
5. **Screening is progressive.** Cheap methods reduce a library before expensive
   docking, rescoring, MD, and FEP stages.
6. **Predictions are evidence, not truth.** Docking scores, learned affinity,
   MM/PBSA, ABFE, and RBFE results must be labeled by method and never merged into
   an unexplained universal energy value.
7. **Installation paths are configurable.** Runtime data, results, compound
   libraries, temporary files, and reference/model files must not depend on the
   source checkout location.
8. **Each scientific operation is reusable.** Import, cleaning, pocket
   prediction, ligand preparation, docking, refolding, design, MD preparation,
   simulation, and energy analysis are separate jobs. A workflow composes these
   jobs but does not hide or duplicate them.
9. **Physical campaigns and analytical groupings are distinct.** A launch
   campaign records executed compute. An Analysis Set is a saved, editable
   comparison definition over compatible immutable campaigns and never pretends
   that those jobs were one physical launch.
10. **Identity and modeling state are separate chemistry concepts.** Canonical
    parent identity supports deduplication, while the exact charged/protonated
    modeling SMILES is retained and handed to engines with explicit provenance.

## Campaign-results and Analysis Set contract

Campaign Results is the operational surface for selecting a physical campaign,
reviewing its children, extending missing repetitions, and creating/opening
Analysis Sets. Aggregate scientific interpretation belongs in the comparison
workspace. Results Explorer remains a navigation/index surface and must not
grow a second set of analytical selectors.

An Analysis Set may combine target-ligand redocking/refolding campaigns when
they share a canonical reference ligand and workflow purpose. Targets do not
need to be identical: different prepared versions are an intended comparison
dimension and remain separately labeled. MD, folding-only, unrelated compound
screens, and incompatible reference ligands remain separate. Before saving, the
UI must preview the exact campaign IDs, target origins, prepared-target IDs,
reference ligands, engines, and compatibility reason.

The comparison workspace adapts to campaign shape:

- single-target/multi-compound emphasizes ranking and compound drill-down;
- multi-target/single-compound emphasizes preparation sensitivity and target
  comparison;
- multi-target/multi-compound adds a target × compound overview before entity
  drill-down;
- saved Analysis Sets retain their member campaigns and open without a
  redundant physical-campaign selector.

All plots expose the represented normalized data for offline use. Tabular
metric export uses one row per target/compound/engine/metric combination and
one column per configured independent repetition, followed by summary fields.
Selected static figures can be rendered as PNG and downloaded together as a
ZIP.

## Structural comparison contract

Input-pose recovery and model agreement are separate metrics. Recovery compares
each prediction with the coordinate ligand belonging to that prediction's own
prepared target. Agreement compares poses with one another only inside a common
prepared receptor frame. Cofolded complexes are protein-Cα aligned; classical
docking poses remain in the prepared-target frame; the ligand itself is never
independently superposed.

When molecules are identical, symmetry-aware heavy-atom graph mapping is the
primary metric, including equivalent substituents such as the three iodines in
T3. When molecules differ, ligand RMSD is undefined: centroid displacement,
shape/volume overlap, or interaction-fingerprint agreement must be labeled as
different metrics rather than presented as RMSD.

The analysis supports:

- recovery-to-input matrices with prepared targets as rows and engine/ranking
  series as columns;
- one pose-agreement matrix per prepared target;
- an all-target engine matrix that pools only valid within-target pairs and
  reports mean ± sample SD;
- single-target drill-down and individual-pose matrices;
- both GNINA CNN-ranked and empirical/Vina-ranked poses;
- fixed, documented color ranges so visual severity is comparable between
  campaigns;
- a linked 3D counterpart whose target, compound, repetition, GNINA-ranking,
  and presentation state mirrors the RMSD view. A local engine visibility
  filter may reduce 3D clutter without altering the statistical selection.

## MD settings-template contract

Cloning MD settings is distinct from continuing MD. A settings-template launch
creates a new workflow, snapshots a new source complex, performs fresh system
preparation, and creates new independent replicas. It may inherit force fields,
water/box/ion conditions, equilibration stages, timestep/reporting, replica
policy, and endpoint/aggregate-analysis settings. It must not inherit the old
prepared system, ligand coordinates, residue map, checkpoint, trajectory,
children, comparison group, or extension history.

If the template's earlier workflow reached its current duration through an
extension, a new template launch starts directly at that cumulative duration.
For example, a 50 ns workflow later extended to 100 ns becomes one fresh
25,000,000-step production at 4 fs per replica, not a new 50 ns run followed by
an automatic continuation. Template ID and the `template_settings_only` policy
are retained in provenance.

## 3. Recommended application shape

Reuse the successful UI organization pattern observed in `mn-protein-design`,
while keeping every page, entity, and scientific action specific to
`mn-ligand`:

- one unified Jobs page as the default screen;
- Settings beside Jobs;
- one-stop workspaces with subtabs for tightly related preparation steps;
- independent scientific task pages only where the result is reusable on its own;
- workflow/campaign pages that compose reusable jobs;
- asset/library pages that expose reusable outputs;
- hidden result-detail routes opened from job or asset tables.

Recommended sidebar:

```text
Jobs
  Jobs                       # default; all jobs with filters and saved views
  Workflows                  # parent workflows and child progress
  Settings                   # paths, images, models, GPUs, diagnostics

Prepare
  Structure Import            # import, inspect, clean/repair, assemble complex
  Sequence Modification       # trim termini or append modeled C-terminal sequence
  Compound Datasets           # import and normalize reusable compound libraries

Discover
  Pocket Prediction
  Docking / Refolding
  Virtual Screening
  Generative Molecule Design

Simulate
  MD Simulation

Evaluate
  ABFE / RBFE
  Ligand Properties
  Quantum Chemistry

Assets
  Targets / Receptor Ensembles
  Pockets / Pharmacophores
  Compound Sets
  Pose Sets / Complexes
  Trajectories / Energy Results

Hidden routes
  Direct Protein Import / Cleaning
  Legacy MD Preparation / Production
  Specialized Job Tables
  Job Result Details
  Workflow Result Details
  Asset Details
```

The unified Jobs page replaces separate sidebar entries such as Jobs - Structure,
Jobs - MD, and Jobs - OpenFE. It should offer task group, tool, status, workflow,
GPU, date, warning, and text filters. Rows open one result route selected by the
job's declared result renderer.

The sidebar must expose user goals, not internal child-job boundaries. Structure
Import is the primary guided workspace; protein import, cleaning/repair, and
complex assembly remain distinct jobs and artifacts beneath its subtabs and
workflow record. Their direct routes stay hidden for advanced handoffs and old
links, avoiding duplicate primary navigation.

Prepare is the only visible ingestion group. Structure Import owns target
and complex ingestion from PDB, typed Boltz-2/AlphaFold 3 predictions, and
manual files. Target Sequence Modification combines Target Trimming and
C-terminal Repair as separate nested workflows on one Prepare page. Repair
operates on registered prepared complexes and uses the
isolated MODELLER 10.8 environment to append short canonical C-terminal
sequences. It preserves the source complex and ligand, ranks an ensemble by
severe clashes and DOPE score, publishes typed prepared-complex/receptor/model
artifacts, and labels the added coordinates as a flexible modeled terminus
requiring review and equilibration before MD. Target Trimming operates only on
registered imported complexes;
it creates a new lineage-linked complex with chain-aware protein terminal
ranges while retaining the ligand. Compound Datasets owns reusable SDF and
SMILES-library ingestion. Discover, Simulate, and Evaluate pages select
immutable artifacts produced there rather than add independent upload widgets.
PDB, manual-structure, and promoted AF3/Boltz complex paths use the same strict
Ligand-X/PDBFixer/OpenMM cleaning contract. Its HiQBind-compatible safety
profile skips missing termini and gaps longer than 15 residues by default.
Every noncanonical polymer site is resolved explicitly by the user and modeled
with MODELLER; no residue name such as CAS is assumed to have one universal
canonical identity. Eligible internal gaps are modeled as ten independent
MODELLER conformations after the user confirms or supplies the missing sequence
and both observed flanks. PDBFixer repairs missing atoms and OpenMM minimizes
the prepared structure against a fixed SMIRNOFF-parameterized ligand context
while restraining the experimental scaffold. Repair reports
record added/skipped segments and reasons, modified-residue normalization,
missing and terminal atoms, pH, refinement energies, force field, ligand
context, and input normalization; failure prevents publication of a prepared
complex. Gemmi owns macromolecular mmCIF/PDB conversion, sequence-record
preservation, and optional biological assembly construction. RDKit plus CCD
references owns ligand graph identity, bond orders, and normalized SDF/SMILES
artifacts. Open Babel is not used to classify or convert whole predicted
protein-ligand complexes.

Structure Import Results is global rather than scoped to the selected source
tab. It lists every prepared structure job and exposes the complete structure,
typed artifacts, import metadata, repair report, and detailed result route.

Boltz-2 and AlphaFold 3 have separate complex-preparation interfaces within
Structure Import. Each consumes one FASTA/raw protein sequence and one
`LIGAND_ID,SMILES`, uses an `Input`, `Engine`, `Run`, `Completed predictions`
layout, and stages typed sequence/ligand artifacts before worker submission.
This differs from Discover Docking / Cofolding, which applies checked engines
to an already imported target and one or more prepared compound sets. Completed
prediction selectors are origin-filtered to Structure Import runs and render
the selected complex inline. Both engines use local MSA resources: AF3 reuses
the sequence-hashed cache or runs local AlphaFast/MMseqs, while Boltz Structure
Import requires a matching cached local MSA and never enables its MSA server.
Sequence entry accepts only canonical amino-acid codes rather than inventing a
mapping for noncanonical chemistry. Predicted coordinates are cleaned and
repaired, with the ligand retained, when the selected output is promoted.

The Compound Datasets page registers SDF, SMILES text, CSV, and complete Excel
workbooks as immutable artifacts. CSV/Excel imports expose worksheet and column
selection for compound IDs and SMILES. RDKit parsing/sanitization excludes only
chemically unreadable rows from the normalized `compound_set`; the untouched
source workbook, rejected rows, every additional selected-sheet column, and
validation report remain typed artifacts. Results provide dataset summaries,
descriptive numeric-column profiles without screening thresholds, selectable
2D structures, and calculated molecular descriptors on the job-specific
Dataset tab. Invalid rows and multi-fragment components remain separate.
Formulation recognition is driven by a validated, versioned YAML registry;
the active entries and source are visible read-only in Settings, and an
installation may select an extended registry with
`MN_LIGAND_FORMULATION_REGISTRY`. The same normalized dataset can then be
selected by docking, screening, property, QC, and generative-design workflows.
Rejected compounds support explicit one-at-a-time PubChem review by CAS or
product name. The page renders the vendor-side and PubChem structures side by
side and requires the user to confirm a candidate or skip it. Every decision is
an immutable typed child job. Name search combines and deduplicates PubChem
word-name CIDs for both the full vendor label and its base name without a
trailing formulation qualifier, with match provenance shown per candidate.
Accepted structures are re-sanitized by RDKit,
described afresh, and labeled as PubChem-confirmed imports rather than silently
replacing vendor records. Fractional formulation annotations are recognized
for review previews, while exact rational formula comparison distinguishes an
integer-scaled PubChem formulation from a true formula mismatch. The PubChem
component count is shown before confirmation. Accepted review artifacts join
the effective usable dataset immediately. Multi-component PubChem records are
grouped by canonical component; the unique largest component type is the
default parent candidate, but the user confirms the displayed selection.
Only one selected parent is published, while the full formulation and
component multiplicities remain provenance. Resolved source rows remain
available in rejected-row review history.
Parent-aware duplicate reporting operates on the effective source plus
confirmed-review inventory. It compares the unique largest component after
removable-charge neutralization, retains the original and exact-parent SMILES,
separates exact and charge-standardized groups, preserves stereochemistry, and
flags equal-size parent ambiguity. Results are visible and downloadable without
mutating the source dataset. A separate `Docking-ready parents` view contains
one representative row per unique standardized parent, aggregates every source
alias and formulation, and reports the unique-parent and redundant-source
counts at the top of the job result.

Scientific workflow pages use ordered tabs rather than one long setup form:
`Target / Input`, optional additional inputs, `Tool / Engine`, `Run`, and
`Results`. Engine launch and missing-input navigation actions belong in `Run`.
That tab also shows CPU capacity, queue depth, active GPU leases, physical GPU
lease availability, GPU worker-slot state, and the dedicated CPU worker before
submission. Results read persisted jobs rather than page session state and open
the shared result route for detailed structures, artifacts, metrics, lineage,
and logs.

Docking / Cofolding automatically inherits a coordinate-bearing ligand stored
beside the selected prepared target as its reference/template artifact; users
may explicitly override it. Shared target-orientation jobs transform the
receptor and all coordinate references together, and downstream docking,
RosettaLigand, AlphaFold 3, and Boltz-2 records retain the exact transformed
reference for provenance and overlays. Historical results may resolve the same
artifact through target lineage without rewriting immutable job inputs.

Every task page follows the same interaction shape:

1. Select typed input artifacts from reusable asset/job tables.
2. Inspect provenance and compatibility warnings.
3. Configure the tool or choose a preset.
4. Preview the exact job inputs and resource request.
5. Queue the job and return to Jobs or the parent workflow.

Target and prepared-complex inputs use one shared inventory selector rather than
dropdowns. The filterable table shows target identity, receptor/complex kind,
bound ligands, chains, residue count, origin, preparation/manipulation history,
producer tool, job code, and creation time. Selecting one row updates an
adjacent 3D structure viewer. When the workflow has a selected predicted pocket,
the viewer draws that pocket's stored docking box; when a bound ligand defines
the site, it highlights the ligand and derives a padded box from its coordinates.
Pocket choices are restricted to the selected target's lineage. Multi-complex
workflows such as RBFE use the same inventory with multi-row selection.

All future pages that consume `prepared_target`, `prepared_receptor`, or
`prepared_complex` artifacts must reuse this selector. Workflow pages may add
domain-specific filters, but must preserve artifact identity and provenance and
must not introduce another target upload or an independent target list.

Every result page follows common tabs: `Overview`, `Artifacts`, `Metrics`,
`Structure/Trajectory`, `Lineage`, and `Logs`. Tool-specific panels may be added
inside this frame. The Lineage tab supplies `Derived from`, `Used by`, and
contextual actions such as `Use in pocket prediction`, `Dock compounds`, or
`Prepare MD system`.

Workflow pages should resemble campaign pages organizationally: setup, ordered
modules, per-module engine settings, run preview, execution status, and combined
results. They create visible child jobs rather than executing scientific work in
the Streamlit page process.

## 4. Portable runtime and installation contract

### 4.1 Installation

The Python package should remain lightweight and installable with `pip` or
`pipx`. Scientific engines belong in versioned Docker images. A fresh computer
should need only:

- Linux/x86_64
- Docker Engine and Docker Compose
- NVIDIA driver and NVIDIA Container Toolkit for GPU jobs
- the `mn-ligand` Python package
- separately downloaded model weights or licensed programs where required

Recommended commands:

```text
mn-ligand init --app-home /data/mn-ligand
mn-ligand doctor
mn-ligand images verify
mn-ligand app --app-home /data/mn-ligand
```

### 4.2 Configurable paths

Use one ligand-app runtime module. The neighboring app demonstrates a useful
configuration pattern, but this implementation and its settings remain owned by
`mn-ligand`. Use this precedence: CLI option, environment variable, persisted
config, default.

| Purpose | Environment variable | Portable default |
|---|---|---|
| app state | `MN_LIGAND_APP_HOME` | `/mnt/data/RESULTS/mn-ligand-workdir` in this deployment |
| jobs/results | `MN_LIGAND_RUN_DIR` | `<app-home>/workdir/runs` |
| reference/model files | `MN_LIGAND_REFERENCE_DIR` | `<app-home>/reference_files` |
| compound libraries | `MN_LIGAND_LIBRARY_DIR` | `<app-home>/libraries` |
| temporary files | `MN_LIGAND_TMP_DIR` | `<app-home>/tmp` |
| configuration | `MN_LIGAND_CONFIG` | `<app-home>/config/settings.toml` |

The Settings page validates and persists the results and reference roots and
shows directory availability, permissions, free disk space, and expected
reference resources. GPU, Docker, and image diagnostics remain part of Step 9.

### 4.3 Relative artifact references

New run metadata must never require the original host path. Store run-local
paths relative to the run directory:

```json
{
  "kind": "run_artifact",
  "run_id": "20260720-101530-a1b2c3d4",
  "artifact_type": "prepared_complex",
  "path": "artifacts/complex/prepared_complex.pdb"
}
```

Cross-run inputs should use `run_id` plus relative path, or a stable asset ID.
Absolute paths may be resolved at execution time but should not be persisted as
the only source of truth. Keep a legacy resolver for existing runs.

### 4.4 Container mount contract

All adapters should use the same paths:

```text
/work/input       staged inputs, read-only where possible
/work/output      native tool output
/work/artifacts   normalized output contract
/ref              model/reference root, read-only
/libraries        compound library root, read-only
/scratch          temporary workspace
```

The application source tree should not need to be mounted into production
containers. Each image should contain its runner or expose the native CLI.

## 5. Unified job and artifact model

Use a ligand-owned run contract, informed by the generic file-backed job pattern
seen in `mn-protein-design`:

```text
runs/<task-group>/<run-id>/
  input.json
  metadata.json
  command.json
  workflow.json          # parent workflows only
  stdout.log
  stderr.log
  result.json
  artifacts.json
  artifacts/
    normalized/
    raw/<tool>/
    reports/
```

Required job states are `queued`, `preparing`, `running`, `paused`, `completed`,
`failed`, `cancelled`, and `blocked`. A job should include parent/child links,
progress, resource request, selected GPU, timestamps, retry count, and resumable
checkpoint information.

Every tool adapter declares:

- tool ID, version, image tag and immutable digest
- license and model-weight terms
- CPU/GPU and memory requirements
- accepted artifact types
- produced artifact types
- parameter schema and defaults
- command builder
- result parser and scientific validation checks
- smoke-test fixture and expected outputs

The first versioned ligand-owned registry is bundled under
`mn_ligand/manifests/`. Its scope is the image families already referenced by
mn-ligand plus the explicitly selected `openvs:local` candidate; unrelated
protein-design images installed on the same workstation are excluded. Sixteen
workflow/tool roles map onto twelve unique images because shared images expose
separate Vina, GNINA, Uni-Dock Pro, MD, ABFE, and RBFE roles. Each entry carries
an explicit integration status so availability is not confused with validation.
All current app Docker command builders now consume this contract and persist
tool/image/resource command provenance before launch. The presence of a registry
entry alone remains distinct from scientific validation. OpenVS subsequently
received an experimental CPU-only adapter for its Rosetta GALigandDock stage;
the broader iterative ML campaign remains separate.

The first durable-worker foundation is also available through
`mn-ligand worker`. Queued jobs are protected by atomic per-run claims, shared
CPU-slot leases, and per-GPU leases, record the selected device and worker heartbeat, stream native
stdout/stderr to the run folder, and distinguish process failure, normalized
result failure, and cancellation. Dedicated GPU workers accept only GPU jobs,
while a separate CPU worker handles CPU-only tools without blocking a free GPU.
Every CPU and GPU job reserves its declared threads from the same systemwide
pool, capped by the runtime `cpu_process_limit`; multiple worker processes may
safely serve different GPUs without oversubscribing that CPU budget. Pocket Detection is
the first fully worker-owned scientific
page: fpocket and PeSTo submissions return immediately, while the worker owns
container execution, logs, native-output validation, normalized artifacts, and
GPU leases. AutoDock Vina is also worker-owned from Docking submission through
native-pose validation and typed pose/score publication. GNINA and Uni-Dock Pro
now use the same worker lifecycle with explicit per-GPU leases. AlphaFold 3 and
Boltz-2 typed refolding campaigns are also worker-owned; AF3 may execute an
ordered data-pipeline/inference sequence while retaining one lease. Other
page-process execution remains incremental migration work. Automatic worker
service management is available through `mn-ligand worker-service`: the
generated systemd user units are ordered after rootless Docker, use explicit
runtime paths and stable CPU/per-GPU worker IDs, start at login/boot with user
lingering, and restart unexpected failures. Docker commands receive run-local
CID files so cancellation, service interruption, and command failure can remove
the exact container before releasing claims and GPU leases. Workers publish a
durable idle/running/stopping/stopped heartbeat under `.worker/workers`.
Settings combines those records with read-only systemd state, queue depth, and
GPU lease records, so normal frontend users can see worker health without giving
the Streamlit process service-control authority. The same read-only snapshot is
rendered on scientific Run tabs so resource selection and submission share one
current view of GPU availability. The current MD Simulation
workflow is worker-owned
from preparation through production-child finalization and replica analysis.
Endpoint energy can also be launched after any completed typed production run as
an immutable `md-mmgbsa` child job. Each evaluation records its own backend,
percentage window, stride, GPU request, native outputs, normalized portable
artifacts, and source-production lineage.

Unified Jobs and generic Results share a worker-control contract. Cancellation
is offered only when a valid worker command exists; queued jobs transition
immediately and running jobs are terminated by the worker with claims and GPU
leases released. Retry is deliberately immutable: a new run copies only the
adapter's staged inputs, rewrites the old run mount, clears runtime/output state,
and records `retry_of_run_id`, `retry_root_run_id`, and a monotonic
`retry_attempt`. Verified retry profiles cover Pocket Detection, Vina/GNINA/
Uni-Dock Pro, AlphaFold 3, Boltz-2, and post-run MD endpoint energy. Parent-aware
workflow-child replacement remains separate work.

Worker resource admission consumes the registry-backed `cpu_threads`, `ram_gb`,
`scratch_gb`, and `min_vram_gb` declarations. A snapshot records host CPU
capacity, shared CPU slots leased/capacity, available/total RAM, free/total
run-filesystem space, and per-GPU free/total VRAM. CPU slots use atomic per-slot
lease files with heartbeat and dead-owner recovery. Permanently impossible
requests fail before process launch; temporary CPU, RAM, scratch, VRAM,
device-scope, or lease shortages remain queued.
Eligible GPU devices are filtered by free VRAM before the exclusive lease is
acquired. The decision and reason are visible in Unified Jobs and generic
Results.

Core normalized artifact types:

| Artifact | Minimum content |
|---|---|
| `target_structure` | mmCIF/PDB, chains, sequence, residue map, provenance |
| `prepared_receptor` | protonation, retained cofactors/metals/waters, repair report |
| `compound_set` | stable compound IDs, canonical SMILES, parent/source IDs |
| `prepared_ligand_set` | protomers, tautomers, stereoisomers, conformers, charges |
| `pocket` | source target, residues, center/box, method, rank and descriptors |
| `pharmacophore` | typed features, coordinates/distances, source |
| `pose_set` | receptor, compounds, poses, engine scores and ranks |
| `prepared_complex` | receptor, selected pose, ligand chemistry and atom mapping |
| `trajectory` | topology, trajectory, checkpoints and simulation protocol |
| `energy_result` | method, estimate, uncertainty, convergence and replicate data |

### 5.1 Modular job graph

A job performs one bounded scientific operation and publishes immutable typed
artifacts. Any compatible later job can consume those artifacts. The same
prepared target must not be cleaned again merely because it is selected in a
different workflow.

```text
protein import job
  -> imported target
  -> protein cleaning/repair job
  -> prepared target
       +-> pocket prediction job -> pocket set
       +-> receptor refolding job -> receptor ensemble
       +-> docking job + pocket + compound set -> pose set
       +-> screening workflow + pocket + compound library -> ranked pose set
       +-> pocket-conditioned design job + pocket -> generated compound set
       +-> MD system preparation job + ligand pose -> prepared MD system

prepared complex
  +-> MD system preparation -> equilibration -> production MD -> trajectory
  +-> endpoint energy job + trajectory -> MM/GBSA or MM/PBSA result
  +-> ABFE job -> absolute free-energy result

related prepared complexes
  -> RBFE network-planning job -> RBFE edge jobs -> network energy result
```

The UI should expose these relationships as `Inputs`, `Outputs`, `Used by`, and
`Derived from`. Starting a downstream task from a result should preselect the
artifact reference while still creating a distinct new job.

### 5.2 Module boundaries

| Module/job | Consumes | Produces |
|---|---|---|
| protein import | PDB ID, PDB/mmCIF, sequence | imported target |
| protein cleaning/repair | imported target | prepared target, repair report |
| complex preparation | prepared target, ligand/pose | prepared complex |
| structure prediction/refolding | sequence or target, optional ligand | predicted target/complex or receptor ensemble |
| pocket prediction | prepared target or ensemble | pocket set |
| compound import/preparation | SDF/SMILES/table | compound set, prepared ligand set |
| docking | prepared target, pocket, prepared ligand set | pose set |
| screening | receptors, pockets, compound library | ranked pose set, selected compounds |
| generative design | pocket, optional ligand/pharmacophore | generated compound set |
| MD system preparation | prepared complex | parameterized system, preparation fingerprint |
| MD equilibration | prepared system | equilibrated state, strict restart checkpoint |
| MD production | equilibrated state or production checkpoint | trajectory, checkpoints |
| trajectory analysis | trajectory | stability/contact/convergence report |
| endpoint energy | trajectory and topology | MM/GBSA or MM/PBSA result |
| ABFE | prepared complex | absolute free-energy result |
| RBFE planning/execution | related complexes/ligands | network and relative free energies |

Module contracts should support fan-out and fan-in. One prepared target can feed
many pockets or docking campaigns; one screening result can promote selected
poses into separate MD jobs; one energy comparison can aggregate many child
calculations. Deleting an upstream job must warn about downstream references and
must not silently invalidate them.

## 6. Ordered scientific workflows

### 6.1 Prepare an experimental protein-ligand complex

Inputs: PDB ID, uploaded PDB/mmCIF, or an existing complex.

1. Import the biological assembly and preserve source mmCIF metadata.
2. Select chains, ligand, cofactors, metals, structural waters, and alternate
   locations.
3. Run structure validation and repair.
4. Standardize ligand identity from CCD or user-supplied reference SMILES.
5. Assign bond orders, stereochemistry, protonation, and tautomer state.
6. Generate receptor-only, ligand-only, and combined prepared artifacts.
7. Report every removed, changed, unresolved, or inferred component.

Primary tools: the existing PDBFixer/RDKit/OpenMM logic, followed by the local
HiQBind workflow as the reference implementation for higher-quality repair.

### 6.2 Predict or refold a receptor/complex

Inputs: protein sequence or structure, optional ligand/SMILES, optional template.

1. Validate chains and ligand chemistry.
2. Select one or more prediction engines.
3. Run independent child jobs with shared input provenance.
4. Normalize structures, confidence metrics, and ligand poses.
5. Compare predictions and optionally create a receptor ensemble.
6. Pass each accepted model through the same preparation workflow as an
   experimental structure.

Current `mn-ligand` engines:

- **Boltz-2** for protein-small-molecule structure and affinity prediction,
  with ligand-aware sampling controls and typed predicted-complex, confidence,
  affinity, and metrics artifacts.
- **Nesso-1** for affinity-only coarse-grained protein-small-molecule cofolding.
  It consumes typed prepared targets and compound sets, retains the native
  log10(IC50 / µM) prediction, derives pIC50, and publishes uncertainty/confidence
  fields without claiming a predicted complex or docking pose.
- **AlphaFold 3 through AlphaFast** for typed prepared-target plus compound-set
  refolding. The adapter generates native AF3 JSON inputs with one protein
  entity per receptor chain and a ligand `smiles` entity. It reuses the shared
  sequence-hashed target MSA repository and runs the GPU MMseqs pipeline only
  for cache misses. Predicted complexes and confidence summaries are normalized
  into run-relative artifacts. All three engines queue through the local worker from
  Docking / Refolding rather than executing inside Streamlit. Their maximum
  compound setting defaults to `0` for the complete selected dataset; AF3 JSONs
  and Boltz-2 YAMLs are submitted as input directories rather than one container
  invocation per compound.

Other structure predictors may be integrated independently when they support the
ligand application's protein-small-molecule artifact contract. Prioritize
Boltz-2, Protenix, and AlphaFold 3 for complex prediction. Protein-only
predictors can be used to create apo receptor ensembles, but protein-design
workflows and binder candidates are outside this application's scope. AlphaFold 3
must remain an opt-in external image because its code and model parameters have
non-commercial/use restrictions.

### 6.3 Discover and select binding sites

Inputs: prepared target or receptor ensemble.

1. Detect geometric pockets with fpocket.
2. Predict residue-level small-molecule interaction propensity with PeSTo's
   ligand-interface output and cluster high-probability residues into reusable
   ranked pocket assets.
3. Detect complementary machine-learning pockets with the CPU-native P2Rank
   adapter, retaining calibrated profile-specific probabilities and residue scores.
4. For trajectories or ensembles, use mdpocket to find persistent/transient
   pockets.
5. Merge overlapping predictions and known-ligand sites into editable pocket
   assets with residue lists and docking boxes.
6. Allow pharmacophore creation from a ligand, complex interactions, or manual
   feature placement.

The current Pocket Detection page treats fpocket, P2Rank, and PeSTo as
complementary indicators. It shows prior engine runs for the selected target,
groups each engine's parameters behind an independent checkbox, and can queue
all checked engines from one Run action. Results show the selected pocket
residues and docking box in the context of the complete prepared target.

### 6.4 Prepare a compound library

Inputs: SDF, SMILES, CSV/TSV, vendor library, generated molecules, or prior jobs.

1. Parse and assign stable compound IDs.
2. Salt-strip/normalize, validate valence, preserve stereochemistry, deduplicate.
3. Enumerate selected protonation, tautomer, and stereoisomer states.
4. Generate 3D conformers and docking formats.
5. Calculate basic properties and optional PAINS/reactive-group alerts.
6. Store parent-to-enumerated-variant relationships.
7. Shard large libraries without losing global IDs.

Use RDKit, Dimorphite-DL, Meeko, and Open Babel behind one normalized adapter.

### 6.5 Dock, redock, and ensemble dock

Inputs: prepared receptor(s), pocket, and prepared ligand set.

Supported engines should be:

- AutoDock Vina: dependable CPU baseline and redocking benchmark.
- GNINA: CNN rescoring/refinement and GPU-capable docking.
- Uni-Dock Pro: primary high-throughput GPU engine for large libraries, including
  classical, similarity, and hybrid docking.
- Boltz-2 and AlphaFold 3: learned complex cofolding, with native confidence
  evidence clearly separated from classical docking scores.
- Nesso-1: affinity-only coarse-grained cofolding with no claimed pose.

The workflow supports typed redocking with RMSD validation and should add consensus scoring,
multiple pockets, receptor ensembles, per-compound pose retention, and promotion
of selected poses to prepared-complex assets.

The current Docking / Cofolding page executes Uni-Dock Pro, Vina, GNINA,
RosettaLigand, Boltz-2, AlphaFold 3, and Nesso-1 campaigns from reusable target and
compound-set artifacts. One named launch may include multiple targets and a
different checked engine combination for each target while retaining one shared
compound selection and launch-campaign identity. It preserves the
established editable docking box, ligand scrubbing, pH, tautomer, search, pose,
advanced argument, and GPU controls while organizing target, compounds,
engine, running jobs, and results as workflow tabs. Multi-target setup is an
ensemble contract for similar systems: the alignment option and box dimensions
are shared, while every target derives its own automatic center from its own
associated ligand (or the selected pocket). No per-target settings panels are
constructed. An on-page selector switches the single 3D viewer between models.
The target inventory is narrowed with four text filters—Target/PDB, Ligand,
Receptor, and Last step—and an Any (OR) / All (AND) word-matching control.
Campaigns normalize CSV,
SMILES, or SDF records to stable IDs and publish a merged `pose_set`, ranked
`docking_scores`, logs, and artifact lineage. Vina, GNINA, Uni-Dock Pro, and
RosettaLigand exposes one to 100 independent native-seeded runs with a default of one.
Preparation is shared, native result directories remain run-relative and
separate, and multi-run jobs additionally publish mean, sample SD, extrema, and
a median-score representative. Library sharding, receptor
ensembles, and resumable queue execution remain part of the virtual-screening
layer rather than this direct docking job.

The Compounds tab lists only completed Compound Import jobs. It accepts one
imported dataset at a time and defaults to manual selection with the first
parent visibly checked. Users change checkbox rows or explicitly switch to all
unique docking-ready parents. Submission creates a
separate immutable completed `compound-selection` job containing the exact
chosen IDs and typed `compound_set`; all checked engines consume that same
selection. Engine-specific ligand preparation still occurs inside the
scientific child campaigns.

Compound Dataset list rows and detailed Compound Dataset Results provide a
dataset-specific link to Compound Campaign Comparison. This result workspace
collects completed classical docking, RosettaLigand, AF3, Boltz-2, Nesso-1 and
supported rescoring campaigns that resolve to the same immutable source
dataset. Users may compare multiple targets, engines and campaign attempts.
Target, launch-campaign, engine, and engine-run selectors are reconciled as a
cascade, so changing an upstream filter repopulates compatible downstream
defaults instead of retaining stale or empty widget state.
Engine-native bar plots retain their native units and show mean plus sample SD.
All selected engines are visible together as consecutive sections. Every
section has its own metric selector, initialized from the engine's primary
metric when emitted and otherwise from the best available native fallback.
Changing that selector reapplies its scientific ranking direction so the best
compound is always leftmost: lower-is-better energies, RMSDs and concentrations
sort ascending; higher-is-better confidence, probability, pIC50, and GNINA CNN
outputs sort descending.
Repetition handling defaults to the best one attempt per compound and logical
campaign, with controls for best X, one median-representative attempt, or all
attempts. Error bars and tables summarize only the retained attempts.
Late supplemental attempts are analytically merged with the original child job
when their immutable launch campaign, engine, and target agree. Child jobs
remain separately selectable for provenance and structure inspection.
For GNINA, all emitted pose models and their empirical score, CNN pose score,
and CNN affinity are retained. Result viewers expose CNN-pose-score and
empirical/minimizedAffinity selection modes, including historical multi-record
outputs, and campaign metrics distinguish the properties of the two selected
poses instead of silently assuming model 1 is universally best.
Cross-engine prioritization uses within-campaign percentiles only and is
presented separately, including target-by-compound and engine-by-compound
matrices; heterogeneous raw metrics are never averaged together. Campaigns
covering less than 80% of the best-covered selected campaign or fewer than two
compounds remain visible in native views but do not influence consensus.
The Viewer switches between one-compound inspection and a selectable compound
matrix. The single view can overlay a median-score representative per campaign,
best per engine, all repetitions, or manually selected predictions. The matrix
renders up to 12 compound panels with linked py3Dmol cameras, then aggregates
chemically compatible engine-to-engine fixed-frame pose RMSDs across compounds
as mean ± sample SD and renders one square engine-by-engine matrix per compound
in the same grid order with one shared color scale. Structure colors, legends,
and matrix axes always use AlphaFold 3, Boltz-2, GNINA, Uni-Dock Pro, AutoDock
Vina, then RosettaLigand order. It never calculates
atom-mapped RMSD between different compounds. Cofolded complexes are rigidly
aligned to the selected prepared input target through matched protein Cα atoms;
classical poses retain their prepared coordinate frame. Nesso is correctly
omitted because it produces no structure.
The Correlations view offers a focused summary and an all-metrics inspection
mode. The focused Spearman matrix includes docking/energy scores, non-log IC50
in µM, binder probabilities, and AF3 ipTM; lower-is-better metrics are
sign-inverted so positive correlation consistently means favorable agreement.
Values are first aggregated over repetitions and campaigns per compound, one
target is analyzed at a time, and fewer than five overlapping compounds yield
no cell. Rescoring is deliberately separate: each rescoring engine receives a
paired original-score-versus-rescored-output plot and table and cannot add an
independent vote to cross-engine consensus.
Correlation results can switch between the complete matrix and a
compound-level scatterplot. Scatterplot pairs are generated automatically from
the current focused, all-metrics, or custom feature set, ordered by absolute
correlation, and default to the strongest sufficiently overlapping pair. A
static scatterplot-matrix mode renders every combination of two to ten selected
metrics in one compact grid.

Container images are installation-managed through the versioned tool registry
and deployment configuration. Scientific workflow pages do not expose editable
image tags; Settings and diagnostics may display the installed image/version
read-only.

Classical box-based launches offer one shared sizing choice. Fixed sizing is the
default and begins at 20 × 20 × 20 Å. Padding sizing begins at 15 Å on each side
of the selected pocket, bound ligand, or stored source region. Both modes retain
an editable center and record the mode, padding, and effective final dimensions
in typed job provenance. Padding initializes editable X/Y/Z dimensions; manual
edits persist until padding or the source region changes, which intentionally
recalculates all three values. A manually adjusted padded box is recorded as
`padding_manual`. Cofolding-only engines do not show this control.
For classical docking, the Compounds tab can run an on-demand 3D box-fit
preflight over the unique-parent table. A deterministic conformer supplies
principal-axis dimensions and maximum heavy-atom span. The fit check compares
sorted ligand dimensions with the full sorted box dimensions without a
clearance subtraction and labels parents as fitting, likely too large, or not
estimated. Missing estimates run concurrently across bounded worker threads;
raw method-versioned results persist in the runtime cache and are reused across
box sizes and later sessions. Exclusion is never implicit: the default retains
warnings, while an explicit exclusion choice changes the immutable membership
shared by every selected engine. The immutable selection job publishes a
separate typed exclusion report with omitted parent IDs, estimated dimensions,
the exact reason, excess and effective box dimensions; the imported source
dataset is not modified. A selectable warning-only table appears below the
complete parent table
and exposes the exact failed dimensions and Å excess. Selecting one row reuses
the Compound Dataset Results detail presentation for its 2D structure, source
information, and calculated descriptors.
Targets with a sibling prepared ligand or stored `pose_set` use the first
coordinate pose as the unpadded box center and extent source independently of
target orientation; ligand-derived padding defaults to 5 Å per face. A separate
default-off control optionally aligns that ligand's longest principal axis with
global X. Disabled leaves the prepared protein and ligand byte coordinates
unchanged. Enabled creates a completed typed target-orientation job and applies
one centroid-pivoted rigid rotation to the protein, associated ligand, and any
other coordinate reference ligand together. The transformed target, ligands,
box, matrix, and lineage are immutable and shared by every selected engine; no
component is translated or aligned independently.

Vina, GNINA, Uni-Dock Pro, RosettaLigand, Boltz-2, and AlphaFold 3 inherit one
shared default-one, maximum-100 campaign repetition control. For AF3 the shared
value becomes its native model-seed count. Each repetition is a complete
separately seeded attempt under the campaign's single worker/GPU lease.
Boltz-2 diffusion samples remain a distinct within-run ensemble control.
Nesso-1 retains a separate affinity-repetition control and publishes
log10(IC50 / µM), derived pIC50,
and arithmetic/geometric IC50 summaries in µM, but no structure artifact.
Campaign extension and recovery operate on validated logical repeat IDs, not
directory counts. Engine-native completion requires the expected pose,
score/silent, confidence/structure, or affinity artifacts for every staged
compound as applicable. AF3 seed output is mapped back to its logical repeat.
If an attempt is interrupted, only its exact missing ID is queued; already
valid attempts are retained even when the gap is internal. The same contract
governs Docking / Cofolding and Redocking / Refolding launches.
The Results Metrics tab plots AlphaFold 3 structural/interface confidence,
disorder, clash status, and mean ± sample-SD confidence stability across
model-seed attempts. Nesso plots per-run affinity, IC50, binder probability,
model entropy diagnostics, replicate-summary uncertainty, IC50 summaries, and
ensemble disagreement; these remain learned model outputs rather than
experimental measurements.
Static py3Dmol viewers share browser-session camera persistence: changing a
Streamlit control restores the previous rotation, zoom, and translation for the
same page role and target/job instead of returning to the initial camera.
Different structures receive distinct camera-state keys. Trajectory views keep
their equivalent frame-aware persistence.

Benchmark datasets are reusable Prepare assets rather than engine-specific
folders. One canonical case contains stable case/target IDs, reference receptor,
coordinate-bearing ligand, optional complete complex and sequence, split,
SMILES, metadata, and exact source paths. Generic manifests and archives are
the long-term contract; PoseBench Astex Diverse, PoseBusters Benchmark, DockGen,
and CASP15 are importer profiles over that same contract.

The Benchmark Redocking / Refolding page uses
`Dataset → Engines → Run → Results`; engine
groups select redocking, refolding, and rescoring within one familiar campaign
surface and join them on immutable dataset/case/campaign identity. Installed structural
engines are Vina, GNINA, Uni-Dock Pro, RosettaLigand, Boltz-2, and AlphaFold 3.
Nesso remains visible as a structure-free affinity comparator. GNINA score-only
and Boltzina consume exact benchmark poses without moving them. Docking /
Cofolding and Redocking / Refolding share the same engine-control pattern:
parameter accordions remain present, selected engines expand, and deselected
engines collapse. Dataset Results
is the shared aggregate surface: per-case and engine RMSD, top-pose recovery at
1/2 Å, filtered/downloadable tables, campaign status, refolding protein
alignment diagnostics, and score-to-source-pose joins.

Redocking ligand RMSD is symmetry-aware and measured directly in the fixed
reference receptor frame. Refolding first rigidly aligns predicted and reference
protein Cα atoms, then measures the ligand without independent ligand fitting.
The two coordinate-frame definitions remain explicit and are not silently
pooled as if they were the same experimental procedure.
General AlphaFold 3 and Boltz-2 input builders retain protein, DNA, and RNA
chains as typed polymer entities. Benchmark campaigns intentionally apply a
narrower rule: one immutable receptor is derived from the single protein chain
with the strongest reference-ligand contact support; redocking uses only that
receptor and AF3/Boltz-2 receive only that chain's sequence. CASP15 reference
splitting keeps genuine small-molecule reference components in the
coordinate-bearing ligand artifact used for evaluation.

Benchmark Datasets is import-first: the uncollapsed primary workflow is
`Input → Format → Validation → Run → Results`. The Results table contains every
imported collection, its case counts and derived-job coverage. Its linked Job ID
opens a dedicated dataset explorer with canonical cases and a
receptor-plus-reference-ligand viewer. Cases and Viewer are combined: selecting
a case-table row updates the structure and its ligand/receptor property summary
directly below the table. A separate link opens the combined
redocking/refolding page with `Overview → Artifacts → Metrics → Viewer →
Lineage → Logs`.
Complete local PoseBench v1.1.0 registrations contain 85 Astex Diverse,
428 PoseBusters Benchmark, 260 DockGen, and 15 ligand-evaluable CASP15 cases;
the CASP15 ion-only H1135 reference is recorded as rejected rather than being
misrepresented as a ligand benchmark.

Native all-engine smoke evidence is attached to official PoseBusters case
`8AEU_M0L` in benchmark dataset `B9C86`. Every locally compatible adapter
completed, including both coordinate-preserving rescoring engines. The smoke
uses one replicate, three classical docking poses, low exhaustiveness, one
refolding seed, and minimal recycle/sampling controls; it validates contracts,
provenance, artifacts, RMSD aggregation, and score joins, not benchmark-level
scientific accuracy.

### 6.6 Large virtual-screening campaign

RosettaLigand is an optional specialist adapter, not the core scheduler. The
local Docking-page integration uses Rosetta GALigandDock for VSH/VSX and does
not require CSD. OpenVS is the broader framework; its optional CSD analysis and
iterative machine-learning campaign remain separate because they have different
dependencies and execution contracts.

An app-owned campaign should run this resumable graph:

```text
target -> repair -> pocket(s) -> receptor ensemble
library -> standardize -> enumerate -> conformers -> shards
receptors x pockets x shards -> fast docking
fast poses -> GNINA rescore/refine -> consensus rank
ranked hits -> diversity/property filters -> selected candidates
selected candidates -> complex prediction or MD -> MM/GBSA
top related series -> RBFE network
```

Campaign results must retain failed compounds, shard progress, score columns by
engine, diversity clusters, promotion decisions, and full lineage.

### 6.7 Pocket-conditioned and pharmacophore-guided design

Generated molecules must enter the same `compound_set` contract and pass
standardization, validity, novelty, property, docking, and diversity filters.

One Generate campaign owns the shared scientific intent: prepared target,
optional pocket, reference ligand/scaffold/fragment, editable pharmacophore
hypothesis, and optimization objectives. Engine children remain independent
because their conditioning representations and model outputs differ. The
Engine tab must show whether a child consumes the shared hypothesis, converts
it, re-detects ligand features, detects protein-ligand interactions, or uses
only pocket geometry.

The app-owned `pharmacophore_hypothesis` is the editable source of truth.
Engine-private Pharmit JSON, OMTRA XYZ, PGMG `.posp`/`.edgep`, or detected
ProLIF interaction tensors are derived artifacts with provenance, never silent
replacements. A hypothesis may be created from a coordinate ligand, a
protein-ligand interaction analysis, an existing hypothesis, or manual
features. Focused optimization creates another immutable hypothesis so the
original scientific assumption remains auditable.

For PLIP-analyzed complexes, observed ligand-atom features may be retained
directly from native PLIP atom identifiers. A mandatory side-chain contact is
a separate typed design constraint with author and prepared residue numbering,
protein atom, complementary ligand feature, target distance, tolerance, and
whether the reference contact was actually observed. Pocket-only engines carry
that requirement as a downstream pose-validation gate; pharmacophore-aware
engines consume the spatial point and still require the same downstream gate.
When the required side-chain contact is intended to replace an observed
backbone contact to the same residue, the backbone observation stays in source
provenance but is excluded from the active exported feature set by default.
The visual editor must show the complex, bound ligand, feature tolerance
volumes, required target atom, and constraint vector alongside the feature
table. A guided interaction selector maps protein-side hydrogen-bond, ionic,
hydrophobic, aromatic/π, and halogen intent to a complementary ligand feature
and exposes engine-format limitations before launch. Numeric/focused edits and
row additions/removals update the preview; saving produces a new immutable
hypothesis and synchronizes campaign constraints from the visible feature
state.

Implemented generation-engine set:

1. **OMTRA**: broad modern adapter for pocket-conditioned design, docking,
   conformers, and pharmacophore-conditioned generation; Docker support exists.
2. **PocketXMol**: pocket-conditioned generation, docking, conformation, and
   partial-structure manipulation; local source uses an MIT license.
3. **FLOWR.root**: interaction-conditioned, scaffold, fragment, and spatial-
   reference generation with integration and diversity controls.
4. **conDitar**: permissioned pocket-conditioned diffusion using external
   Diff/PocketAE references.
5. **paOPT**: a separate permissioned image for multi-objective ADMET-steered
   conDitar inference. Multiple endpoints and minimize/maximize directions are
   combined by native MGDA; upstream inference has no manual objective weights.
6. **DrugRPG**: pocket-conditioned generation with directed ligand atom count.
7. **PFM**: structure-based generation retaining its learned size prior and
   fixed validated 20-step ODE path.
8. **PocketFlow**: autoregressive pocket generation with atom/bond temperature,
   focus, growth-size, and protein-distance controls.
9. **PGMG**: pharmacophore-only generation directed by the editable hypothesis;
   non-commercial ShareAlike licensing remains explicit.

All engines also support per-engine attempts, batch, seed, and an optional
inference time limit. The UI explains each exposed native parameter's quality,
diversity, molecular-size, and runtime effect and provides empirical
attempt/time planning.

Every completed generator now fans out to an immutable molecule-qualification
child before its output is offered downstream. Canonical stereochemistry-aware
SMILES is used as the cross-engine identity, while native SDF coordinates are
retained unchanged for provenance. Common chemical plausibility and
synthetic-accessibility checks precede deterministic ETKDGv3 conformer
generation, MMFF94s/UFF optimization, and PoseBusters molecule-only validation.
Core chemical and geometry failures are hard exclusions; isolated
non-aromatic-ring flatness is an explicit review warning rather than proof of
an invalid molecule. The accepted 3D SDF is the only generated `compound_set`
handed to docking/cofolding. Its viewer exposes every available standardized 3D
candidate—including warning and rejected records—with status and reasons, and
does not imply pocket placement.

Prepared targets are also the entry point to a combined design-campaign
summary. The Structure Import results table links each target to all engines
and latest qualification revisions generated from that exact structure. Users
can compare per-engine acceptance, inspect QED-versus-MW distributions, filter
by engine/status/QED/MW/SA/cLogP, select table rows, preview each standardized
3D conformer, and save the selection as an immutable typed compound dataset.
The resulting dataset is directly selectable by Docking / Cofolding with its
target, filter scope, properties, warnings, and source jobs preserved.

### 6.8 Molecular dynamics and stability

Inputs: prepared complex or selected docking/prediction pose.

1. Parameterize protein, ligand, cofactors, ions, and solvent.
2. Validate atom mapping and initial geometry.
3. Minimize, heat, equilibrate, and save strict restart checkpoints.
4. Run one or more production replicates.
5. Analyze RMSD/RMSF, ligand RMSD, contacts, hydrogen bonds, distances, secondary
   structure, radius of gyration, pocket volume, and convergence.
6. Cluster poses and export representative complexes.

OpenMM remains the default interactive engine. GROMACS is available in the same
page for established production protocols and long trajectories. Users may run
either engine or launch both from one source complex; the latter creates
separate immutable workflows linked by a comparison-group ID. Do not translate
checkpoints between engines.

#### User-facing MD workflow

Expose one **MD Simulation** task page rather than requiring users to move between
system-preparation and production pages. The default `New MD simulation` action
creates visible modular child jobs:

```text
prepared complex
  -> MD system preparation
  -> equilibration
  -> production replica 1..N
  -> endpoint-energy replica 1..N (optional)
  -> optional trajectory analysis
```

Keep system preparation, equilibration, and production as separate job types.
This preserves checkpoints, provenance, reuse, retries, and independent failure
states even though the normal UI presents one coherent workflow.

The page should offer two modes:

1. **New MD simulation**: prepare a new system, equilibrate it, and launch one or
   more production replicas. This is the default for a new complex.
2. **Reuse prepared system**: launch additional replicas or continue production
   from a compatible prepared/equilibrated system.

Create a new system-preparation job whenever any system-defining input changes:

- protein structure, mutation, ligand pose, ligand identity, protonation,
  tautomer, or stereochemistry;
- force field, ligand charges, cofactors, metals, retained waters, or ions;
- solvent model, box, ionic strength, constraints, or preparation engine;
- conversion between OpenMM and GROMACS.

Reuse a prepared system for additional production replicas, new random seeds or
velocities, longer checkpoint continuation, and compatible trajectory analyses.
Changing temperature, pressure, restraints, or integrator settings may require a
new equilibration child job even when full parameterization can be reused.

Exact continuation restores the serialized OpenMM System and Integrator and then
loads their matching checkpoint without rebuilding or reparameterizing the
system. Independent replicas restore topology from PDB and positions plus box
vectors from the serialized OpenMM State, then generate seeded velocities and
run unrestrained production-like NPT until the replica-specific Roe density
plateau passes. The configured minimum replaces a blind fixed burn-in; the run
extends in increments up to its configured maximum, and its density
revalidation frames are excluded from production analysis. The State is
required because a PDB `CRYST1` record alone is not a reliable restart
representation.

Completed MD campaigns expose a separate **Extend simulation duration** action
in Campaign Results. The requested value is the new cumulative duration for
every completed replica; it must be greater than the workflow's current target.
The extension stays inside the existing workflow and replaces each production
step with a continuation child linked to the previous production run. It does
not create a second scientific campaign or reinterpret a duration extension as
an additional independent replica.

OpenMM continuation requires the existing trajectory and final structure, an
exact endpoint checkpoint, and the same serialized System and Integrator used
to create that checkpoint. Production writes an explicit checkpoint after the
final integration step so newly created runs continue directly from the stored
endpoint. DCD and state-data outputs are appended with their original reporting
interval. A legacy run may continue only when its native checkpoint and append
contract are compatible; reporter/header mismatch fails rather than duplicating
or silently dropping frames.

GROMACS continuation is engine-native and never consumes an OpenMM restart. It
requires the matching `.cpt`, `.tpr`, native `.xtc`, `.edr`, topology, index,
and native production log. The adapter runs `gmx convert-tpr -extend` for the
additional physical time and resumes with `gmx mdrun -cpi ... -append`. GROMACS
therefore verifies its normal output checksums before appending. Missing or
incompatible artifacts are fatal; a `.gro` or PDB coordinate file alone is not
a continuation source.

For either engine, extending production supersedes the old endpoint-energy and
aggregate-analysis children and queues replacements against the extended
replicas. Original children and artifacts remain immutable history. The UI
requires explicit confirmation and never extends completed workflows merely by
opening a result page.

Every prepared system must publish a compatibility fingerprint containing source
artifact IDs and hashes for topology/coordinates, ligand chemistry and atom map,
force fields and charges, solvent/ions/box, constraints, engine, and preparation
settings. Production may reuse it only after strict fingerprint validation. If
compatibility is uncertain, the UI defaults to new preparation and explains why.

The separate **MD Analysis** page consumes existing trajectories and can run many
analysis jobs without repeating system preparation or production.

The current MD Simulation implementation exposes Smoke, Ligand MM/GBSA,
Stability, and Manual production presets in ns while retaining exact steps,
engine integration profile, timestep, hydrogen mass, frame interval, random seed, and
replica density-revalidation window in immutable metadata. Stability is the
default and starts three
independent 100 ns trajectories. Ligand MM/GBSA starts from three independent
50 ns trajectories and enables automatic endpoint jobs; Smoke is technical
only, and Manual exposes expert-controlled values. Task presets lock the two
analysis choices to their intended workflow; Manual unlocks them. No preset
claims convergence.
Endpoint MM/GBSA is a
separate child of each production replica. Automatic endpoint analysis is off by
default; the recommended route is post-run evaluation after inspecting stability
and selecting a justified trajectory window. Its backend/window/frame controls
remain available in a collapsed advanced section for standardized campaigns.
The aggregate child reports per-replica RMSD and energy values together with
mean and sample SD. MD Results browses one aligned frame at a time for responsive
in-app inspection and can export or explicitly launch a PyMOL trajectory script.

New system preparation also offers a Roe–Brooks 2020-inspired sequence for
OpenMM and GROMACS. The GROMACS adapter expresses the same engine-neutral stage
contract as reproducible `.mdp` files and molecule-local position-restraint
includes, uses double-precision minimization and GPU dynamics, and preserves
native topology, coordinate, checkpoint, run-input, index, density, and
trajectory artifacts.
It protects the starting complex with progressively reduced protein/ligand
heavy-atom positional restraints, fully releases the ligand before
production-like NPT, and gates downstream production on a fitted density
plateau. The page exposes minimum/maximum stabilization duration, evaluation
increment, sampling interval, and whether all three published density criteria
are mandatory. Results show the density series, fitted final density, individual
criterion values, and pass/fail status. Ligand escape after full release is a
scientific trajectory result; the ordinary workflow does not add permanent
pose, distance, or receptor–ligand anchor restraints.
Roe–Brooks is the recommended default and presents its fixed sequence as a
summary rather than showing legacy step presets that it ignores. Density-gate
settings are advanced controls. The previous staged protocol remains an explicit
compatibility choice and reveals its legacy presets only when selected.
The engine view provides select/deselect actions, one checkbox per engine, and
one expandable settings panel per engine. Protein and ligand force fields,
water model, solvent box, padding, ionic strength, temperature, and pressure
are selected independently and retained in provenance. Recommended defaults are
Amber ff14SB-family/OpenFF 2.2.0/TIP3P for OpenMM and
ff14SB/GAFF2/TIP3P for GROMACS. Roe's reported solvent setup is TIP3P in a
truncated octahedron with 1.0 nm (10 Å) solute padding.

### 6.9 Binding-energy calculations

Keep separate result families:

- Learned affinity: Boltz-2 and later affinity models.
- Docking scores: Vina, GNINA, and Uni-Dock Pro.
- Endpoint estimates: OpenMM MM/GBSA, AmberTools MMPBSA.py, and GROMACS
  `g_mmpbsa`. Geometric analyses are shared across DCD and PBC-normalized XTC,
  while thermodynamic and endpoint-energy calculations remain engine-native.
  GROMACS 2026.3 remains the simulation engine; GROMACS endpoint jobs generate
  a GROMACS 2025.4 compatibility TPR for the GROMACS 2025 core embedded in
  `g-mmpbsa` 3.0.13. The original XTC and simulation TPR remain unchanged.
  `gmx_qk` is not a runtime dependency.
- Alchemical absolute free energy: OpenFE ABFE.
- Alchemical relative free energy: OpenFE RBFE with an explicit ligand network.

Every energy workflow should expose replicate count, uncertainty, overlap and
convergence diagnostics, failed windows, protocol version, and units.

### 6.10 Ligand properties and quantum chemistry

ADMET-AI can provide fast learned-property triage. Add RDKit descriptors and
rules as transparent baselines. Quantum chemistry should use a real, separately
licensed ORCA mount or an open alternative such as Psi4; the current mock QC
results must never be presented as calculated values.

## 7. Tool categories and implementation priority

| Category | Default/priority tools | Alternatives or later tools |
|---|---|---|
| structure repair | current prep + HiQBind | PDBFixer modules, user-defined repair |
| complex prediction | Boltz-2, optional AlphaFold 3 | Protenix, OMTRA docking |
| pocket detection | fpocket, P2Rank, PeSTo ligand interface | mdpocket |
| ligand preparation | RDKit, Dimorphite-DL, Meeko | Open Babel |
| docking | Vina, GNINA, Uni-Dock Pro | Boltz-2, OMTRA |
| virtual screening | app-native campaign | OpenVS specialist adapter |
| pharmacophores | RDKit/interaction extraction, OMTRA | PGMG |
| generative design | OMTRA, PocketXMol, GenMol | DrugRPG, PFM, PocketFlow, FlowR |
| molecular dynamics | OpenMM | GROMACS |
| endpoint energy | OpenMM MM/GBSA, AmberTools MMPBSA.py | g_mmpbsa |
| alchemical energy | OpenFE ABFE/RBFE | additional validated protocols later |
| properties | ADMET-AI, RDKit | additional calibrated models |
| quantum chemistry | ORCA external mount or Psi4 | xTB for fast pre-optimization |

### 7.1 Post-prediction physical pose validation

PoseBusters is exposed under **Evaluate > Pose Validation**, because it judges
existing structural predictions rather than creating a new pose. The page
accepts completed classical docking, RosettaLigand, Boltz-2, and AlphaFold 3
results. Its coverage table distinguishes not-run, active, completed, and
failed/retryable source jobs. All missing sources are selected by default;
users check or clear jobs directly in that same coverage table, with shortcuts
for all missing, all eligible, or none. A batch submission creates one immutable
CPU child per source. The immutable selection policy validates the best emitted
classical/RosettaLigand pose per compound and repetition, Boltz-2 model 0 and
AlphaFold 3 sample 0 per attempt, and both GNINA's CNN-best and emitted
Vina/minimized-affinity-best poses per compound and repetition. Identical GNINA
selections are deduplicated. There is no separate pose-level selector or
redundant prediction table.
The source table supports status, engine, and text filters. Its stored-prediction
counts are reported as focused selected-pose counts from versioned per-source
inventories in the normal workdir cache;
the first calculation is concurrent and subsequent UI reruns reuse relative
artifact-path records instead of rescanning completed run folders.

The normalized result separates technical completion from scientific validity.
It records applicable pass/fail tests per pose, preserves the complete native
PoseBusters table, and displays the tested receptor-ligand complex focused on the
ligand. The current `dock` profile covers molecular loading and sanitization,
connectivity/radicals, bond and angle plausibility, internal clashes and energy,
ring/double-bond geometry, protein proximity, minimum distances, and volume
overlap. Nesso has no structural output and is therefore not an input.

The adapter preserves native cofolded coordinates while restoring ligand bond
topology from the exact engine input YAML/JSON. It is CPU-only and parallelizes
independent poses with PoseBusters' process pool; no CUDA image, GPU lease,
weights, or shared reference bundle is required.

The dataset-scoped Compound Campaign Comparison has a separate **Pose
validation** tab. It resolves validation children through their exact source-job
IDs and presents a target-specific compound-by-engine PASS/FAIL/not-tested map.
The accompanying engine statistics compare the fraction of assessed compounds
with at least one fully passing pose against the pass fraction across all poses,
while also reporting untested compounds and raw passing/failing pose counts.
The tab lists common failed checks, a sortable compound summary, and links to
the immutable PoseBusters result pages. Legacy outputs without the current
applicable-check manifest or focused selection-policy metadata are not used for
scientific pass decisions; they remain accessible as historical results.

## 8. RTX 5090 Docker policy

RTX 5090 is Blackwell (`sm_120`). GPU images should use CUDA 12.8 or newer and
must not assume that an older CUDA binary will run merely because the host driver
is new. PyTorch images need a CUDA 12.8+ build.

For every GPU image:

- build from a pinned CUDA 12.8+ base digest;
- install PyTorch from a pinned `cu128` or newer channel when PyTorch is used;
- compile CUDA extensions with native `sm_120` plus PTX for forward compatibility;
- expose OCI labels for source commit, build date, license, CUDA, and model version;
- keep model weights outside the image under `/ref` where practical;
- run an image-specific `doctor` and a small scientific smoke test;
- record the image digest and detected GPU in `command.json`/`result.json`;
- retain CPU images for CPU-native tools instead of adding CUDA everywhere.

Required acceptance checks include `nvidia-smi` in the container, PyTorch CUDA
availability and device capability, OpenMM's installation test, and a minimal
engine run that produces parseable non-empty artifacts. A successful import is
not sufficient evidence of Blackwell compatibility.

## 9. Licensing and distribution rules

Each manifest needs separate fields for code license, model-weight license,
database license, redistribution permission, commercial-use status, and required
citation. Images or weights with restricted redistribution should be configured
as user-supplied external resources.

Known examples:

- Boltz code and weights: MIT, including commercial use.
- AlphaFold 3: restricted source and model-parameter terms; opt-in only.
- GNINA: GPL when built with Open Babel, otherwise Apache conditions apply.
- GROMACS: LGPL 2.1.
- OpenFE, fpocket, HiQBind, and PocketXMol: permissive open-source licenses in
  their checked-out repositories.
- PGMG: CC BY-NC-SA; not a default distributable commercial component.
- ORCA: separately licensed and mounted by the user.

Licenses must be checked again and pinned to the selected source commit before an
image is published.

## 10. Source references

Architecture inspiration reviewed in the neighboring application:

- `mn_protein_design/core/jobs.py`, `artifacts.py`, `docker_runner.py`,
  `manifests.py`, `modules.py`, and `pipeline.py`
- `mn_protein_design/runtime.py` and `app/pages/settings.py`
- `mn_protein_design/workflows/design_campaigns.py`
- local source checkouts under `tools_to_implement/`

`mn-protein-design` and `mn-ligand` are separate applications with different
scientific domains, entities, workflows, and release lifecycles. Only generic
engineering patterns may be copied and adapted into `mn-ligand` when useful. Do
not modify the neighboring repository, import it at runtime, or copy its
protein-design workflow semantics into this application.

Primary upstream references:

- NVIDIA Blackwell compatibility guide:
  https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
- Boltz: https://github.com/jwohlwend/boltz
- AlphaFold 3: https://github.com/google-deepmind/alphafold3
- OpenMM: https://docs.openmm.org/
- GROMACS: https://manual.gromacs.org/
- OpenFE: https://github.com/OpenFreeEnergy/openfe
- GNINA: https://github.com/gnina/gnina
- Uni-Dock Pro: https://github.com/NiBoyang/UniDock-Pro
- fpocket: https://github.com/Discngine/fpocket
- HiQBind: https://github.com/THGLab/HiQBind
- OpenVS: https://github.com/gfzhou/OpenVS
- OMTRA: https://github.com/gnina/OMTRA
- PocketXMol: https://github.com/pengxingang/PocketXMol
- GenMol: https://github.com/NVIDIA-BioNeMo/genmol

## Protein-ligand interaction analysis

- One Interaction Analysis page exposes PLIP and PandaMap as distinct engines.
- Both engines can be submitted together and run as independent CPU jobs.
- Configurable within-job threading parallelizes independent poses.
- Eligible sources include immutable prepared target complexes as well as
  focused docking and cofolding poses.
- Prepared target complexes expose each non-polymer ligand as an exact
  residue-identified candidate so a cofactor cannot be silently analyzed in
  place of the intended bound ligand.
- Source-job coverage distinguishes missing, queued, running, completed, and
  retryable analyses.
- Focused pose selection matches Pose Validation, including separate GNINA CNN
  and Vina-score choices when they identify different poses.
- Native reports/diagrams, prepared complexes, normalized contacts, contacted
  residues, and per-pose summaries are retained.
- AF3/Boltz protein residue IDs are sequence-mapped from their native
  one-based numbering to the imported target's author numbering before contact
  detection; a per-pose mapping artifact preserves both coordinate systems.
- Job Results provides interaction fingerprints and a complex viewer.
- One exact PLIP/PandaMap-analyzed complex or pose can initialize an editable
  immutable pharmacophore hypothesis. PLIP retained native atom identifiers
  support atom-resolved observed features and residue-directed side-chain
  constraints. PandaMap and generic cross-engine associations remain labeled
  as pose-level feature-class evidence.
- Compound Campaign Comparison provides a dataset/campaign/target-filtered
  Interactions tab with coverage, recurrent residues, engine statistics, result
  provenance, and PLIP/PandaMap contacted-residue Jaccard agreement.

## Results Explorer

`Results > Results Explorer` provides a scientific inventory above the
individual workflow result pages. It answers which results exist for a dataset,
target, campaign, compound, or engine without requiring opaque run identifiers.

The explorer provides:

- overview counts and engine/status coverage;
- dataset, biological-target, and compound browsing;
- dataset → biological target → prepared-target variant → campaign → engine
  grouping;
- one summarized engine row when a campaign contains separately launched
  repetitions, with access to every underlying immutable job;
- links to the latest prediction, PoseBusters, PLIP, and PandaMap result that
  belongs to the exact source job;
- direct links into Compound Campaign Comparison for dataset-wide quantitative
  analysis;
- completed scientific results by default, with an opt-in troubleshooting view
  for failed, queued, running, or blocked jobs.

Generated receptor filenames are deliberately not treated as unique target
names. A prepared-target section reports its biological origin, such as PDB
4LNW, stable preparation job code, concrete artifact filename, and ordered
preparation history. This distinguishes identically named artifacts created
from different structures or preparation workflows and makes rotations,
trimming, minimization, and other coordinate-changing steps explicit.

The explorer is a navigation and provenance surface, not a new analysis
engine. Native metrics, pose comparison, physical validation, interaction
analysis, and downloads remain on their purpose-specific result pages.
