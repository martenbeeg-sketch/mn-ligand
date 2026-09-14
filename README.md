# mn-ligand

Standalone Streamlit app for ligand-centric computational chemistry workflows.

Developers and new coding sessions should begin with
[`APP_DEVELOPMENT.md`](APP_DEVELOPMENT.md), then read `APP_feature.md` and
`APP_MISSING.md`, and finally this README.

The app runs local Docker containers directly and stores runtime state under
`/mnt/data/RESULTS/mn-ligand-workdir` by default. It does not require the
external platform to be installed.

## Workflows

- Structure Import
- Target Sequence Modification
- Compound Datasets
- Benchmark Datasets
- Pocket Detection
- Docking / Cofolding
- Benchmark Redocking / Refolding / Rescoring
- Virtual Screening
- Bound ligand MD
- Molecular dynamics
- ADMET prediction
- Boltz-2 prediction
- Nesso-1 affinity-only protein-ligand cofolding
- Quantum chemistry
- ABFE
- RBFE

## Campaign and MD extension

`Results > Campaign Results` is the campaign-level review and extension
surface. It preserves the existing campaign and child-job history while adding
only missing repetitions. Raising a campaign target to three, for example,
leaves three-repeat jobs unchanged and adds the missing children to jobs that
currently have one or two repeats.

Repeat completion is output-validated rather than inferred from a nonempty
directory. Vina, GNINA, and Uni-Dock Pro require every expected compound pose;
RosettaLigand requires native score and silent files plus an extracted pose for
every chunk; Boltz-2 requires confidence and predicted-structure files; Nesso-1
requires readable affinity output; and AlphaFold 3 requires a predicted
structure and summary confidence for every expected input at the corresponding
model seed. An interrupted or partial directory is therefore not reported as a
completed repetition. Extension computes the exact missing logical repeat IDs
(or AF3 seeds) and reruns only those holes, including non-contiguous cases such
as completed repeats 1 and 3 with repeat 2 missing. This behavior is shared by
campaigns launched from both Docking / Cofolding and Redocking / Refolding.

Completed MD workflows can also be extended to a longer cumulative production
time under the same workflow ID. This operation is a native checkpoint
continuation, not a coordinate restart:

- OpenMM restores the matching serialized System and Integrator, loads the
  production checkpoint, and appends to the existing DCD and state-data log.
- GROMACS requires the matching CPT, TPR, native XTC, EDR, topology, index, and
  native log. It extends the TPR and resumes with `mdrun -cpi ... -append`.

Both adapters fail closed when the restart artifacts are incomplete or
incompatible; neither silently minimizes a final structure or regenerates
velocities. New OpenMM production jobs explicitly save a checkpoint at the
exact final integration step. After a duration extension, prior endpoint-energy
and aggregate-analysis children are superseded and recomputed from the extended
trajectory. Existing runs are never extended automatically.

Campaign Results also manages saved **Analysis Sets**. An Analysis Set is not a
new physical campaign; it is a named comparison definition over compatible,
immutable target-ligand redocking/refolding campaigns. Compatibility requires a
matching workflow purpose and shared canonical reference ligand. MD workflows
are kept separate. A membership preview shows campaign IDs, biological/prepared
targets, reference ligands, engines, and the reason each campaign is compatible
before saving. Opening an Analysis Set goes directly to its aggregate comparison
scope.

The comparison workspace separates input-pose recovery from inter-engine pose
agreement. Recovery compares every prediction with its own prepared target's
reference ligand. Agreement is calculated only within one prepared receptor
frame and can be displayed per target, pooled across targets as mean ± sample
SD, or as a single-target/individual-pose drill-down. Identical ligands use
symmetry-aware heavy-atom mapping; different molecules are not assigned a
ligand RMSD. GNINA's CNN-ranked and Vina-ranked selections remain distinct.
The linked 3D view mirrors the RMSD scope and provides a local engine visibility
filter. Fixed color scales make matrices comparable between campaigns.

Completed MD workflows can additionally serve as **settings templates** for a
new target. This differs from checkpoint continuation: only protocol settings
are inherited. The new workflow snapshots the selected source complex, performs
fresh system preparation, and creates newly seeded independent replicas. Old
prepared systems, ligand payloads, residue maps, checkpoints, trajectories,
children, comparison groups, and extension history are never reused. If a
template was originally 50 ns and later extended to 100 ns, the new workflow
runs 100 ns from the beginning with a freshly calculated step count.

## Benchmarking

`Prepare > Benchmark Datasets` registers reference complex collections through
one versioned, engine-neutral case contract. ZIP, TAR, TAR.GZ, TGZ, and
server-local directories are accepted. A CSV/JSON/YAML manifest may explicitly
map `case_id`, receptor PDB, coordinate-bearing reference ligand SDF, optional
reference complex, sequence, split, SMILES, and arbitrary metadata. Without a
manifest the importer recognizes standard PoseBench layouts for Astex Diverse,
PoseBusters Benchmark, DockGen, and CASP15, as well as generic per-case
`*_protein.pdb`/`*_ligand.sdf` pairs. Archives are checked for traversal paths;
source files, validation failures, canonical case tables, and every per-case
artifact remain immutable and run-relative.
The importer also accepts a source DOI/URL and citation, version, or license
note and publishes them as a typed provenance artifact.
Import is the primary `Benchmark Datasets` surface, with
`Input → Format → Validation → Run → Results` tabs. The Results table links the
dataset Job ID to a dedicated reference-dataset explorer and separately links
to the combined redocking/refolding report. The combined report uses the
standard `Overview → Artifacts → Metrics → Viewer → Lineage → Logs` result
organization.

The `Benchmark` sidebar group provides one dataset-driven
`Redocking / Refolding` page with
the familiar `Dataset → Engines → Run → Results` flow. Redocking covers Vina,
GNINA, Uni-Dock Pro, and RosettaLigand. Refolding covers Boltz-2 and AlphaFold 3
structure prediction, with Nesso-1 available as an explicitly structure-free
affinity panel. Rescoring covers GNINA score-only and Boltzina when a compatible
Boltz-2 case context exists. Engine-native input files are prepared only at
campaign launch; PoseBench itself is not a runtime dependency. Ordinary
target-based `Redocking / Refolding` remains available under `Discover`.
Benchmark launch derives one immutable receptor containing the single protein
chain with the greatest reference-ligand contact support. Redocking and
RosettaLigand use only that chain, and AlphaFold 3/Boltz-2 receive only its
sequence. The selected chain, cutoff, contact count, minimum distance, source
receptor, and reference ligand are recorded in provenance. Ordinary
target-based refolding still preserves all selected protein, DNA, and RNA
polymer entities; nucleic acids are not misclassified as small-molecule ligands.

Ligand pose RMSD is the primary structural endpoint. Redocking uses a
symmetry-aware heavy-atom comparison in the fixed reference-receptor frame.
Refolding rigidly aligns the predicted protein to matched reference Cα atoms
before the same symmetry-aware ligand comparison; the ligand is never fitted
independently. Dataset Results reports per-case values, median/mean RMSD,
recovery at 1 Å and 2 Å, engine/mode filters, downloads, campaign coverage, and
rescoring values joined to the exact inherited source-pose RMSD.

The native integration smoke uses the official PoseBusters paper archive from
Zenodo. Dataset `B9C86` contains the single explicit case `8AEU_M0L`; all
locally compatible redocking, refolding, affinity-only, and rescoring adapters
completed. This is execution evidence at deliberately small sampling settings,
not a reproduction of the full PoseBusters or PoseBench benchmark.

The complete official PoseBench v1.1.0 collections from Zenodo record
`19138652` are registered locally as immutable datasets: Astex Diverse
(85 cases), PoseBusters Benchmark (428), DockGen (260), and CASP15
(15 ligand-evaluable reference complexes plus one explicitly rejected
protein/ion-only reference). Their source archives and extracted read-only
inputs live below
`/mnt/data/RESULTS/mn-ligand-workdir/benchmark_sources/`; canonical campaign
inputs remain copied into run-relative artifacts.

## Generate

`Generate > De Novo Molecule Design` is the shared campaign surface for OMTRA,
PocketXMol, FLOWR.root, conDitar, conDitar + paOPT, DrugRPG, PFM, PocketFlow,
and PGMG.
Users define one prepared target and combine a pocket, reference
ligand/scaffold/fragment, editable pharmacophore hypothesis, and transparent
optimization objectives. The Engine tab records how each model maps that common
intent into its native conditioning route.

Existing-ligand work is split into task-specific pages rather than mixed into
de novo generation. `Ligand Redesign` replaces an explicitly selected local
atom region. `Fragment Growing` extracts the selected connected atoms as an
immutable coordinate-bearing fragment and grows from it with PocketXMol or
FLOWR.root. `Scaffold Hopping` previews the RDKit Bemis–Murcko core and invokes
FLOWR.root's native scaffold-hopping transform. `Ligand Optimization` uses
PocketXMol full-molecule optimization and exposes the departure strength from
the bound reference state. Every task defaults to the ligand and bound-ligand
pocket belonging to the selected prepared complex.

`Prepare > Pharmacophore Hypotheses` creates immutable hypotheses from a
coordinate ligand, an exact PLIP/PandaMap-analyzed complex or pose, an existing
hypothesis, or manual features. For PLIP sources, a user may add an
author-numbered mandatory side-chain contact: observed ligand-atom features
remain distinct from the newly proposed design constraint. Deterministic
exchange artifacts cover Pharmit JSON, OMTRA XYZ, and compatible PGMG `.posp`;
users do not need to author those private formats.

The feature table and persistent 3D molecular viewer form one editor. Tolerance
volumes, observed points, bound ligand, required target atom, and constraint
line are shown together. Users can focus a row, change its type, XYZ, radius,
enabled/required state, add or remove rows, and save a new immutable revision.
For target contacts, a guided interaction selector maps plain-language
protein-side chemistry to the complementary ligand feature and explains
typical residues, geometry, and engine limitations.

Generator output is qualified before downstream use. A completed generation
job automatically creates an immutable CPU child that starts from canonical
stereochemistry-aware SMILES, applies common chemical plausibility and
synthetic-accessibility gates, creates a deterministic ETKDGv3 conformer,
optimizes it with MMFF94s or UFF, and runs PoseBusters in molecule-only mode.
Core chemistry, bond/angle, clash, and energy failures remain hard exclusions.
An isolated non-aromatic-ring flatness result is retained as a review warning,
not treated as proof of an invalid molecule. The Viewer keeps every available
standardized 3D candidate visible with its status and reasons; only accepted
records become the typed compound set used by docking/cofolding. Native engine
SDF files remain unchanged for diagnostics, and no standardized conformer
claims a valid pocket placement.

The prepared-target table in `Structure Import > Results` is the entry point
for combined design review. Its **Design results** link opens all campaigns and
latest qualification revisions associated with that exact target. The summary
includes per-engine qualification counts, QED-versus-MW visualization,
filterable property/status tables, and standardized 3D previews. Selected
accepted rows can be saved as an immutable, stereochemistry-aware 3D compound
dataset and opened directly in `Docking / Cofolding`.

Generation checkpoints belong below the configured reference root:

```text
generation/omtra/
generation/pocketxmol/
generation/flowr_root/
generation/conditar/
generation/drugrpg/
generation/pfm/
generation/pocketflow/
generation/pgmg/
```

paOPT uses the permissioned `generation/conditar/Diff.pt` and `PocketAE.pt`
references rather than duplicating them. Separate CUDA 12.8/
Blackwell-compatible images and experimental queue, native-command,
normalization, Results display, and typed compound-set handoff are present for
all nine engines. The Engine tab exposes documented native quality, diversity,
size, and steering controls where supported. The Run tab adds per-engine
attempts, batch, seed, runtime limits, empirical runtime estimates, and
time-to-attempt plus expected-unique-output suggestions.

The structure viewer is vendored in
`mn_ligand/app/components/molstar_viewer`, including the built Mol* frontend,
so the app does not depend on an external viewer package.

The Docker runner mounts this repository into the workflow containers and uses
the vendored Ligand-X-derived modules under `mn_ligand/ligandx`.

## Install With Conda

```bash
git clone <repo-url>
cd mn-ligand

conda env create -f environment.yml
conda activate mn-ligand
pip install -e .
```

If the environment already exists:

```bash
conda activate mn-ligand
conda env update -f environment.yml --prune
pip install -e .
```

## Build Containers

If the Docker images are not already available locally:

```bash
docker compose build
```

Build one image:

```bash
docker compose build docking
docker compose build fpocket
docker compose build p2rank
docker compose build pesto
```

Local image tags:

- `ovolig-structure:latest`
- `ovolig-docking:latest`
- `ovolig-fpocket:latest`
- `ovolig-p2rank:latest`
- `mnprot-pesto-cu128:latest`
- `ovolig-md-cu128:latest`
- `ovolig-admet:latest`
- `ovoex-boltz2:latest` for Boltz2 by default
- `ovolig-qc:latest`
- `ovolig-abfe-cu128:latest`
- `ovolig-rbfe-cu128:latest`

These tags are installation-managed. Scientific workflow pages do not ask users
to edit Docker image names; every submitted job still records the selected
registry image and command provenance. Use Settings/diagnostics to inspect the
installed runtime images.

PeSTo keeps its model checkpoint outside the image. Place `model_ckpt.pt` at
`pesto/i_v4_1/model_ckpt.pt` below the reference directory configured on the
Settings page. The default shared location is
`/mnt/db/reference_files/pesto/i_v4_1/model_ckpt.pt`.

## Run The App

```bash
mn-ligand app
```

The command creates runtime folders automatically. `mn-ligand init` is optional
and only pre-creates those folders.

Default runtime paths:

```text
app home: /mnt/data/RESULTS/mn-ligand-workdir
runs:     /mnt/data/RESULTS/mn-ligand-workdir/workdir/runs
tmp:      /mnt/data/RESULTS/mn-ligand-workdir/tmp
```

The checkout keeps `./mn-ligand-workdir` as a compatibility symlink so
historical metadata containing the former absolute path continues to resolve.

You can override the runtime location:

```bash
mn-ligand app --app-home /path/to/mn-ligand-runtime
```

You can pass Streamlit options through:

```bash
mn-ligand app --server.address 127.0.0.1 --server.port 8501
```

## Installation Diagnostics

The bundled tool registry records 16 app workflow/tool roles across the Docker
image families already used by mn-ligand plus `openvs:local`. It records typed
inputs and outputs, resource requests, reference files, license status, health
checks, and scientific integration status. Other locally installed
protein-design images are intentionally outside this ligand-app registry.
Inspect the local installation with:

```bash
mn-ligand doctor
mn-ligand doctor --skip-images
mn-ligand doctor --json
```

The same checks are available from `System > Settings`. Missing image digests
are reported as warnings until a reproducible digest is recorded. A registry
entry indicates configuration readiness only; it does not replace native-output
and downstream-handoff validation for a scientific adapter.

## Local Worker

Run queued jobs independently from Streamlit with:

```bash
mn-ligand worker --gpu-ids 0,1
```

The worker uses atomic run claims, a shared CPU-slot pool, and one exclusive
lease file per GPU. Every job, including a GPU job, reserves its declared CPU
threads from the same pool; sparse legacy requests reserve at least one slot.
The pool capacity is the smaller of the host-visible CPU count and the runtime
`cpu_process_limit` (16 by default). CPU and GPU leases are heartbeated together,
recorded in job metadata, and released after success, failure, cancellation, or
dead-worker stale-lease recovery. To prevent CPU-only work from occupying GPU
worker slots, run one CPU worker plus one worker per GPU:

```bash
mn-ligand worker --job-class cpu --worker-id mn-ligand-cpu-0
mn-ligand worker --job-class gpu --gpu-ids 0 --worker-id mn-ligand-gpu-0
mn-ligand worker --job-class gpu --gpu-ids 1 --worker-id mn-ligand-gpu-1
```

For persistent workstation operation, install the bundled systemd user-service
units. They are ordered after the rootless Docker user service and start one
CPU-only worker plus one worker per GPU at login/boot when user lingering is
enabled:

```bash
mn-ligand worker-service install --gpu-ids 0,1
mn-ligand worker-service status --gpu-ids 0,1
```

The units record the current Python interpreter, checkout, app home, and temp
directory explicitly. Manage them without replacing the units:

```bash
mn-ligand worker-service restart --gpu-ids 0,1
mn-ligand worker-service stop --gpu-ids 0,1
mn-ligand worker-service start --gpu-ids 0,1
journalctl --user -u 'mn-ligand-worker@*.service' \
  -u mn-ligand-cpu-worker.service -f
```

`worker-service uninstall` stops/disables the selected instances and removes
the template. Worker-owned Docker runs receive a run-local CID file;
cancellation, service interruption, or command failure force-removes that exact
container so an attached Docker client cannot leave an orphan behind.

`System > Settings` provides the normal frontend view of this infrastructure.
It displays the CPU and GPU services, stable worker identity and heartbeat age,
idle/running state, current run, queued-job count, shared CPU slots leased versus
capacity, and active GPU leases.
Workflow pages distinguish a lease-free physical GPU from an occupied worker
slot. This panel is deliberately read-only: submitting and monitoring work
happens in Streamlit, while service installation and lifecycle administration
remain explicit operating-system actions.

For diagnostics or service probes, process at most one job and exit:

```bash
mn-ligand worker --once --gpu-ids 1
```

Pocket Detection submits fpocket, CPU-only P2Rank, and PeSTo jobs without
running containers in Streamlit, and AutoDock Vina campaigns use the same
worker-owned lifecycle. P2Rank offers separate experimental-X-ray and
predicted/NMR/cryo-EM profiles and records its profile-specific calibrated
probabilities without requesting a GPU. A
worker must be running to execute them. GNINA and Uni-Dock Pro campaigns are
 also worker-owned and acquire explicit GPU leases. Typed AlphaFold 3, Boltz-2,
and Nesso-1 campaigns follow the same lifecycle. Nesso publishes learned affinity
results and a normalized campaign table, not a predicted complex or pose.
Scientific launch pages place runtime controls and submission actions in a
dedicated Run tab. It reads the same worker-health snapshot as Settings and
shows CPU threads, queue depth, active leases, and whether GPU 0 and GPU 1 are
Free, Busy, Offline, or reporting a stale heartbeat.
For a selected target, Pocket Detection also lists all previous engine runs and
their settings. fpocket, P2Rank, and PeSTo may be checked and queued together
from grouped engine panels. Completed pocket results show the chosen pocket
residues and docking box against the full prepared target structure.
One shared Docking / Cofolding control supplies one to 100 independent runs,
defaulting to one, to Vina, GNINA, Uni-Dock Pro, RosettaLigand, Boltz-2, and
AlphaFold 3. For AF3 the value is its native model-seed count. Nesso-1 retains
a separate affinity-repetition control. Inputs are prepared once, every run
uses a recorded consecutive native seed, native outputs remain separated by
run, and campaigns with multiple runs publish per-run scores plus mean, sample
SD, and a representative result. Boltz-2 full repetitions are distinct from
its within-run diffusion samples. Nesso summaries retain native
log10(IC50 / µM) and additionally report arithmetic/geometric IC50 and sample SD
in µM. These repetitions are distinct from redocking validation.
Docking / Cofolding presents the four classical engines together with Boltz-2,
AlphaFold 3, and Nesso-1 as checkboxes with grouped parameters. One named launch
can queue different checked engine combinations for multiple prepared targets
against the same compound list. These targets are treated as one similar
ensemble: alignment and box dimensions are shared, while each member derives
its own automatic center from its associated ligand or selected pocket. No
per-target parameter expanders are rendered. An on-page selector switches one
3D viewer between selected models. Target discovery uses only Target/PDB,
Ligand, Receptor, and Last-step text filters, with selectable OR/AND matching
for multiple words. AlphaFold 3
results show native ranking score, ipTM, pTM, disorder, and clash confidence;
these are structural/interface confidence measures, not binding affinities.
Its Metrics tab separates structural/interface confidence, disorder, and clash
diagnostics and summarizes confidence stability as mean ± sample SD across
model-seed attempts. Nesso Metrics plots per-run affinity, IC50, binder
probability, and model entropy features, plus replicate-level affinity and
binder-probability uncertainty, IC50 summaries, and ensemble disagreement.
Result plots and complex overlays use sample 0 from each AlphaFold 3 model-seed
attempt and model 0 from each independently seeded Boltz-2 attempt. Remaining
models and samples stay available as immutable native artifacts. When the
prepared receptor is protein-only, the viewer resolves its sibling prepared
ligand artifact so the input/reference ligand can still be overlaid.
Each row in Compound Datasets and each detailed Compound Dataset Results page
links to a dataset-scoped Compound Campaign Comparison. It joins completed
Vina, GNINA, Uni-Dock Pro, RosettaLigand, AlphaFold 3, Boltz-2, Nesso-1, GNINA
rescoring, and Boltzina rescoring results through immutable source-dataset
lineage. Targets, engines, and individual campaigns are multi-selectable.
These selectors form a cascade: changing a target refreshes the compatible
launch campaigns, engines, and engine runs, while an intentional empty
selection remains empty until its upstream context changes.
Native metrics remain in separate engine-specific plots with mean ± sample SD.
The Native metrics tab renders every selected engine sequentially rather than
hiding all but one engine. Each engine has an independent emitted-score
selector defaulting to its primary scientific output; summarized numeric rows
remain available in a collapsed section below that engine's bar plot.
Every native-metric plot orders compounds from most to least favorable at the
selected metric: energies, RMSD, IC50, log10(IC50), disorder, and disagreement
are ascending, while pIC50, confidence, probability, and GNINA CNN affinity or
pose scores are descending.
Native metrics default to the single best repetition per compound and logical
campaign. Users can increase the retained best-X count, select one
median-representative repetition, or restore all repetitions; mean and sample
SD are calculated only from the attempts retained by that mode.
Supplemental attempt jobs that share the same recorded launch campaign, engine,
target, and dataset are combined into one analytical campaign. They contribute
to one bar and one mean/SD rather than appearing as extra campaigns or adding
extra consensus weight; their individual job pages remain accessible.
The comparison page also has a target-scoped Pose validation tab. It joins
completed PoseBusters jobs to their exact immutable docking/cofolding parents
and shows a compound-by-engine PASS/FAIL/not-tested map, per-engine compound
and pose pass rates, common failed checks, compound coverage flags, and links
back to every validation result. A compound is counted as having “made it” only
when at least one stored pose passes every applicable check; untested cells are
never treated as passes. Newer reruns supersede older results pose by pose, and
legacy results without an applicable-check manifest are excluded from the
scientific summary.
GNINA outputs are parsed pose by pose rather than treating the first model as
the only result. The single-job and campaign viewers can switch between the
highest-CNN-pose-score model (GNINA's normal output ranking) and the model with
the most favorable emitted empirical/minimizedAffinity score. Correlation and
scatterplot inputs label metrics from the CNN-ranked and empirical-ranked poses
separately; CNN affinity is reported as a property of the selected pose and is
not described as the pose-selection criterion.
A distinct consensus view converts only each engine's primary metric to a
within-campaign percentile before summarizing it; raw kcal/mol, REU, confidence,
and learned-affinity values are never pooled. Consensus excludes tiny subset
campaigns that cover less than 80% of the best-covered selected campaign or
fewer than two compounds. Target-by-compound and engine-by-compound matrices
expose selectivity and model disagreement. The Viewer switches between a
single-compound overlay and a selectable compound matrix. The single view
supports a median-score representative per campaign, best per engine, all
repetitions, or manually selected predictions. The matrix uses one linked
py3Dmol camera across up to 12 compound panels and summarizes engine-to-engine
fixed-frame pose RMSD across compounds as mean ± sample SD plus one square
engine-by-engine RMSD matrix per compound, arranged like the linked 3D grid and
using one shared color scale. Viewer colors, legends, and RMSD axes retain the
canonical order AlphaFold 3, Boltz-2, GNINA, Uni-Dock Pro, AutoDock Vina, then
RosettaLigand. Different compounds are never atom-mapped directly.
Cofolded complexes are aligned through protein Cα atoms while classical poses
retain their prepared-target frame; Nesso remains non-structural.
The Correlations tab defaults to a focused, direction-normalized Spearman
matrix of docking/energy scores, predicted IC50 in µM, binder probabilities,
and AF3 ipTM. Repetitions are averaged before compounds are correlated, one
target is analyzed at a time, and cells require at least five overlapping
compounds. `All available metrics` adds confidence and diagnostic outputs;
Pearson is optional but is not the default for skewed concentration values.
The same selected metrics can switch from the matrix to a compound-level
scatterplot. Every valid metric pair is generated automatically, ordered by
absolute correlation, and selectable from one paired control; the strongest
observed pair is shown first. A third static Scatterplot Matrix view displays
all row/column combinations for two to ten selected metrics simultaneously.
Rescoring has a separate paired tab per rescoring engine, comparing its output
only with the recorded original docking score for the same poses. Rescoring
does not contribute an additional consensus vote.
All py3Dmol preparation and result viewers retain their camera rotation, zoom,
and translation across Streamlit parameter reruns within the browser session.
Camera state is scoped to the page role and target/job, so selecting a genuinely
different structure starts from its own view.
For box-based classical docking, Fixed box is the default at 20 × 20 × 20 Å.
Padding mode instead defaults to 15 Å on each side of the selected pocket,
bound ligand, or stored region. The same choice is available in Redocking
Benchmark and is retained in job provenance. Its calculated X/Y/Z dimensions
remain editable; manual values persist until padding or the source region
changes, at which point all three dimensions are recalculated.
The Docking / Cofolding compound-selection tab also provides an optional
docking-box fit preflight for classical engines. It builds one deterministic
3D conformer per unique parent, reports principal-axis length/width/thickness
and maximum heavy-atom span, and compares those dimensions with the current box
using its full X/Y/Z dimensions without subtracting clearance. Missing
measurements are calculated concurrently; raw deterministic estimates are
stored in the runtime cache and reused for other box sizes and later sessions.
Flagged compounds are kept by default; users may explicitly exclude them before
the immutable shared selection is created. An explicit exclusion produces a
separate typed report containing every omitted parent ID, its estimated
dimensions, the exact failed box dimensions and the box used; the source
dataset remains unchanged. A focused selectable warning table
below the complete parent table reports the exact failed dimension and excess;
selecting a warning shows the compound structure, source fields, and calculated
properties used by Compound Dataset Results. Principal-axis alignment is used
only for measurement because docking engines rotate and translate ligands
during search; input coordinates are not reoriented.
When a receptor has a sibling prepared ligand or stored pose set, that
coordinate ligand defines the search-region center and padding source whether
or not orientation is enabled; 5 Å per face is the ligand-derived default. A
separate default-off Target-tab toggle can align the ligand's longest principal
axis with global X. Off preserves the prepared protein and ligand coordinates
exactly. On creates one immutable typed target-orientation job and applies the
same centroid-pivoted rigid rotation to the protein and every coordinate
ligand. The rotated structures, box, transform matrix, and source lineage remain
in one shared coordinate frame for all selected engines.
The legacy single-complex Redocking Benchmark route remains hidden for retained
links. New benchmark work starts from `Prepare > Benchmark Datasets` and fans
the selected immutable case subset across the compatible installed engines.
The current MD Simulation page also queues preparation and production children
through the worker. Users select OpenMM, GROMACS, or both, plus named protocols
expressed in ns, independent replica count, stored-frame interval, and replica
density-revalidation window. A paired launch creates separate engine-native
workflows linked by a
comparison-group ID; checkpoints are never converted between engines. When
endpoint analysis is enabled, each completed trajectory automatically launches
its own immutable `md-mmgbsa` child with a recorded frame window, stride,
engine-compatible backend, and GPU contract; source runs are mounted read-only.
OpenMM supports its existing endpoint paths and GROMACS uses `g_mmpbsa`.
Replicate results report per-run values plus mean and sample SD.
The browser trajectory view extracts one requested DCD or XTC frame on demand,
while a
downloadable PyMOL script and explicit local-PyMOL button handle full-trajectory
inspection. The MD Simulation Results tab also links historical production runs.
Other hidden legacy pages retain synchronous compatibility dispatch during
migration.

The default production preset is **Stability**: three independently seeded
100 ns trajectories with separate Roe-style NPT density revalidation.
**Ligand MM/GBSA** starts from
three 50 ns trajectories and enables immutable endpoint jobs, **Smoke** is a
0.2 ns technical test, and **Manual** exposes expert values. None of these
presets alone establishes convergence or affinity. Task presets lock their
analysis checkboxes to the intended workflow; Manual unlocks both overrides.
Replica stability aggregation is enabled by default. Automatic MM/GBSA is
disabled in the default Stability preset because the preferred workflow is to
inspect density, RMSD, and ligand stability first, then launch one or more
immutable post-run endpoint evaluations with a justified stable trajectory
window. It is enabled by the Ligand MM/GBSA preset for standardized campaigns.

New system preparation defaults to a
[Roe–Brooks 2020](https://doi.org/10.1063/5.0013849)-inspired sequence for
OpenMM and GROMACS.
The Tool / Engine tab provides select/deselect actions, engine checkboxes, and
separate expandable OpenMM and GROMACS preparation panels. Protein and ligand
force fields, water, box, padding, salt, temperature, and pressure are
independent and recorded in the compatibility contract. Recommended defaults
are Amber ff14SB-family/OpenFF 2.2.0/TIP3P for OpenMM and the smoke-qualified
ff14SB/GAFF2/TIP3P combination for GROMACS. Roe's reported solvent setup is
TIP3P in a truncated octahedron with 1.0 nm (10 Å) solute padding.
Ligand heavy atoms are harmonically restrained while solvent and the starting
complex relax, then progressively released together with the protein restraints.
The final NPT density-stabilization phase and all ordinary production replicas
are fully unrestrained. The workflow does not permanently lock a ligand in the
pocket; post-release escape is retained as a scientific result. Density is
sampled and fit to the published exponential model, with all three plateau
criteria stored in typed JSON/CSV artifacts and shown on the result page.
Each newly seeded independent replica repeats the unrestrained density check:
the configured NPT duration is a minimum, the run extends only as needed up to
its maximum, and analyzed production starts after the plateau passes.
Both engines default to an accelerated 4 fs integration profile with hydrogen
masses repartitioned to 4 amu. A standard 2 fs/normal-mass profile is also
selectable. The chosen mass model is applied during system construction and is
kept unchanged through every Roe stage, density gate, replica revalidation, and
production trajectory.

The OpenMM path is an adaptation rather than an exact engine-level reproduction:
it retains its standard H-bond constraints and hydrogen-mass repartitioning
during staged minimization. The GROMACS adapter implements the same
engine-neutral stages with reproducible `.mdp` files, molecule-local position
restraints, double-precision minimization, GPU dynamics, and native
`.top`/`.gro`/`.cpt`/`.tpr`/`.ndx` artifacts. Shared geometric analysis consumes
DCD or PBC-normalized XTC, while thermodynamics, checkpoints, and endpoint
energies remain engine-native. The prior staged equilibration remains available
as an OpenMM compatibility choice.

The `containers/gromacs-md` definition targets GROMACS 2026.3, CUDA 12.8,
AmberTools/GAFF2, MDTraj, MDAnalysis, and pip-distributed `g-mmpbsa`. Its focused
contracts are tested. The image also includes an isolated CPU GROMACS 2025.4
utility to create a TPR readable by the GROMACS 2025 core embedded in
`g-mmpbsa`; simulation trajectories and checkpoints remain GROMACS 2026.3
artifacts. CUDA water and shortened 4LNW preparation/production/MM-PBSA smokes
pass on RTX 4090. RTX 5090 execution, the full density gate, meaningful
production sampling, and scientific energy regression remain qualification
tasks.

Unified Jobs and generic Job Results expose confirmed cancellation for queued or
running worker jobs. Queued jobs become cancelled immediately; running Docker
jobs receive a durable cancellation request and the worker owns termination and
lease cleanup. Failed or cancelled standalone Pocket Detection, classical
docking, AlphaFold 3, Boltz-2, and post-run MD endpoint jobs can be retried as a
new immutable run. Retry copies only verified staged inputs, rewrites run-local
mounts, starts with an empty artifact manifest, and records root/previous-run
lineage and an incrementing attempt number. Workflow-managed children remain
ineligible until parent-aware replacement is implemented.

Before claiming execution, the worker now checks and atomically reserves declared
CPU threads, then checks available RAM, free scratch space on the run filesystem,
and per-device free/total VRAM. Requests larger than the shared CPU-pool or
physical capacity fail with an actionable admission error; temporary CPU-slot,
RAM, scratch, GPU, or VRAM shortages remain queued and expose their reason in
Jobs and Results. GPU jobs hold their CPU reservation while acquiring an eligible
exclusive GPU lease, and release the CPU reservation immediately if no GPU is
available. Sparse legacy resource records remain compatible and reserve one CPU
slot when they do not declare a positive value.

Scheduler-code changes take effect when the worker services are restarted.
Do not restart workers merely to activate a new scheduler while scientific jobs
are running; existing processes safely finish with the version they loaded.

All Docker command construction used by current mn-ligand adapters is routed
through the registry-backed shared runner. New runs record the tool role, image,
resource request, selected GPU, and exact argument list in `command.json` before
launch. RosettaLigand now has a CPU-only local Docker adapter on the Docking page. It
prepares MMFF94 MOL2 and Rosetta generic-potential params, supports VSH
high-precision, VSX express, and exhaustive convergence protocols, preserves
silent/score/native files,
normalizes scores explicitly as relative Rosetta energy units, extracts typed
docked complexes, and hands the best complex to the normal MD preparation path.
All RosettaLigand protocols support repeated explicit seeds; multi-run summaries add
running statistics and symmetry-aware pose clustering in the fixed receptor
frame.
This adapter covers Rosetta GALigandDock and is therefore named RosettaLigand
throughout the app. OpenVS is the broader framework; this integration does not
claim its iterative machine-learning campaign loop or optional CSD analysis.

## Development Tests

Install the optional test dependency and run the regression suite:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

The normal pytest suite is CPU/local and must not require Docker, network
access, checkpoints, or a GPU. Streamlit AppTest and Docker/GPU smoke-test
instructions are documented in `APP_DEVELOPMENT.md`. The current verified full
suite baseline is **340 passed**.

## Modular Protein Preparation

`Structure Import` is the visible one-stop workspace for importing,
inspecting, and registering reusable protein-ligand inputs. Completed Boltz-2
and AlphaFold 3 predicted complexes have dedicated promotion tabs that create
new typed prepared-complex jobs while preserving the native prediction runs.
Manual imports may include a complete complex PDB in addition to separate
protein and ligand files. Uploaded structures and promoted AF3/Boltz complexes
now pass through the same strict HiQBind-compatible Ligand-X/PDBFixer/OpenMM
cleaning job before they are published as prepared complexes. Residue-specific
canonical replacements and eligible internal gaps up to 15 residues are modeled
as MODELLER ensembles; missing sequences and their observed flanks are confirmed
in the UI and recorded in provenance. Missing termini and longer gaps remain
unresolved. PDBFixer then repairs missing atoms, adds hydrogens at pH 7.4, and
OpenMM validates and locally minimizes rebuilt content.
The ligand is parameterized with SMIRNOFF and retained as a fixed interaction
context during minimization. The experimental heavy-atom scaffold is frozen
outside rebuilt-neighbor residues. Every decision and energy is recorded in a
machine-readable repair report with import/cleaning lineage.

Gemmi handles native macromolecular mmCIF-to-PDB conversion before repair,
preserves sequence records for gap detection, and optionally constructs a
declared biological assembly with PDB-safe chain IDs. Non-water cofactors and
metals can be retained explicitly. RDKit plus a CCD reference remains
responsible for ligand graph chemistry, bond orders, canonical SMILES, SDF
output, and sanitization. Open Babel is limited to narrow ligand-format
interoperability paths such as PDBQT/MOL2 conversion. The installation-managed
`ovolig-md-cu128:latest` image contains the complete Gemmi/PDBFixer/OpenMM/
OpenFF/RDKit repair runtime and runs on CPU by default or the RTX 5090 when
selected.

Structure Import has an unfiltered Results tab listing every prepared import.
It provides an inline full-complex viewer, typed artifact table, source/result
metadata, repair report, and a link to the detailed structure viewer. Detailed
results resolve typed artifacts as well as historical filename conventions.

The Boltz-2 and AlphaFold 3 tabs also provide a dedicated preparation workflow
for one FASTA/raw protein sequence and one `LIGAND_ID,SMILES`. Their Input,
Engine, Run, and Completed-predictions stages are separate from Discover:
Docking / Cofolding uses existing imported targets and compound collections,
whereas these preparation tabs create a new complex for registration. Their
completed lists contain only predictions launched from Structure Import and
show the selected native complex inline before promotion. Raw sequence inputs
accept only the 20 canonical one-letter residue codes: noncanonical chemistry
is never silently guessed, and structural repair occurs after prediction during
promotion.

`Target Sequence Modification` combines trimming and C-terminal repair as two
tabs on one Prepare page. Trimming accepts registered imported complexes from
PDB, cofolding, or manual import, applies chain-specific N/C-terminal residue
bounds to protein atoms only, retains ligand coordinates and typed ligand
artifacts, and publishes a new prepared complex/receptor. C-terminal repair
adds a short canonical sequence with MODELLER 10.8 from the isolated
`mn-ligand-modeller` environment, preserves the immutable source complex and
ligand coordinates, generates and ranks an ensemble, validates peptide-junction
geometry and severe clashes, and publishes the best result plus every model and
a machine-readable report as typed artifacts. The appended residues are
explicitly labeled as a flexible modeled terminus that must be reviewed and
equilibrated before production MD.
Derived complexes retain a recursive modification history. Target tables show
the complete source/tool chain in Origin (for example
`PDB → Target trimming → MODELLER repair`) and the residue-level operations in
Preparation. Receptor, organism, UniProt, ligand identity, physicochemical
metadata, method, resolution, and title are inherited through any number of
typed parent jobs rather than only the immediate parent.
Structure Import still creates separate immutable `imported_target` and
reusable `prepared_target` artifacts underneath. Direct Import and Cleaning
routes remain hidden so old links and advanced handoffs work without
duplicating the primary sidebar.
The separate `Prepare > Compound Datasets` page imports SDF, SMI/SMILES, TXT,
CSV, and Excel XLSX/XLSM libraries as reusable `compound_set` artifacts for
downstream campaigns. Complete workbooks are retained unchanged. Users select
the worksheet and map ID/SMILES columns; all other selected-sheet columns are
preserved. RDKit validates and canonicalizes downstream SMILES, records rejected
rows separately, and calculates descriptive molecular properties without
applying screening thresholds. The workflow Results tab is the import-job
index; each job's Dataset tab provides usable structures, separately reported
invalid rows and multi-fragment components, numeric profiles, downloads, and
selectable 2D structures. Common salts, counterions, solvates and formulation
partners are recognized through the versioned
`mn_ligand/manifests/formulations.yaml` registry. Settings displays the active
registry read-only. Set `MN_LIGAND_FORMULATION_REGISTRY` to an extended,
schema-compatible YAML file and restart the app and workers to use a
site-specific registry.
Rejected rows are reviewed one at a time on the import job's Dataset tab.
The user may search PubChem by CAS or product name, compare the renderable
vendor-side candidate and PubChem structure side by side, confirm one PubChem
candidate, or skip and keep the row rejected. Decisions are immutable typed
review jobs. Product-name searches combine PubChem word-name results for the
complete vendor label and the base name after removing a trailing formulation
qualifier such as `(trimethylamine)`. Returned CIDs are deduplicated and each
candidate records which query matched it. Confirmed candidates are sanitized
and canonicalized again with
RDKit, receive freshly calculated descriptors, and are stored as
`reviewed_compound` artifacts explicitly labeled `PubChem confirmed import`;
the original workbook and vendor SMILES are never overwritten. Vendor
fractional hydrate/salt annotations such as `1/4 H2O` or `3/2 acetate` are
recognized for review previews. Formula comparison uses exact rational
stoichiometry, reports when PubChem stores an integer-scaled formulation unit,
and shows the PubChem disconnected-component count before confirmation.
After confirmation, the immutable reviewed artifact is included automatically
in the dataset's effective Usable view. For a multi-component PubChem record,
the user confirms one displayed single parent component; a unique largest
component type is selected by default, while alternatives remain selectable.
Only that one component is sanitized, described, and published for downstream
use. The full PubChem formulation SMILES, formula, multiplicities, and
selection method remain provenance. The source rejected row moves out of the
active queue but remains visible in Review history.
The job-specific `Parent duplicates` tab recalculates duplication across the
effective dataset after selecting each record's unique largest component and
neutralizing removable charge for a comparison key. It reports every member
of every group, distinguishes exact-parent from charge-standardized matches,
preserves stereochemistry, does not merge tautomers, and lists ambiguous
equal-size parents separately. This analysis and its downloadable CSV do not
rewrite the imported compound set. The adjacent `Docking-ready parents` tab
provides one representative row per unique stereochemistry-aware standardized
parent and aggregates all source IDs, products, formulations, and review
provenance.

Docking / Cofolding lists only completed compound-import jobs. It selects one
dataset and defaults to manual selection with the first parent visibly checked.
Users change checkbox rows or explicitly choose all unique docking-ready parents.
Launch creates a separate immutable completed
`compound-selection` job with the exact IDs and a typed `compound_set`; every
checked classical or cofolding engine consumes the same selection. For the
HY-L126 checkpoint (job `16259`), 766 effective usable records reduce to 724
unique docking parents with 42 redundant sources and 60 multi-component source
records.

The selection contract distinguishes chemical identity from the exact engine
modeling state. `identity_parent_smiles` is the deduplication/reference identity;
`modeling_smiles` carries the sanitized, pH-specific protonation and formal
charge actually submitted downstream, together with its preparation label and
pH. Classical engines consume the shared Scrub-prepared state. RosettaLigand
can preserve that input protonation/formal-charge state while still performing
its required conformer generation and MMFF94 partial-charge assignment; this
does not imply retention of Scrub's atom coordinates or per-atom charges.
Automated tests cover the handoff. Native validation is running as campaign
`0c3f2e1a-91ea-4e9a-8501-1a8778254ef7`: 28 unique parents, 16 prepared
targets, seven engines, and three repetitions.

Older run directories can be enriched with target/ligand identities and
derived modification history recovered from their retained PDB, SDF/SMILES,
and typed parent provenance. Preview first, then apply:

```bash
mn-ligand backfill-target-metadata
mn-ligand backfill-target-metadata --apply
```

By default the command queries the official RCSB Chemical Component API for
missing PDB receptor and ligand provenance. Use `--no-fetch-rcsb` for an
entirely offline backfill. Molecular coordinate and result artifacts are never
changed.

## Shared Reference Files

Boltz-2 and AlphaFold 3 use the same reference layout as
`mn-protein-design`:

```text
/mnt/db/reference_files/boltz_models
/mnt/db/reference_files/boltz_models/msa_repository
/mnt/db/reference_files/alignment
/mnt/db/reference_files/alphafold3
```

On another machine, configure the root once:

```bash
export MN_LIGAND_REFERENCE_DIR=/path/to/reference_files
```

Workflow pages use the layout names beneath that root and do not persist
machine-specific absolute reference paths.

The visible `System > Settings` page shows the effective results and reference
roots, directory availability, permissions, free space, and expected reference
subdirectories. Saving new roots writes
`<app-home>/config/runtime.json`, creates the directories, and applies them to
the current app process. Existing jobs are not moved to a newly selected results
root.

## AlphaFold 3 / AlphaFast, Boltz-2, and Nesso-1

The optional ligand-refolding adapter uses the externally supplied
`alphafast:latest` image:

```bash
export MN_AF3_IMAGE=alphafast:latest
```

The database directory must contain `mmseqs/`; the weights directory must
contain an `af3*.bin.zst` file. AF3 first looks up each protein chain in the
shared sequence-hashed MSA repository. Missing sequences use the local
AlphaFast MMseqs-GPU pipeline, and generated A3Ms are written back for later
jobs; no remote MSA server is used. Structure Import exposes AF3 recycling,
native model-seed range, and local-MSA batch controls. AlphaFold 3 code and
parameters remain subject to their upstream license and model terms.

Boltz-2 uses `ovoex-boltz2:latest` and expects `boltz2_conf.ckpt` plus
`boltz2_aff.ckpt` beneath `${MN_LIGAND_REFERENCE_DIR}/boltz_models` (or
`MN_BOLTZ_CACHE_DIR`). The typed refolding adapter publishes predicted CIF,
confidence, affinity, and metrics artifacts. One to 100 independently seeded
full predictions may be requested; each keeps a separate output directory and
multi-run campaigns publish mean/sample-SD affinity, binder-probability, and
confidence statistics. Structure Import disables the Boltz MSA server and
requires a matching MSA from the shared local sequence-hashed repository.

Nesso-1 uses `ovolig-nesso-cu128:latest`, built from the pinned local source in
`tools_to_implement/nesso`. It expects `nesso/v1.0.0/model.safetensors`,
`nesso/v1.0.0/hparams.json`, and the publisher-trusted `nesso/ccd.pkl` under the
shared reference root. The adapter reuses the existing
`facebook/esm2_t33_650M_UR50D` Hugging Face cache read-only. Native
`affinity_pred_value` is retained as log10(IC50 / µM), pIC50 is normalized as
`6 - affinity_pred_value`, and ensemble spread, binder probability, and cropped
protein-ligand entropy are published for interpretation. Nesso does not perform
structure diffusion and therefore publishes no `predicted_complex` artifact.
Its independent-run summary reports both log-space statistics and real IC50
values in µM.
All three campaigns require a running local worker. `Maximum compounds` defaults to `0`, which
processes every compound in the selected datasets in one directory-based engine
campaign; use a positive value to apply a cap.

## Pose validation

`Evaluate > Pose Validation` runs PoseBusters 0.6.5 against completed Vina,
GNINA, Uni-Dock Pro, RosettaLigand, Boltz-2, and AlphaFold 3 pose-producing
jobs. It uses the upstream `dock` configuration to test ligand chemistry,
bond/angle geometry, internal strain, distance from the protein, protein
clashes, and volume overlap. Nesso is intentionally absent because it does not
produce a structure.

Selection is performed only at source-job level in the editable coverage table.
Each checked job is reduced automatically to its scientifically relevant poses:
the best emitted classical/RosettaLigand pose per compound and repetition,
Boltz-2 model 0 per attempt, and AlphaFold 3 sample 0 per attempt. GNINA retains
both the best CNN-ranked pose and the best emitted Vina/minimized-affinity pose
per compound and repetition, collapsing them when both criteria choose the same
model. There is no second prediction-level selection step. The result viewer
displays the exact tested ligand, selection criterion, receptor context, and
failed checks for that row. A failed scientific check does not make the compute
job fail.

The source table reports PoseBusters coverage for every eligible docking or
cofolding job as not run, queued, running, completed, or failed/retryable.
Source selection defaults to all missing or failed-only jobs. Individual rows
can be checked or cleared directly, with shortcuts for all missing, all
eligible, or none. One submission queues one
immutable validation child per selected source so provenance and result pages
remain source-specific.
The page persists a versioned prediction inventory per immutable source under
`workdir/cache/pose-validation-inventory-v2/`. Inventories store source-relative
artifact paths and are built concurrently on first use, then reused on later
reruns. Status, engine, and text filters therefore do not rescan native result
folders. Validation children made with an older all-pose policy remain immutable
and accessible, but are marked as requiring a focused-policy refresh and are
excluded from current campaign statistics.

PoseBusters is CPU-native and uses its built-in process pool through the page's
worker-count setting. The image therefore does not request a GPU lease or ship
an unnecessary CUDA stack; it is compatible with an RTX 5090 host while leaving
the GPUs free for docking and cofolding. Standard dock-mode validation needs no
global reference files. Receptors, ligand topology, and complexes are copied
from their typed source jobs into each immutable validation job.

## Protein-ligand interaction analysis

`Evaluate > Interaction Analysis` analyzes both immutable prepared target
complexes and the focused poses from completed docking and cofolding jobs with
PLIP, PandaMap, or both engines together. Every non-polymer ligand in a prepared
target complex is inventoried by residue name, chain, author residue number,
insertion code, and heavy-atom count; the selected identity is passed to the
native adapter and retained in normalized output rather than silently choosing
a cofactor. Users can analyze all inventoried bound residues or a manual
multi-row subset, and coverage is tracked for the exact residue selection.
Source selection is per immutable job and defaults to missing or
retryable engine/source combinations. Each selected analysis engine is queued
as its own CPU job, so PLIP and PandaMap can run concurrently; within each job
independent complexes or poses are processed by a configurable thread pool.

AF3 and Boltz-2 commonly emit protein residues numbered from 1. For comparable
residue-level summaries, cofolded protein chains are globally sequence-aligned
to the immutable imported target and rewritten to deposited author chain and
residue IDs before either detector runs. Every prepared pose retains its
predicted-to-author residue mapping as native provenance.

The workflow publishes the exact prepared protein-ligand complexes, native
engine output, a normalized interaction table, and a per-pose summary. PLIP and
PandaMap results remain separate because their interaction definitions are not
interchangeable. Job Results provides interaction-type fingerprints, contacted
residues, a complex viewer, and PandaMap's explicitly labeled empirical-energy
estimate.

Completed interaction results are also valid starting points under
`Generate > Pharmacophore Hypotheses`. The user selects one exact analyzed
complex or pose. PLIP hypotheses can retain atom-resolved observed features
from native PLIP identifiers and add a distinct mandatory author-numbered
side-chain constraint. PandaMap and generic cross-engine associations have no
stable shared ligand-atom contact map, so those hypotheses remain explicitly
labeled as pose-level feature-class evidence. Every route stays editable and
saves a new immutable hypothesis.

Compound Campaign Comparison includes an `Interactions` tab. It respects the
selected dataset, launch campaign, target, and source-engine filters and shows
compound/engine coverage, separate PLIP and PandaMap fingerprints, recurrent
pocket residues, linked result provenance, and matched-pose contact-residue
Jaccard agreement. The agreement statistic compares detectors and does not
treat either engine as ground truth.

## Run A Container Directly

Example:

```bash
docker run --rm \
  -v "$PWD/examples:/input:ro" \
  -v "$PWD/output:/output" \
  ovolig-docking:latest \
  /bin/bash -lc 'cp /input/example-protein.pdb /output/protein.pdb'
```

For the OpenMM/OpenFE images, validate CUDA at runtime on the target machine:

```bash
docker run --rm --gpus all ovolig-md-cu128:latest \
  python -m openmm.testInstallation
```

## Results Explorer

Open `Results > Results Explorer` to find completed docking, cofolding,
rescoring, PoseBusters, PLIP, and PandaMap results without searching the Jobs
table by run ID.

Results are organized as:

```text
compound dataset
└── biological target
    └── prepared-target variant
        └── launch campaign
            └── prediction engine
```

The dataset view links to Compound Campaign Comparison. Target and compound
views provide the same campaign-aware navigation from their respective
perspectives. Multiple jobs from the same engine and campaign are summarized
in one row; expand the individual job history to open a specific repetition.
Failed and incomplete jobs are hidden by default and can be shown for
troubleshooting.

Prepared-target headings show where the structure came from and how it was
prepared. The displayed provenance includes the source PDB or other biological
origin, stable prepared-target job code, concrete artifact filename, ordered
preparation history, and a link to the target lineage. Thus a generated name
such as `target_longest_axis_x.pdb` is shown as an artifact of a target derived
from PDB 4LNW rather than as an ambiguous target identity.

Evaluation links are source-specific: a PoseBusters, PLIP, or PandaMap link
opens the latest compatible evaluation of that exact prediction job. Compound
links also open the result viewer on the selected compound. Nesso has no
structure-validation or interaction links because it does not emit a predicted
complex.
