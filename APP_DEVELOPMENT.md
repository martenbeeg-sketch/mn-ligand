# mn-ligand Development Handoff

Updated: 2026-08-03

This is the starting point for a new development session. Read these files in
order:

1. `APP_DEVELOPMENT.md` for current engineering conventions and verification.
2. `APP_feature.md` for the intended product and scientific architecture.
3. `APP_MISSING.md` for remaining work, migration history, and validation gates.
4. `README.md` for installation, startup, images, paths, and operator commands.

`APP_MISSING.md` contains historical planning as well as current status notes.
When they conflict, later implementation-status notes and the current code/tests
take precedence over older roadmap text.

Current collaboration instruction: accept the recorded verified baseline and
become familiar with the relevant implementation before continuing app work.
Do not rerun pytest, Streamlit AppTest, Docker/container smoke tests, GPU/native
scientific jobs, or other validation campaigns unless the user explicitly asks
for testing. Reading existing tests to understand contracts is allowed. This
instruction replaces older handoff language that automatically requested a
fresh audit or full-suite run at the start of every chat.

## Current checkpoint (2026-08-03)

MD finalization concurrency correction (2026-09-11): durable workers now
serialize advancement per MD workflow and finalization per MD child. Successful
terminal children with complete typed artifacts are treated idempotently, so
worker polling does not repeatedly hash multi-gigabyte trajectories. Atomic
artifact, workflow, MD-orchestration, campaign-extension, and runtime-settings
writes use writer-unique temporary files. Replacing a queued workflow child now
cancels the superseded child before it can consume resources. These contracts
prevent concurrent workers from racing on `.artifacts.json.tmp` and from
creating multiple executable MM/PBSA children for one production source.
Worker queue discovery is limited to the canonical task-group/run directory
depth and caches unchanged metadata records. Global pending-workflow scans are
also serialized and throttled, so idle GPU workers no longer recursively walk
trajectory and analysis output trees or consume a CPU core while CPU-only
endpoint work is running.

The local shared CPU pool is configured for 20 slots. Current AmberTools
MM/PBSA children retain their explicit 16-thread request, allowing one endpoint
job to run concurrently with a four-thread GPU simulation while preserving the
systemwide admission limit. Finalizer-race recovery must validate native return
code, success marker, and typed trajectory/structure/checkpoint artifacts and
retain an audit backup; `scripts/reconcile_md_finalizer_race.py` implements that
narrow repair without rerunning dynamics.

Curated MD complex selection (2026-08-05): Prepare now includes a Complex
Datasets page that imports an Excel pose-selection manifest, validates each
row against the immutable interaction-analysis job, source prediction job,
pose ID, compound, file checksum, and selectable ligand. The spreadsheet's
interaction-prepared complex path is selection evidence only: the importer
resolves the exact immutable docking/cofolding source artifact and materializes
the MD complex from that original pose and receptor, never from a detector's
PLIP/PandaMap working copy. It then publishes one typed immutable
`prepared_complex` record per accepted row. MD Simulation has
a separate Curated complex datasets input mode, so explicitly reviewed docking
and cofolding poses remain distinct from the general prepared-complex inventory
while retaining their target/campaign/prediction/selection lineage.
Campaign interaction-review workbooks expose these paths separately as
`Source pose path`, optional docking `Source receptor path`, and `Interaction
evidence path`; detector-prepared coordinates are never labeled as the source
prediction.

MD residue-numbering correction (2026-08-05): the MD launch boundary now
traces legacy prepared targets to their immutable imported structure and
derives deposited author IDs by coordinate/sequence alignment plus stable
chain-offset inference. It does not assume a selected, cleaned, or trimmed PDB
still contains author numbering. This closes the gap where PDBFixer changed
3GWS chain X residues 202–460 to chain A residues 1–259 before trimming and
OpenMM later renumbered the topology again. Aggregate analysis can publish an
immutable replacement revision that relabels stored interaction/RMSF results
without rewriting production jobs or rerunning trajectories. Completed 3GWS
workflow `9d240e37-b83e-4ab4-b352-6b0ce220d677` now uses analysis revision
`10810c23-1731-4781-a869-01e185c44525`, verified as deposited chain X residues
215–460; earlier incorrect aggregate revisions remain superseded history. The
Job Results renderer follows `superseded_by_run_id` links for MD aggregate
analysis, so an old bookmarked revision displays the current corrected plots.

Campaign review now separates physical launch campaigns from saved **Analysis
Sets**. `Results > Campaign Results` opens one physical campaign, shows its
canonical children and repeat completion, and can create an Analysis Set only
from compatible target-ligand redocking/refolding campaigns. Compatibility is
based on workflow purpose plus shared canonical reference-ligand identity; MD
is never mixed with docking/refolding. The creation preview exposes campaign
IDs, biological targets, reference ligands, shared targets, engines, and the
compatibility reason before saving. Opening an Analysis Set goes directly to
its aggregate scientific scope without rendering an irrelevant campaign
selector.

`Compound Campaign Comparison` is now an adaptive analysis workspace rather
than a uniform collection of tabs. It retains existing native metric,
PoseBusters, interaction, correlation, and molecular-viewer functionality while
adding campaign-shape-aware perspectives, prepared-target/engine coverage,
target-by-compound summaries, and entity drill-down. Prepared-target labels use
`PDB · ordered preparation codes · target job code`; the immutable run ID
remains authoritative. The in-app legend expands the preparation codes and the
full ordered origin remains visible in provenance.

Structural comparison distinguishes two questions:

- **Recovery to input** compares every predicted ligand with its own prepared
  target's coordinate reference ligand.
- **Pose agreement** compares engines/replicates only within the same prepared
  receptor frame. It can show every target separately, a pooled engine matrix
  as mean ± sample SD, or one target drill-down.

For identical ligands the comparison uses symmetry-aware heavy-atom atom
mapping; receptor alignment is applied to cofolded complexes but the ligand is
never independently fitted. GNINA exposes CNN-ranked and Vina-ranked poses as
distinct series. Matrix scales are fixed across campaigns, and linked 3D target
panels use the same scope, repetition policy, GNINA ranking, and target
presentation. A local 3D-only engine filter changes displayed structures
without silently changing the RMSD matrices. Viewer colors are unique for both
GNINA series.

Native metric plots and selected result figures can be exported as publication
PNG files in a ZIP. Offline metric export uses one row per entity/metric and
dedicated repetition columns bounded by the campaign's configured repeat count,
rather than a sparse engine-native wide table.

Compound preparation now separates canonical identity from the exact modeling
state. The shared selection artifact records `identity_parent_smiles` and
`modeling_smiles`, formal charge, preparation label, and pH. Classical engines
consume the Scrub-prepared state. RosettaLigand preserves an explicitly prepared
input protonation state while still generating conformers and MMFF94 partial
charges; it does not claim to preserve Scrub atom coordinates or partial
charges. Campaign `0c3f2e1a-91ea-4e9a-8501-1a8778254ef7` is the current
28-parent, 16-target, seven-engine, three-repeat shared-Scrub validation launch.

MD orchestration now has a public settings-template launch API:
`create_md_simulation_from_template`. It copies protocol settings from an
existing MD workflow but deliberately excludes the old source complex, ligand
payload, prepared system, residue map, checkpoints, trajectories, children,
comparison group, and duration-extension history. A new source always receives
fresh system preparation and newly seeded independent replicas. If the template
was extended, `target_duration_ns` is normalized into one fresh production
duration and step count; it is not replayed as an extension. Workflow
`9d240e37-b83e-4ab4-b352-6b0ce220d677` validates the API on prepared target
`7C4A9`: OpenMM, three independent 100 ns replicas from the outset, per-replica
AmberTools MM/PBSA children, and aggregate analysis. The focused MD module
baseline is 38 passing tests.

## Scope and neighboring application

`mn-ligand` is a ligand-centric application for reusable target preparation,
pocket prediction, docking/refolding, screening, molecule design, MD, and energy
evaluation. It is not a protein-design application.

`/home/user/programs/git-projects/mn-protein-design` is read-only inspiration.
Do not modify it. A tool may be ported only when requested, and must be adapted
to mn-ligand jobs and typed artifacts rather than sharing runtime state between
the applications.

## Runtime and portability

Runtime paths are resolved by `mn_ligand/runtime.py` and exposed on Settings:

- machine bootstrap: `${XDG_CONFIG_HOME:-$HOME/.config}/mn-ligand/installation.json`
  records the selected app home and optional required filesystem mount;
- app home: `MN_LIGAND_APP_HOME`, defaulting to
  `${XDG_DATA_HOME:-$HOME/.local/share}/mn-ligand` on a fresh installation;
  an existing `/mnt/data/RESULTS/mn-ligand-workdir` deployment is retained
  automatically for backward compatibility;
- runs: `MN_LIGAND_RUN_DIR` or the configured `runs_dir`;
- references/models: `MN_LIGAND_REFERENCE_DIR` or configured `reference_dir`;
- compound libraries: `MN_LIGAND_LIBRARY_DIR`;
- imported inputs: `MN_LIGAND_INPUT_DIR`.
- temporary files: `MN_LIGAND_TMP_DIR` or `<app home>/tmp` by default.

Run `mn-ligand init` once per machine to create the bootstrap record and the
app-home `config/runtime.json`. When `required_mount` is configured, app and
worker startup must fail before creating directories if that path is not an
active writable mount. Generated systemd worker units must include both
`RequiresMountsFor` and `ConditionPathIsMountPoint`. Existing installations
without a bootstrap record retain legacy/default discovery and must not be
moved automatically.

Artifacts stored in manifests must use run-relative paths. Never persist a host
absolute path as an artifact reference. Host paths may appear transiently in a
Docker command or runtime diagnostic.

Cross-job and shared-resource paths use the resolver in
`mn_ligand/core/portable_paths.py`. Existing recorded paths always win; only a
missing path is remapped to the configured runs, reference, library, or app
root. New operational metadata uses the `runs:///`, `reference:///`,
`library:///`, and `app:///` schemes. Historical command paths remain immutable
provenance. `mn-ligand portability audit` is read-only and must remain so.

`mn-ligand portability export DESTINATION` is the only supported metadata
rewrite path. It operates on a new staged copy, rewrites only recognized
operational fields whose files exist under a managed root, validates declared
artifact checksums, verifies that source JSON did not change, and atomically
publishes the destination. It must never edit the source archive, overwrite an
existing destination, rewrite command/container/provenance paths, or publish a
bundle with unresolved operational references. `portability verify` is
read-only. The portable bundle itself uses the ordinary app-home layout so a
new machine can point `mn-ligand init --app-home` directly at it.

New job writers that opt into `portability_schema_version: 1` must pass
`assert_job_portable(run_dir)` before becoming runnable. The durable worker
enforces the same check before executing the command. Canonical operational
fields may contain run-relative paths, `runs:///`, `reference:///`,
`library:///`, or `app:///` references, and documented container paths; host
absolute paths belong only in command/native provenance. Use
`mn-ligand portability check-job RUN_DIR` as the CI/preflight rule. The legacy
MD submission pages are read-only; all new MD jobs use the current MD
Simulation workflow.

Historical run folders are immutable inputs. Do not rename, move, or rewrite
them merely to fit a new schema. Legacy inference belongs in loaders.
The complete app home was relocated as one unit to
`/mnt/data/RESULTS/mn-ligand-workdir`; the checkout path remains a compatibility
symlink for historical metadata that contains the former host path.

## UI workflow contract

Scientific workflow pages use an ordered tab pattern:

1. `Target` or another primary reusable input.
2. Additional inputs such as `Compounds` only when scientifically required.
3. `Tool` or `Engine` with tool-specific settings.
4. `Run` with a configuration summary and active jobs.
5. `Results` with searchable jobs linking to a separate result-detail page.

Target selection uses the shared searchable table in
`mn_ligand/app/pages/discover_inputs.py`. Downstream pages consume prepared
artifacts and do not add separate protein, ligand, SDF, or SMILES uploads.
Imports belong under Structure Import or Compound Datasets.

## Job and artifact contract

Each scientific operation creates a reusable job. Parent workflows compose jobs
without hiding child provenance. Prefer artifact types over filename matching.
Important examples include:

- `imported_target`, `prepared_target`, `prepared_receptor`, `prepared_complex`;
- `compound_set`, `prepared_ligand_set`;
- `pocket_set`, `pocket`, `pocket_points`, `residue_score_table`;
- `pose_set`, `docking_scores`;
- MD systems, trajectories, analyses, and free-energy outputs.

Core definitions are in `mn_ligand/core/jobs.py`,
`mn_ligand/core/artifacts.py`, `mn_ligand/core/workflows.py`, and
`mn_ligand/core/pockets.py`.

## Current implementation landmarks

- Structure Import is the one-stop target and protein-ligand complex ingestion
  workspace. Completed Boltz-2 and AlphaFold 3 `predicted_complex` artifacts can
  be promoted into separate typed prepared-complex jobs without altering their
  prediction runs. Its Boltz-2 and AF3 preparation tabs are intentionally
  distinct from Discover campaigns: each accepts one FASTA/raw protein sequence
  plus one `LIGAND_ID,SMILES`, exposes engine settings, queues through an inner
  Run tab, and lists completed predictions for promotion.
- Target Sequence Modification is the visible Prepare page for imported
  complexes. Its `Trimming` tab applies chain-specific N/C-terminal residue
  bounds only to protein atoms while retaining the ligand. Its
  `C-terminal Repair` tab uses the interpreter selected by
  `MN_LIGAND_MODELLER_PYTHON` from the separately installed
  `mn-ligand-modeller` environment (see `environment-modeller.yml`) to
  generate a short-extension ensemble while retaining all source-complex and
  ligand coordinates. Both paths publish new typed `prepared_complex` and
  `prepared_receptor` artifacts without changing the source. The former
  Target Trimming and Repair routes remain hidden compatibility entry points.
- Derived target jobs retain a recursive `modification_history` snapshot.
  Target selectors resolve identity through the complete immutable parent
  lineage. Their Origin column reports the source and transformation chain
  (for example `PDB → Target trimming → MODELLER repair`), while Preparation
  reports the concrete residue-level changes. `backfill-target-metadata`
  recovers both target identity and this derived provenance for historical jobs.
- Compound Datasets is the only compound-library import workspace. It retains
  complete uploaded Excel workbooks, maps worksheet ID/SMILES columns, preserves
  additional source fields, and publishes RDKit-validated normalized sets plus
  descriptive reports without property screening thresholds. Formulation
  components are recognized from the schema-versioned packaged
  `mn_ligand/manifests/formulations.yaml`; site extensions use
  `MN_LIGAND_FORMULATION_REGISTRY` and must pass schema, ID, category, RDKit
  SMILES, and conflicting-assignment validation. Settings displays the active
  registry read-only.
- Rejected compound repair is never automatic or batched. PubChem CAS/name
  candidates are reviewed one rejected row at a time with side-by-side 2D
  structures. Accept and skip actions create immutable `compound-review` jobs.
  Name lookup must retrieve and deduplicate CID sets for the full vendor name
  and the base name without a trailing parenthetical formulation qualifier;
  each candidate retains its matched query terms and scope.
  Accepted SMILES must pass RDKit sanitization/canonicalization and descriptor
  calculation before a provenance-labeled `reviewed_compound` artifact is
  published. The source workbook and vendor SMILES remain immutable.
  Fractional hydrate/salt annotations may be removed only for a review preview;
  their stoichiometry must remain visible and be compared exactly against the
  PubChem formula, including integer-scaled formulation units and disconnected
  component counts. The effective dataset view merges the latest accepted
  `reviewed_compound` artifacts without rewriting the original compound set.
  A reviewed artifact must contain exactly one selected parent component.
  Multi-component PubChem records retain their complete formula, SMILES,
  component multiplicities, and selection method in provenance; they are not
  published as a disconnected downstream compound. Resolved rejected rows
  remain in review history rather than the active queue.
  Duplicate counts used for docking-library review are parent-aware rather
  than complete-formulation-SMILES counts. Select only a unique largest
  component, neutralize removable charge for the comparison key, preserve
  stereochemistry, do not conflate tautomers, and expose every grouped source
  row plus ambiguous-parent cases. The job result also publishes a
  `Docking-ready parents` view with one representative row per unique,
  stereochemistry-aware standardized parent and aggregated source aliases.
  This derived view never rewrites the imported workbook or source rows.
- Docking / Cofolding consumes reusable targets and compound sets and can check
  any combination of Uni-Dock Pro, Vina, GNINA, RosettaLigand, AlphaFold 3, Boltz-2,
  and Nesso-1, then queue every selected engine for the same inputs in one
  action. Engine parameters remain grouped on the same page. Classical docking
  and typed cofolding engines submit separate worker-owned campaigns with explicit
  GPU leases where required. One shared campaign control supplies one to 100
  explicitly seeded independent runs (default one) to Vina, GNINA, Uni-Dock Pro,
  RosettaLigand, Boltz-2, and AlphaFold 3; for AF3 this count becomes its native
  model-seed count. Nesso-1 retains a separate affinity-repetition control.
  Multi-run campaigns publish conditional replicate statistics. AF3 reports
  ranking score, ipTM, pTM, disorder, and clash evidence without treating those
  confidence measures as affinity.
  GNINA normalization parses every emitted MODEL block. Existing score columns
  remain backward-compatible aliases for the CNN-pose-score-selected model,
  while typed columns also record the empirical/minimizedAffinity-selected
  model and the CNN score/affinity attached to both selections. Historical jobs
  are interpreted dynamically and are not rewritten.
  Campaign analysis uses the logical `(launch campaign, engine, target)` key.
  This allows explicitly linked supplemental AF3/Boltz attempts to extend the
  original replicate distribution without creating another native-metric bar
  or consensus vote; raw child-job identity is preserved for filtering.
  Its Compounds tab lists only completed Compound Import jobs. Selection
  defaults to every unique docking-ready parent in one imported dataset; users
  may switch to a multi-row manual selection. Launch creates an immutable
  completed `compound-selection` job whose typed `compound_set` is consumed by
  every checked engine, so exact membership and provenance are shared.
  Compound Campaign Comparison resolves those selection artifacts back to their
  immutable source Compound Import job. It may join docking, cofolding, and
  rescoring campaigns only through recorded lineage, never by matching display
  names. Native metrics and units remain engine-specific; the optional combined
  ranking is calculated from within-campaign percentiles. Its Viewer supports
  both single-compound overlays and a selectable, linked-camera py3Dmol compound
  matrix. Across-compound pose summaries aggregate only within-compound,
  chemically compatible fixed-frame RMSDs; unlike compounds are never directly
  atom-mapped. Its Pose validation tab joins PoseBusters children through exact
  parent job IDs, keeps targets separate, and distinguishes PASS, FAIL, and
  untested engine-compound cells. Engine statistics report both the fraction of
  assessed compounds with at least one fully passing pose and the pass fraction
  across all assessed poses.
- `Prepare > Benchmark Datasets` owns reusable reference-complex imports.
  `mn_ligand/workflows/benchmark_datasets.py` defines the canonical schema,
  safe ZIP/TAR ingestion, explicit CSV/JSON/YAML manifests, generic layout
  detection, and PoseBench-aware Astex/PoseBusters/DockGen/CASP15 profiles.
  The source checkout under `tools_to_implement/PoseBench` is format and metric
  guidance only; benchmark execution never imports it at runtime.
  Import is the primary page content and uses
  `Input → Format → Validation → Run → Results`. Each Results row links its Job
  ID to the immutable dataset explorer and has a separate combined-results link
  for all derived redocking/refolding jobs. That combined page follows the
  standard `Overview → Artifacts → Metrics → Viewer → Lineage → Logs` layout.
- The visible `Benchmark` group contains one dataset-driven
  `Redocking / Refolding` page
  with `Dataset → Engines → Run → Results` tabs, matching the organization of
  Discover workflows. Redocking queues Vina, GNINA, Uni-Dock Pro, and
  RosettaLigand; Refolding queues Boltz-2, AlphaFold 3, and the explicitly
  structure-free Nesso affinity panel; Rescoring queues GNINA score-only and
  Boltzina where a case-matched Boltz-2 context exists. The former standalone
  benchmark routes and the old single-complex Redocking Benchmark route remain
  hidden for link compatibility. General target-based Refolding is visible
  under Discover. Benchmark campaigns derive one typed ligand-bound-chain
  receptor per case. The shared artifact is used by every redocking engine and
  RosettaLigand, while AF3/Boltz-2 receive only the corresponding protein-chain
  sequence. Selection maximizes protein atoms within 6 Å of the reference
  ligand, breaks ties by minimum ligand distance, and records the evidence and
  source artifact IDs.
- Docking / Cofolding and both Redocking / Refolding surfaces use the same
  grouped-engine interaction: every engine parameter accordion remains visible,
  a selected engine's accordion is expanded, and deselection collapses it.
  Select-all/deselect-all updates both campaign membership and panel state.
- Aggregate benchmark results are keyed by immutable dataset run, campaign,
  case, engine, replicate, and pose rank. Fixed-receptor redocking RMSD and
  protein-Cα-aligned refolding RMSD are labeled separately. Recovery at 1/2 Å
  is never calculated for Nesso because it emits no structure, and rescoring
  inherits the exact source-pose RMSD rather than recalculating or guessing it.
- Box-based Docking / Cofolding and Redocking Benchmark launches share a search
  region selector. `Fixed box` is the default with editable 20 × 20 × 20 Å
  dimensions. `Padding` defaults to 15 Å on every side of the selected source
  pocket, bound ligand, or stored region. The selected mode, padding, center,
  and calculated final dimensions are recorded in job provenance.
- Pocket Detection follows `Target`, `Tool`, `Run`, `Results`. Target selection
  shows every prior pocket job for that prepared target. The page is titled
  `Pocket Detection and Extraction`: a prepared complex defaults to the explicit
  `Bound ligand — generation pocket` method, selects the complex-owned ligand by
  chemical identity, and exposes ligand-envelope padding and protein-lining
  cutoff. This deterministic route creates the same immutable typed pocket job
  used by Molecule Design; it is visually and scientifically separated from
  predicted-site methods. fpocket, CPU-native P2Rank, and PeSTo can still be
  checked independently, configured in grouped panels, and queued together as
  durable worker jobs. Pocket Results render the selected pocket residues and
  docking box in the context of the complete prepared target.
- Scientific Run tabs now own launch and missing-input navigation actions and
  show the shared read-only CPU, queue, GPU-lease, worker-heartbeat, and per-GPU
  Free/Busy/Offline/Stale resource snapshot. This is applied to Docking /
  Cofolding, Redocking Benchmark, Pocket Detection, MD Simulation, Virtual Screening, Generative
  Design, ADMET, QC, and OpenFE launch sections.
- Docker image tags are installation-managed defaults from the tool registry or
  workflow constants. They are no longer editable on scientific pages; image
  identity remains recorded in command and job provenance and visible through
  Settings/diagnostics.
- PeSTo uses `--interface ligand`, publishes raw residue probabilities, and
  normalizes localized residue clusters into reusable pocket artifacts.
- A versioned ligand-owned tool registry now declares 16 app workflow/tool roles
  across the 11 image families referenced by mn-ligand plus
  `openvs:local`. It records artifact contracts, resources, references,
  licenses, health checks, and an explicit validated/implemented/experimental/
  compatibility/disabled/candidate status. The shared Docker command builder and `mn-ligand doctor`
  diagnostics are implemented. A separate `mn-ligand worker` can claim queued
  jobs atomically, acquire shared CPU-slot and per-GPU leases, maintain
  heartbeats, capture logs,
  honor cancellation requests, and recover dead stale leases. Docker command
  construction for every current app adapter now uses the shared runner and new
  runs write registry-backed command provenance. Pocket Detection is the first
  scientific page whose container execution, logs, GPU lease, native-output
  validation, and artifact finalization are worker-owned. Several legacy pages
  still retain synchronous direct-execution compatibility paths. Classical
  docking is fully worker-owned for Vina, GNINA, and Uni-Dock Pro; typed complex
  refolding is worker-owned for AlphaFold 3, Boltz-2, and affinity-only Nesso-1. Ordered AF3 pipeline and
  inference commands execute under one claim and GPU lease. Current MD
  preparation and production children are worker-owned, and worker startup
  advances pending MD workflow children without relying on a page render.
  Post-run MM/GBSA/MM/PBSA evaluations are separate immutable `md-mmgbsa` jobs
  that consume typed trajectory/topology artifacts and mount source runs
  read-only. Unified Jobs and generic Results now expose shared confirmed
  cancellation and immutable retry controls. Retry is enabled only for verified
  standalone worker adapters; workflow-managed children are rejected with a
  parent-aware replacement explanation.
- MD is one user-facing workflow with interchangeable OpenMM and GROMACS
  engines and internal preparation, equilibration, production, endpoint-energy,
  and analysis children. Selecting both engines creates separate immutable
  engine-native workflows linked by one comparison-group ID; checkpoints are
  never translated between engines. The engine view follows the grouped
  docking controls: select/deselect actions, one checkbox per engine, and one
  expandable preparation panel per engine. Protein, ligand, water, box, and
  physical conditions are independent engine inputs and are part of the
  compatibility fingerprint. The recommended OpenMM defaults are Amber
  ff14SB-family/OpenFF 2.2.0/TIP3P; the smoke-qualified GROMACS defaults are
  ff14SB/GAFF2/TIP3P. Production is configured in ns with task-oriented
  Smoke/Ligand MM/GBSA/Stability/Manual presets; Stability (three independent
  100 ns trajectories) is the UI default, while Ligand MM/GBSA starts from
  three independent 50 ns trajectories with endpoint jobs enabled. Task presets
  lock their analysis choices; Manual exposes both overrides. Independent
  trajectories receive distinct recorded seeds and their own unrestrained
  Roe-style NPT density revalidation. The configured duration is a minimum,
  not an automatic acceptance point: each replica extends in increments up to
  its maximum and production starts only after the same density plateau fit
  passes. Revalidation frames are excluded from production analysis;
  endpoint MM/GBSA is never embedded in production and instead creates one
  immutable `md-mmgbsa` child per completed replica. The final analysis publishes
  per-replica CSV/JSON plus mean and sample SD for available RMSD and endpoint
  terms. Automatic endpoint evaluation is opt-in and disabled by default;
  post-run evaluation is the recommended route after choosing a stable
  trajectory window. A new production run must re-equilibrate; it does not
  blindly continue a prior prepared system.
- Both engine panels expose a system-defining integration profile. The default
  is accelerated 4 fs dynamics with hydrogen masses repartitioned to 4 amu;
  standard 2 fs dynamics with normal hydrogen masses remains selectable. The
  chosen masses are applied once during OpenMM System or GROMACS topology
  construction and remain unchanged through all Roe stages, density gates,
  replica revalidation, and production. Timesteps above 2 fs fail validation
  unless HMR is enabled.
- MD Results loads only one requested DCD or XTC frame into the browser at a
  time. Complete trajectories remain native engine artifacts and can be opened
  through a downloadable PyMOL script or an explicit local-PyMOL launch action.
  The shared MDTraj/MDAnalysis-compatible analytics layer handles both formats
  and reports protein/ligand RMSD and RMSF, contacts, hydrogen bonds, secondary
  structure, radius of gyration, and persistent interaction networks.
  Historical production jobs remain discoverable from the MD Simulation
  Results tab.
- New MD preparation defaults to a Roe–Brooks 2020-inspired protocol for both
  engines. OpenMM uses its existing adapter. GROMACS uses app-owned reproducible
  `.mdp` files, molecule-local position-restraint includes, double-precision
  minimization, GPU dynamics, native `.top`/`.gro`/`.cpt`/`.tpr`/`.ndx`
  artifacts, and PBC-normalized XTC analysis input. Useful Pymacs concepts are
  adapted into these typed modules rather than executing its monolithic scripts.
  Protein and ligand heavy atoms are positionally restrained only during early
  solvent/solute relaxation. The ligand schedule is 5, 2, 0.1, and 0
  kcal mol-1 Å-2 through minimization, followed by 1, 0.5, 0.5, and 0 during
  short dynamics stages. Protein side chains are released before the backbone
  in the final restrained stage. All positional and planarity restraint
  parameters are explicitly zero before the unrestrained NPT density gate and
  remain zero in ordinary production.
- Roe's reported solvent setup is selectable directly: TIP3P in a truncated
  octahedron with 1.0 nm (10 Å) solute padding. OpenMM now passes
  `boxShape="octahedron"` rather than treating this selection as a cube.
- The density gate runs production-like NPT for at least 1 ns and extends in
  configurable 1 ns increments up to a configured maximum. It fits the
  published exponential relaxation model and separately records the final
  fitted slope, fitted-final versus second-half-mean difference, and residual
  chi-square criteria. CSV/JSON density artifacts survive a failed gate and the
  result page displays the trace and each criterion. This is described as
  inspired rather than exact because the current OpenMM system retains its
  normal H-bond constraints and hydrogen-mass repartitioning during the staged
  minimizations.
- New independent replicas apply that density gate again after assigning their
  recorded velocity seed. OpenMM and GROMACS preserve replica-specific
  density CSV/JSON artifacts. Historical inputs without the new
  `replica_density_revalidation` contract retain their fixed burn-in behavior,
  so running and completed jobs are not reinterpreted.
- GROMACS production can launch immutable `g_mmpbsa` endpoint jobs with native
  trajectory times derived from the selected percentage window and stride.
  Production remains on GROMACS 2026.3; the image also carries an isolated
  CPU-only GROMACS 2025.4 `grompp` utility that regenerates an endpoint TPR
  compatible with the GROMACS 2025 core bundled by `g-mmpbsa` 3.0.13.
  Geometric trajectory analysis is shared across engines; engine-native
  thermodynamics, checkpoints, and endpoint-energy implementations remain
  separate.
- Campaign Results supports two orthogonal in-place extensions. Repetition
  extension raises every canonical repeatable child to a requested count while
  preserving existing child IDs and adding only missing repetitions. Repeat
  identity is output-validated per engine; the existence of a nonempty
  `replicate_NNN` directory is never sufficient. Classical docking requires
  all expected compound PDBQT poses, RosettaLigand requires complete native
  score/silent output and an extracted pose for every chunk, Boltz-2 requires
  confidence plus structure output for every expected compound, Nesso-1
  requires readable native affinity output, and AlphaFold 3 maps complete
  structure/summary-confidence output back from its model seed to the logical
  repeat number. The extension layer computes exact missing IDs and supports
  internal holes without overwriting valid repetitions. This workflow-level
  contract applies to both Docking / Cofolding and Redocking / Refolding. MD
  duration extension raises the cumulative production duration of every
  completed replica without changing replica identity. These operations must
  not be conflated: one changes ensemble size, the other continues each time
  series.
- MD duration continuation dispatches by the workflow's immutable engine.
  OpenMM requires the prior DCD/log, its exact endpoint checkpoint, and matching
  serialized System and Integrator; the production runner explicitly writes a
  checkpoint after the final step and uses append-mode reporters. GROMACS
  requires the complete CPT/TPR/native-XTC/EDR/topology/index/log set, extends
  the run input with `gmx convert-tpr`, and executes `mdrun -cpi ... -append`.
  Neither adapter may fall back to coordinate minimization or velocity
  regeneration. Artifact/header/checksum incompatibility is a hard failure.
- A successful duration extension replaces the workflow's required production
  references with continuation children that depend on their prior replica.
  Existing endpoint-energy and aggregate-analysis children become historical;
  replacement children consume only the extended production run IDs. The
  workflow ID and campaign membership remain stable, and no extension is
  launched merely by rendering Campaign Results.
- The `ovolig-gromacs-cu128:latest` definition targets GROMACS 2026.3, CUDA
  12.8, AmberTools/GAFF2, MDTraj, MDAnalysis, and the pip-distributed
  `g-mmpbsa`. On 2026-07-29 the image built successfully and passed a CUDA water
  smoke plus shortened 4LNW Roe preparation, 10-step production, shared
  trajectory analysis, and one-frame MM/PBSA artifact smoke on an RTX 4090.
  A subsequent evidence audit on 2026-09-16 validated one native preparation
  (`EE761`), three exact 50-to-100 ns continuation replicas (`349FE`, `36AEC`,
  `D4B94`), and three immutable g_mmpbsa jobs (`511F5`, `CE394`, `AD9C7`).
  Every declared topology, coordinate, XTC, checkpoint, TPR, index, EDR,
  analysis, and endpoint artifact was present, and the focused GROMACS/MD
  regression suite passed. The adapter is therefore integration-validated;
  system-specific convergence and physical interpretation remain scientific
  review tasks. RTX 5090 runtime qualification remains an explicit task.
- The normal Roe–Brooks launch view does not show the irrelevant legacy
  Preview/Short/Longer step presets. Density-gate controls live under Advanced
  preparation settings. Selecting compatibility preparation explicitly reveals
  the legacy preset and step controls.

## Latest verified checkpoint

As of 2026-07-27, compound-import run
`fc0deeb8-0b96-49e0-b95c-162598c5dd4e` (job `16259`, HY-L126 worksheet
`Compound Information`) contains 766 effective usable records, 724 unique
docking-ready parents, 42 redundant source records, 60 multi-component sources,
and no ambiguous/excluded parent. The source workbook remains immutable.

Automated tests cover all-parent and manual multi-row selection, creation of the
typed `compound-selection` artifact, and the Docking / Cofolding Streamlit page.
The remaining scientific validation is a native docking campaign from a manual
subset, confirming exact compound membership and provenance in native and
normalized outputs.

The 4LNW design setup now has exact runtime inputs: minimized target job
`6CFD2`, PLIP interaction job `3DD68`, final residue-directed pharmacophore
`F73C5`, and bound-T3 pocket job `1585D`. PLIP observed eight reference
contacts, but its SER277 hydrogen bond is backbone-mediated. The active
hypothesis therefore replaces that backbone feature with a ligand-acceptor
post-pose review point toward author
`A:SER277:OG` (prepared `A:SER133:OG`) and records that it was not observed in
the reference. The next generation-specific scientific gate is a small native
campaign followed by docking/cofolding and exact side-chain-contact filtering.

Before changing behavior, inspect the relevant implementation and read the
existing tests as contract documentation because this list will evolve. Do not
execute them under the current collaboration instruction unless requested.

## Python environment and optional verification

Use the `mn-ligand` environment. Do not use the neighboring protein-design
environment or an arbitrary system Python.

Portable commands:

```bash
conda activate mn-ligand
python -m pip install -e '.[dev]'
python -m pytest -q
```

`mn-ligand install-launchers` generates machine-local wrappers in
`~/.local/bin` and an optional desktop entry. It uses the environment's exact
installed `mn-ligand` executable and does not depend on shell activation or a
`.bashrc` function. Generated launchers are installation state and are not
stored in the repository.

On the current workstation the explicit interpreter is:

```bash
/home/user/mambaforge/envs/mn-ligand/bin/python -m pytest -q
```

The verified baseline at this update is **340 passed**. Treat it as accepted
handoff evidence. The following focused/full commands are retained for a future
explicitly requested verification session; they are not an instruction to run
tests automatically:

```bash
python -m pytest -q tests/test_pocket_detection.py
python -m pytest -q
```

Tests live in `tests/`. Scientific adapters should test normalization,
run-relative artifacts, failed jobs, and handoff by artifact type. Do not make
the normal pytest suite depend on Docker, network access, model checkpoints, or
a GPU.

The 340-test baseline includes dedicated molecule-generation coverage for all
nine engine command handshakes, immutable per-engine settings, runtime-budget
termination, partial native-output recovery/normalization, the current
five-tab Streamlit campaign page, and the expanded tool-registry inventory.

## Optional Streamlit page tests

When the user explicitly requests UI verification, use Streamlit AppTest:

```python
from streamlit.testing.v1 import AppTest

app = AppTest.from_file("mn_ligand/app/pages/pocket_detection.py")
app.run(timeout=60)
assert not app.exception
```

Check important tabs, enabled/disabled run states, selected values, and tables.
The app starts with:

```bash
mn-ligand app --server.address 127.0.0.1 --server.port 8514
```

## Docker and GPU validation

Docker/GPU smoke tests are explicit integration checks outside pytest. Record
the target, image, GPU, run ID, native outputs, and normalized artifacts in the
relevant status documentation. Use GPU 1 when GPU 0 is occupied.

Every integrated tool must satisfy the validation gates in `APP_MISSING.md`:
reproducible image, declared model/reference files, actionable failures,
non-empty native output, typed normalized output, provenance, result display,
and a downstream artifact handoff.

The RTX 5090 policy is CUDA 12.8/Blackwell-ready images where feasible. CPU-only
tools such as fpocket must be marked as CPU tools rather than requesting a GPU.

## Safe change procedure

1. Read the relevant page, workflow, core types, and tests.
2. Preserve unrelated work in the dirty worktree.
3. Add or update typed contracts before expanding UI claims.
4. Keep unavailable engines visible only when useful, with execution disabled.
5. Run focused pytest and Streamlit AppTest.
6. Run the full pytest suite.
7. Run a real Docker/GPU smoke when adding or changing a scientific adapter.
8. Update `README.md`, `APP_feature.md`, `APP_MISSING.md`, and this handoff when
   architecture, implementation status, or validation commands change.

## Near-term continuation

The initial Docker tool registry, command builder, installation doctor, Settings
diagnostics, durable worker, and per-GPU file leases are present. The registry
scope is deliberately limited to images already used by the ligand app plus
OpenVS; unrelated protein-design images are excluded. Existing adapter command
construction is migrated. OpenVS is now an experimental CPU-only executable
adapter for the Rosetta GALigandDock stage, with VSH/VSX presets, containerized
ligand preparation, native and normalized outputs, result display, and a typed
best-complex handoff to MD preparation. The iterative OpenVS ML campaign and
optional CSD analysis remain out of scope.
Pocket Detection, classical docking, typed AF3/Boltz-2 refolding, and the current
MD Simulation orchestration have completed page-to-worker ownership migrations.
Completed MD production runs can now be evaluated repeatedly without mutation
through worker-owned endpoint-energy children. CPU/RAM/VRAM/scratch-aware
admission is now complete; the next infrastructure slice is supervised worker
service installation. That service slice is now complete: enabled systemd user
instances supervise GPUs 0 and 1 after rootless Docker, with stable identities,
explicit runtime paths, restart policy, and exact CID-based Docker cleanup.
Parent-aware retry/replacement for workflow children remains separate from the
now-complete standalone retry contract. P2Rank is now the third validated pocket
adapter. fpocket, P2Rank, and PeSTo are presented as complementary pocket
indicators; a comparative benchmark may be added later but is not a prerequisite
for selecting and running the adapters.

Shared-runner smoke status (2026-07-22): fpocket completed on prepared receptor
`54115f0a-d99b-4a8f-8130-c92493468da1`, producing three normalized pockets in
run `90cfaa74-cd12-45f1-be0c-8965e6ec3b35`. PeSTo completed on GPU 1 against the
same receptor, producing two pockets, raw residue scores, a scored structure,
and normalized point artifacts in run
`19c32837-aea1-4c13-b8d5-c2fa4a97834c`. Both retained registry-backed command
provenance. Failed run `04e489fa-8fbe-4f59-b8ff-498f4160875f` exposed an
incorrect direct-API default image and remains preserved; method-specific image
selection is now enforced.

Worker-owned smoke status (2026-07-22): fpocket run
`0d21cf53-3781-4c2c-ad0c-7df6fdb8d55c` was queued by the workflow and claimed by
the local worker, which produced eight normalized pockets and portable typed
artifacts. PeSTo run `9330c259-3554-4dce-89af-e026aea48f24` was claimed with an
explicit GPU 1 lease and produced two pockets, the native residue-score table,
the scored structure, and normalized point artifacts. Both completed with
native return code zero and registry-backed command provenance.

Worker-owned Vina smoke status (2026-07-22): run
`a59d4cf8-c58f-4818-a565-9f6c49526cf3` reused the typed 4WBK receptor and STE
compound artifacts from run `537818f6-4b68-426b-9577-f3c592e3069e`. The CPU
worker completed one compound with native Vina poses, a normalized docking-score
table, a combined typed pose set, portable logs, registry-backed provenance, and
successful Docking/Common Results AppTest display.

Worker-owned GPU docking smoke status (2026-07-22): GNINA run
`2c2a2671-caf1-43ef-b0ff-482a8777ff30` acquired GPU 0 and published native CNN
score/affinity values, normalized docking scores, a combined pose set, and logs.
Uni-Dock Pro run `2f86e28e-6fc8-40f3-a207-e4b91da67c9f` acquired GPU 1 and
published native affinity, normalized scores, a combined pose set, and logs.
Both reused the typed 4WBK/STE lineage, completed with return code zero, released
their leases, and rendered without exceptions in Docking/Common Results AppTest.

Typed redocking benchmark status (2026-07-22): workflow
`fc4382e1-80c0-4ad6-a2fd-40f3331a9908` reused the prepared 4LNW/T3 receptor and
crystallographic ligand artifacts and completed two replicates each for Vina,
GNINA, and Uni-Dock Pro. Six worker-owned Docker children published 54 native
poses. The parent normalized symmetry-aware heavy-atom top and best-of-N RMSD,
sample score/RMSD statistics, 1/2 Å recovery fractions, and six typed
crystal/top-pose overlays; every top pose recovered the crystal pose within 1 Å.
After the worker services loaded the orchestration hook, workflow
`a0bd23c4-b7bf-4027-94a5-9d1d7b409576` automatically finalized a new Vina
child into the parent artifacts without a page render.

Typed benchmark-dataset validation status (2026-07-28): the importer, unified
Benchmark Redocking / Refolding page, dataset-level RMSD aggregation, and regression
coverage are implemented. Official PoseBusters paper data were downloaded from Zenodo into
`/mnt/data/RESULTS/mn-ligand-workdir/benchmark_sources/`; immutable dataset
`d24ec3c9-cdec-4436-85bc-b9c86c8e5547` (`B9C86`) contains the explicit
one-case smoke subset `8AEU_M0L`. Native Vina, GNINA, Uni-Dock Pro,
RosettaLigand, Boltz-2, AlphaFold 3, Nesso-1, GNINA score-only, and Boltzina
execution completed. The smoke exposed and fixed three real handshakes:
Boltz offline runs now declare documented `msa: empty` single-sequence mode,
redocking discovers native PDBQT poses below replicate directories, and
workflow benchmark context survives workflow refresh through immutable
parameters. Protein alignment now rejects low-coverage incidental residue-ID
matches in favor of sequence correspondence while preserving exact-ID
alignment for sparse extracted pockets. The resulting fixed-receptor top-pose
RMSDs were 6.190 Å (Vina), 6.709 Å (GNINA), and 7.434 Å (Uni-Dock Pro) at
smoke-level exhaustiveness and three poses. These are integration observations,
not a reproduced PoseBench scientific result.

The four complete official PoseBench v1.1.0 archives from Zenodo record
`19138652` are checksum-verified, extracted below the same benchmark source
root, and registered as immutable datasets: Astex Diverse
`9f3acb37-2a00-42c5-afe9-36c3799cdabe` (85 cases), PoseBusters Benchmark
`76e6173a-aa7d-4d84-ae1d-03aa8091ce2b` (428), DockGen
`73603a2a-1513-40dc-ac15-b0f488cf6254` (260), and authoritative multi-ligand
CASP15 `73b3a36d-ecfc-4f77-9d4f-4fd530a2f2f1` (15 ligand-evaluable cases and
one rejected protein/ion-only reference). The earlier single-fragment CASP15
import is preserved but hidden as scientifically superseded. The importer
lazily reads server-local trees, canonicalizes DockGen PDB ligands to SDF, and
splits CASP15 holo references into protein/DNA/RNA target polymers plus all
small-molecule reference ligands without loading whole collections into memory.

General docking repetition smoke status (2026-07-23): native Vina run
`14ee27a1-c4be-48d6-8c16-091e5d9457ac` reused the typed 4LNW/T3 inputs, prepared
the ligand and receptor once, and completed explicit seeds 9101 and 9102 into
separate run-relative result directories. Scores were -9.134 and -9.457
kcal/mol; the normalized replicate summary reported mean -9.2955 and sample SD
0.2284 kcal/mol. Native two-seed GNINA run
`4ef9e700-98ca-40d4-a8b3-8763290eb021` reported mean -10.1010 and SD 0.0098
kcal/mol plus CNN mean/SD fields; Uni-Dock Pro run
`1621e149-be83-4502-a588-0a8f652f237d` reported mean -9.2700 and SD 0.2319
kcal/mol. OpenVS symmetry-aware clustering reports 0.892 Å between the previous
T3 seed-8101/8102 poses instead of the misleading 2.330 Å name-matched distance.

Worker-owned refolding smoke status (2026-07-22): AlphaFold 3 run
`f265dbd6-3ad7-4c38-ba85-de2b4594c087` acquired GPU 1, reused the cached 4WBK
sequence MSA, and published a native predicted CIF, confidence JSON, and typed
metrics table. Boltz-2 run `4ee3dc88-48c1-4e31-a25f-2a335eeac03d` acquired GPU
0 and published a native predicted CIF, confidence JSON, affinity JSON, and
typed metrics table. Both reused the typed 4WBK/STE lineage, returned zero,
released their leases, and passed Docking/Common Results AppTest display.
Both refolding engines default `Maximum compounds` to `0`, meaning every
compound in the selected typed datasets. Positive limits cap the campaign;
AF3 pipeline batch size remains a separate execution setting.

Independent learned-prediction repetition smoke status (2026-07-23): typed
4LNW/T3 Boltz-2 run `d944ef26-067f-46a4-8518-223193ce54ab` and Nesso-1 run
`4079bb79-6177-42d9-a935-e97df8ba3ad9` used explicit seeds 1201/1202 in separate
run-relative output directories under one worker claim per campaign. Boltz-2
used the existing local MSA with its server disabled. Nesso published two native
affinity JSON files and both log-space and µM aggregate statistics while
preserving `structure_output = false`.

Worker-owned MD endpoint-energy smoke status (2026-07-22): run
`b43f0147-2b76-4ba9-9156-8dd1817e764a` consumed the typed trajectory and final
topology from production run `17b4bfca-629f-407e-9381-615bed7d381a`, acquired
GPU 0, and evaluated frame 199 from a 200-frame trajectory with the OpenMM GBSA
backend. It completed the native frame evaluation and published portable summary,
per-frame CSV, metadata JSON, plot, structure snapshots, and native-result
artifacts. SHA-256 hashes of the source run's metadata, input, and result were
identical before and after execution. Focused worker/MD/AppTest validation and
the full **126 passed** suite completed successfully. AmberTools remains an
available existing backend choice; this smoke specifically validates OpenMM.

Worker-control smoke status (2026-07-22): run
`worker-control-cancel-smoke-20260722` started a real attached
`ovolig-md-cu128:latest` Docker container with a 120-second sleep. Cancellation
was requested through `mn_ligand.core.job_control`; after the grace period the
worker escalated to a hard stop, recorded terminal `cancelled` with return code
`-9`, released its atomic claim, and Docker reported no remaining container.
Automated coverage also verifies cancellation during a Docker-shaped GPU job,
GPU lease release, queued cancellation, immutable retry input reconstruction,
successful retry execution, monotonic attempt lineage, both Streamlit action
surfaces, and read-only page refresh. The full suite passes at **134 tests**.

Worker resource-admission smoke status (2026-07-22): the native snapshot reported
64 CPU threads, 448+ GiB available RAM, GPU 0/1 with approximately 23.2/20.2 GiB
free VRAM, and only 3.58 GiB free on the run filesystem. Run
`worker-resource-admission-smoke-20260722` declared 4 GiB scratch and remained
queued with the exact shortage recorded; no stdout/stderr was created, no Docker
container launched, and the temporary claim was released. The smoke record was
then cancelled so it cannot execute after disk space changes. Automated tests
cover temporary waiting without head-of-line blocking, permanent CPU/RAM/
scratch/VRAM rejection, VRAM-aware GPU selection, UI reason display, and
deterministic capacity fixtures. The current full suite passes at **141 tests**.

Supervised-worker service status (2026-07-22): the generated
`mn-ligand-worker@.service` template passed `systemd-analyze --user verify` and
enabled instances 0 and 1 under a lingering user manager. GPU runs
`worker-service-gpu0-20260722` and `worker-service-gpu1-20260722` completed on
their scoped workers with native CUDA-image output and released leases. The
first stop smoke exposed an attached Docker-client orphan edge case and remains
preserved as failure evidence. After adding run-local CID tracking, run
`worker-service-stop-cleanup-gpu0-20260722` proved that service interruption
records a preserved failure, force-removes the exact container, releases the
claim/lease, and permits a clean restart. Focused worker/service coverage passed
at 32 tests, Streamlit AppTest passed at 10 tests, and the full suite passed at
**151 tests**.

Streamlit worker-health status (2026-07-22): supervised workers now publish
atomic heartbeat records under `.worker/workers` with stable identity, PID, GPU
scope, idle/running state, current run, and selected GPU. Settings merges these
with read-only `systemctl --user show` properties, queued-job count, and GPU
lease files. It does not start or stop services. Missing user-bus access and
stale heartbeats remain renderable, actionable states rather than page errors.
The live snapshot showed both GPU services enabled and `active/running`, fresh
idle heartbeats, and no queue or leases. Focused health/lifecycle/Settings
AppTest coverage passed at 23 tests; the full suite passed at **153 tests**.

CPU/GPU worker separation (2026-07-25): generated GPU service instances accept
only GPU-requesting jobs, while `mn-ligand-cpu-worker.service` accepts only CPU
jobs. This prevents Vina, RosettaLigand, P2Rank, and other CPU-only work from
occupying a GPU worker slot while a physical GPU remains lease-free. Worker
heartbeats record their job class. Run-page resource reporting separates GPU
lease availability from worker-slot state and shows the CPU worker explicitly.
The manual worker default remains `mixed` for backward compatibility; generated
systemd units always declare `cpu` or `gpu`.

Shared CPU-pool scheduling (2026-08-04): all CPU-only and GPU-backed jobs now
reserve their declared `cpu_threads` from one atomic slot pool before execution.
The pool is capped by the smaller of host-visible CPUs and the runtime
`cpu_process_limit` (currently 16); legacy or zero-valued declarations reserve a
minimum of one slot. GPU workers use the same pool and release a provisional CPU
reservation if no suitable GPU lease can be acquired. Per-slot lease files carry
heartbeats, support rollback after partial allocation, and recover stale leases
whose owner has died. Requests larger than pool capacity are rejected as
impossible, while temporary exhaustion preserves the queued job and admission
reason. Settings exposes leased slots/capacity. Focused resource, worker, and
health coverage passes at 34 tests. The implementation is intentionally not
activated by restarting services while existing scientific jobs are running;
running worker processes finish with the scheduler code they originally loaded.

RosettaLigand pocket-center anchor correction (2026-07-25): job `35673`
demonstrated that Rosetta `mol2genparams.py` may name anchor outputs from the
source MOL2 molecule title (a compound ID) even when the requested internal type
name is `reference_anchor`. The runner now discovers the emitted params and PDB
and canonicalizes them to `reference_anchor.params` and
`reference_anchor_0001.pdb` before copying or centering the anchor. This stable
contract applies to reference-guided and pocket-center placement; historical
failed runs remain unchanged as provenance.

Target-associated reference inheritance (2026-07-25): Docking / Cofolding now
uses the selected target's coordinate-bearing associated ligand as the default
reference artifact. An explicit reference selection is an override. When the
optional principal-axis transform is enabled, the reference is resolved to the
corresponding emitted ligand in the shared transformed receptor frame before
any engine jobs are queued. Classical docking, RosettaLigand, AlphaFold 3, and
Boltz-2 inputs persist this typed reference. Result viewers also resolve a
target-sibling coordinate ligand as a read-only lineage fallback, allowing
historical job `DF69D` to display its aligned T3 without modifying that
immutable run.

P2Rank smoke status (2026-07-22): the CPU-only `ovolig-p2rank:latest` image
builds P2Rank 2.6-alpha.5 from local source commit
`d8c8e0d870f79a36b7dbb176edf8c019b6cc789c` on digest-pinned Java 17 bases.
Upstream 1FBL produced four native pockets. Worker run
`6fe5efa5-60b8-4225-9f46-4d1a0b68dac9` consumed the typed 4WBK receptor,
published native pocket/residue tables and SAS points plus one normalized pocket,
used no GPU lease, and passed Pocket Detection/Common Results/Docking handoff
AppTest. Invalid-input run `fd765101-5034-4db5-8fbe-af6efd7885e2` preserved an
actionable failure despite native return code zero. The image digest is
`sha256:ab8e214735f13d39eabfae8310408f2df203eac238e108bae607d1d1f35d3560`;
50 focused tests and the full **157 passed** suite completed successfully.

Nesso-1 smoke status (2026-07-22): `ovolig-nesso-cu128:latest` is built from
local source commit `64e8a92b48d1dd3b0b33a4b64b613758fba04a17` on PyTorch
2.7.1 CUDA 12.8. The image includes `sm_120` and `compute_120` support for RTX
5090/Blackwell while using `--no_kernels` for the initial compatibility route.
Worker retry run `d595d871-2c43-4ba7-87ba-7b2f9423dab0` acquired GPU 1 and
completed native 4WBK/STE inference. It published the native affinity JSON and
normalized affinity table with log10(IC50 / µM), derived pIC50, ensemble spread,
binder probability, and cropped protein-ligand entropy. No predicted structure or
pose artifact is published. The source failed run remains immutable as retry
provenance. Focused workflow/page coverage passed at 34 tests, the completed-run
Job Results AppTest passed, and the full suite passed at **162 tests**.

PoseBusters validation status (2026-07-25): the CPU-only
`ovolig-posebusters:latest` image is built from pinned upstream commit
`1a5f26aa7270fafba21b7fec8b3633f4c4e45ead` and has local image ID
`sha256:6d23ae576afb3803c1757ff0341ad43738df3a99bf261b8e9cd229e638dbbbf0`.
The image healthcheck reports PoseBusters 0.6.5 and intentionally requests no
GPU on the RTX 5090 host. Worker smoke run
`97924dd4-276f-4a88-ac7f-edee7ab938d3` validated two native Vina poses with
22/22 applicable checks passing. Boltz-2 run
`8fa55c48-19e7-4ef2-9560-031195926e8a` successfully reconstructed both model-0
ligands from the exact native YAML topology and reported reproducible geometry,
strain, and protein-overlap failures rather than an adapter failure. AlphaFold 3
run `92b3fadd-efa8-42a1-87d2-a8cf0b15c089` reconstructed sample 0 from its
native JSON topology and passed 22/22 checks. All three jobs completed through
the supervised worker, retained native reports, and used two CPU processes
without a GPU lease. Focused workflow and Streamlit coverage passes at 36 tests,
and the full suite passes at **279 tests**.

Independent positive Boltz-2 control (2026-07-25): run
`aee4cb37-377a-471a-979a-25c218545c75` selected the high-confidence
`ZINCsI000006lZac` model-0 prediction from each of three attempts in the separate
10-compound campaign `237df5d6-8f0e-429a-a9b5-a2e82b4dd7e8`. PoseBusters ran
the three structures concurrently and all three passed 22/22 applicable checks
without topology-preparation errors. Together with the retained failing T3
control, this distinguishes a genuinely broken cofolded ligand geometry from an
adapter or CIF-to-SDF reconstruction failure.

Focused PoseBusters selection update (2026-07-27): source-job selection now
materializes only the best emitted classical/RosettaLigand pose per compound and
repetition, Boltz-2 model 0, AlphaFold 3 sample 0, and GNINA's CNN-best plus
Vina/minimized-affinity-best pose per compound and repetition. Coincident GNINA
choices are deduplicated. The cached inventory schema is version 2 and immutable
validation children record `best-scientific-poses-v1`. Older all-pose validation
children and native reports are preserved, but the UI marks their sources for a
focused refresh and campaign statistics exclude them. The rebuilt image digest
is `sha256:756189c8936bddd13bd8d79cb3a4cbe8ac51e6020ef55317ad4aa5e6d764850a`.

PLIP/PandaMap interaction-analysis update (2026-07-27): both tools are exposed
under one `Evaluate > Interaction Analysis` page as separate CPU engines.
Selecting both queues independent immutable children that can run concurrently;
each child also processes independent poses with a configurable thread pool.
The adapters consume the focused scientific pose inventory, reconstruct the
exact receptor/ligand complex, retain native output, and publish normalized
per-contact and per-pose tables. Compound Campaign Comparison has a filtered
`Interactions` tab with coverage, engine-specific fingerprints, recurrent
residues, provenance links, and matched-pose PLIP/PandaMap residue-overlap
statistics. Joint worker smoke jobs `1F469` (PLIP) and `A3B55` (PandaMap)
completed from the same Vina source and emitted 14 and 17 normalized
interactions respectively. Cofolded AF3/Boltz structures require an additional
numbering normalization: their protein residues commonly start at 1. Before
analysis, each predicted protein chain is globally sequence-aligned to the
immutable imported target and rewritten to its deposited author chain/residue
IDs (for example, 4LNW A:1–263 becomes A:145–407). Each pose retains a native
predicted-to-author mapping JSON. Ligand selection uses chain and sequence ID
because mmCIF residue names such as `LIG1`/`LIG_L` are truncated to `LIG` by
PDB serialization. PandaMap interaction detection remains parallel while its
Matplotlib rendering is serialized because the renderer is not thread-safe.

Interaction validation audit (2026-09-16): the immutable archive contains 343
completed native PLIP jobs and 343 completed native PandaMap jobs. The current
dated images were also rerun in isolated temporary workspaces against three
cofolded poses: PLIP emitted 31 normalized contacts and PandaMap emitted 127
contacts plus three native diagrams, with zero failures. The focused numbering,
normalization, result-display, comparison, and handoff suite passed. Both
adapters are now integration-validated. PLIP contacts remain geometric
classifications; PandaMap delta-G values remain empirical estimates rather than
experimental or rigorous free energies. Most later archived failures were a
resolved deployment-code mismatch involving `NATIVE_THREAD_ENVIRONMENT`, not
native tool failures; historical records remain unchanged.

## Results Explorer status

Results Explorer update (2026-07-27): `Results > Results Explorer` is the
cross-workflow entry point for completed docking, cofolding, rescoring, and
their evaluation children. The index is assembled read-only from immutable job
metadata and typed inputs; it does not rewrite historical run folders or
artifact paths. Its views cover overall inventory, compound datasets,
biological targets, and compounds.

The hierarchy is dataset → biological target → prepared-target variant →
launch campaign → engine. Explicit `launch_campaign_id` metadata is
authoritative. Historical engine jobs without that metadata are grouped only
when their dataset and prepared target match and their launch times fall within
the bounded historical grouping window. Rescoring jobs remain attached to
their source prediction campaign. Repeated AF3, Boltz-2, or other engine jobs
within one campaign are summarized as one engine row with a run count; their
individual immutable job links remain available in the history expander.
Failed and incomplete jobs are hidden by default and can be enabled for
troubleshooting.

Prepared targets are identified by scientific provenance rather than by a
generated filename alone. Each target section exposes the source PDB or other
biological origin, stable prepared-target job code, actual artifact filename,
ordered modification history, and a link to the prepared-target result and
lineage page. For example, EC98E is presented as a prepared target derived from
PDB 4LNW, with
`Modified-residue mapping → PDBFixer cleaning → OpenMM minimization → Target
trimming → Ligand-axis target orientation`; `target_longest_axis_x.pdb` remains
the artifact detail, not the target identity.

Prediction links and the latest compatible PoseBusters, PLIP, and PandaMap
links remain attached to the exact prediction job assessed. Compound links add
the compound ID as a query parameter so result viewers open on the requested
compound. Nesso has no structure-dependent evaluation links because it does
not emit a predicted complex.

The Streamlit index cache includes an explicit schema version in addition to
the runtime metadata revision. Renderers retain fallbacks for rows created by
an older schema during hot reload, preventing stale cached indexes from raising
missing-column errors. Focused Results Explorer tests pass, its Streamlit
AppTest reports no exceptions, and `git diff --check` is clean as of this
update.

## Generate architecture in progress

The former disabled Generative Design placeholder has moved into a dedicated
`Generate` navigation group with `De Novo Molecule Design` and task-specific
existing-ligand design pages. Pharmacophore hypothesis creation lives under
`Prepare > Pharmacophore Hypotheses`. The intended
campaign contract covers the
local OMTRA, PocketXMol, FLOWR.root, conDitar, conDitar + paOPT, DrugRPG, PFM,
PocketFlow, and PGMG checkouts.

Pharmacophore intent is app-owned rather than engine-owned. A versioned,
immutable `pharmacophore_hypothesis` stores enabled features and an internal
`required` compatibility flag now presented to users as `post-pose review`,
coordinates, tolerances, optional directions, source atom/residue identity,
notes, and provenance. Hypotheses may begin from an RDKit BaseFeatures
analysis of a coordinate ligand, one exact PLIP/PandaMap-analyzed target
complex or prediction pose, a prior immutable hypothesis, or manual features.
The interaction-derived route carries the interaction job, engine, exact
pose/complex membership, contacted residues, and interaction classes into
provenance. PandaMap and the generic cross-engine route remain pose-level
feature-class evidence. PLIP can additionally create atom-resolved observed
features from its retained native atom identifiers and add a separate
author-numbered target side-chain review point. That point is explicitly
labeled as inferred geometry and must be inspected after docking/cofolding or
refolding; it is never presented as a guaranteed or observed reference contact.
Saving publishes
canonical JSON and CSV plus deterministic Pharmit JSON, OMTRA XYZ, and—when
compatible—PGMG `.posp` representations.

The hypothesis page places the editable feature table beside a persistent
py3Dmol molecular viewer. It renders the analyzed complex, bound ligand,
feature tolerance spheres, target atom selected for later review, and
point-to-atom constraint.
Row focus and fine-adjustment controls cover feature type, XYZ, radius,
enabled, and post-pose review state. A guided target-interaction selector maps target
donor/acceptor, ionic, hydrophobic, aromatic/π, and halogen intent to the
complementary ligand feature with engine-support guidance. Saving recalculates
target direction and distance from the visible row and creates a new immutable
hypothesis; direct 3D dragging is not implied by the current viewer.

The De Novo Molecule Design page defines shared target, pocket, reference
ligand/scaffold/fragment, pharmacophore, and optimization intent once. Its
Engine tab records how each model will consume or derive that intent;
automatic OMTRA ligand pharmacophores and FLOWR.root ProLIF interactions must
not silently replace the shared hypothesis.

When the selected prepared target is a protein–ligand complex, De Novo Molecule Design
uses that exact complex-owned `prepared_ligand_set` as the default reference
ligand and does not offer unrelated docking poses or compound sets. If a
completed bound-ligand pocket is linked to the same immutable target, the latest
such pocket is supplied automatically. For minimized 4LNW job `6CFD2`, this
means T3 is the fixed default reference and bound-ligand pocket `1585D` is the
automatic coordinate pocket. Targets without a complex-owned ligand retain the
explicit reference selector; targets without a completed bound-ligand pocket
show a specific preparation warning. An inline provenance expander explains the
bound-ligand construction, box padding, lining-residue cutoff, center and size,
and links both to the immutable Pocket Detection result and to target-scoped
pocket creation/replacement.

Pharmacophore selectors display the user-saved hypothesis name first and retain
the short immutable job code as the revision identifier. Generic filenames such
as `pharmacophore.json` are not used as the primary user-facing label. Selecting
a hypothesis on the De Novo Molecule Design Conditioning tab also opens a read-only
viewer of the associated target/reference complex, enabled feature tolerance
spheres, and post-pose-review point-to-target-atom vectors. The user can focus a
feature, toggle observed reference features and labels, inspect the complete
feature inventory, or open the immutable-hypothesis editor.

Nine engine roles and separate Docker Compose images use the normal
`<reference-root>/generation/<engine>/...` layout. The shared image recipe
preserves the CUDA 12.8 Blackwell PyTorch base instead of installing the
projects' historical CUDA 10/11 environments. OMTRA, PocketFlow, PGMG,
PocketXMol, FLOWR.root, conDitar, paOPT, DrugRPG, and PFM have experimental
native-command adapters, immutable campaign/child jobs, normalized
stereochemistry-aware SDF/CSV output, result display, and typed compound-set
handoff. conDitar and paOPT use separate images while sharing the permissioned
Diff/PocketAE reference bundle; their source, weights, and images must remain
inside the authorized group deployment and must not be redistributed.

The Engine tab exposes only native controls that the reviewed inference path
consumes: OMTRA integration/stochastic/size controls; PocketFlow atom/bond
temperature and growth geometry; PocketXMol diffusion/size/redesign controls;
FLOWR.root integration/correction/diversity controls; conDitar and conDitar +
paOPT diffusion and pocket radius; paOPT multi-endpoint direction, steering,
gradient-pair, and perturbation controls; and DrugRPG atom count. PFM retains
its validated learned
size prior and fixed ODE sampler, while PGMG is directed through the editable
pharmacophore. Engine selection is a separate checklist at the top of the tab,
with Select all and Deselect all actions. Every engine has its own configuration
expander and its controls remain instantiated. Selecting an engine expands that
panel and includes the engine in the campaign; deselecting it collapses the panel
and excludes the engine without discarding its parameter values. Manually opening
or closing an expander does not itself change campaign membership.

Per-engine attempts, batch size, seed, and optional inference-runtime limits
are recorded in the immutable campaign. The time-cost planner uses a
conservative historical P75 rate when typed runs exist, otherwise a labeled
seed rate. It estimates time and conservative historical P25 unique-compound
yield for requested attempts, then suggests both attempts and expected unique
outputs for a time window. paOPT accounts for
`1 + steps * (2 * gradient_pairs + 1)` diffusion passes. At a runtime deadline
the worker removes the recorded container, marks the job timed out/failed, then
best-effort imports and normalizes complete native molecules already written.
Finalization may briefly extend beyond the inference limit, and recovered
partial output never changes the job to completed.

Prepared-target interaction update (2026-07-27): Interaction Analysis now
accepts completed immutable `prepared_complex` jobs that retain a
`prepared_ligand_set`, in addition to docking and cofolding predictions. It
enumerates each non-polymer ligand residue as a separate candidate and records
the exact residue selector in the immutable input inventory and normalized
PLIP/PandaMap artifacts. HETATM ligands with peptide-like atom names such as
T3 (`LIG` coordinates containing `N/CA/C/O`) remain ligands rather than being
misclassified as polymer. Chemical identity (`T3`) and coordinate selector
(`LIG|A|501|_`) are both retained. The page accepts all candidates or a manual ligand
subset and tracks engine coverage over exact selection IDs. Both adapters honor
that selector; legacy prediction inputs without one retain their existing
single-ligand behavior.

The exact minimized 4LNW/T3 path was executed as scientific analysis job
`3DD68` (PLIP), not as a test. It found eight contacts. Its SER277 hydrogen bond
uses the backbone nitrogen (`sidechain=false`), so the final immutable
hypothesis `F73C5` keeps that observation only in PLIP provenance and replaces
its active pharmacophore point with a ligand acceptor review target directed to
author `A:SER277:OG`, explicitly mapped to prepared `A:SER133:OG`. Exact
bound-ligand pocket job `1585D` defines the T3 box and lining residues from the
minimized coordinates.

Subsequent focused native generation established typed results for all nine
engines on the 4LNW/T3 path. Campaign `1EE83` contains completed OMTRA, PGMG,
PocketFlow, PocketXMol, PFM, DrugRPG, and FLOWR.root children. Comparison
campaign `40ED1` contains completed conDitar job `38A85` and paOPT job `F7625`.
These validate native launch, normalization, and typed display; they do not
establish target-specific enrichment or the intended SER277 side-chain
interaction.

Generated-molecule qualification update (2026-07-28): RDKit-readable
normalization is no longer treated as sufficient downstream acceptance. Every
new completed molecule-generation job automatically creates an immutable
CPU-only `molecule_qualification` child. Canonical stereochemistry-aware SMILES
is the common chemical identity; native engine coordinates remain unchanged as
provenance. The child applies single-fragment, element, radical, charge,
heavy-atom, severe-reactivity, and synthetic-accessibility gates, then creates
deterministic ETKDGv3 conformers, optimizes them with MMFF94s or UFF, and runs
PoseBusters 0.6.5 in molecule-only mode. Core chemistry, connectivity,
bond/angle, clash, and energy failures remain hard exclusions. An isolated
`non-aromatic_ring_non-flatness` result is retained as a typed review warning
because the check is a conformational heuristic; such a record is accepted for
docking with its warning attached and should receive stronger optimization if
promoted. Pocket placement is deliberately left to docking/cofolding.

Within a qualification job, conformer generation runs concurrently over
compounds with single-threaded force-field calls, followed by PoseBusters'
process pool. The effective worker count is capped by both the configured limit
and the number of input compounds. The durable services use the shared CPU-slot
pool, so this work can coexist with MD analysis and other CPU consumers without
exceeding the systemwide CPU budget.
The initial 2026-07-28 immutable backfill created 13 child jobs for all
historical completed generation runs. A version-2 immutable backfill then
applied the warning policy without changing those first results: all 13 revised
jobs completed, 34 of 42 normalized molecules were accepted, and all 13 jobs
published downstream compound sets. Thirty-two molecules passed PoseBusters
strictly; the two paOPT-related singletons were accepted with the isolated
non-aromatic-ring warning. PFM retained 1/3.

Target-centric design-results update (2026-07-28): each prepared target row in
`Structure Import > Results` now reports its molecule-design campaign count and
links to one combined `Molecule Design Results` page. The page joins the exact
target to all generation campaigns, engine children, and the latest immutable
qualification revision. Summary metrics cover generated, accepted,
stereochemistry-aware unique, strict-pass, and warning counts. Per-engine
qualification bars and a QED-versus-molecular-weight plot provide overview.
The compound table filters by engine, qualification status, strict pass,
standardized-3D availability, QED, molecular weight, SA score, and cLogP while
retaining additional H-bond, rotatable-bond, ring, charge, warning, and SMILES
columns.

The synchronized viewer displays any filtered standardized 3D candidate and
its QED/MW/SA/status without implying a docked pose. Multi-row selection creates
an immutable `molecule_design_selection` job. It deduplicates by canonical
stereochemistry-aware SMILES, preserves engine/campaign/qualification
provenance and 3D coordinates, and publishes a typed `compound_set`.
`Docking / Cofolding` now accepts these datasets alongside compound imports;
the continuation link preselects both the design dataset and prepared target.

Design-result structural-context update (2026-07-28): the molecule-generation
Viewer now shows the immutable campaign reference ligand by default as a
distinct magenta stick/sphere overlay, rather than a thin green model that can
disappear underneath a similarly positioned generated molecule. Pocket context
remains dark grey and the generated record cyan. A separate full-receptor
control adds the staged target as a translucent light-grey overlay. The receptor
is rigidly aligned to the extracted pocket using exact chain/residue C-alpha
identities when available, and the UI reports the matched-atom count and RMSD.
For the minimized 4LNW/T3 campaign the pocket, T3, generated coordinates, and
receptor already share the same frame (30 matched C-alpha atoms, 0.000 A RMSD),
so the fit introduces no material movement.

The common protein aligner now prefers exact chain, residue number, insertion
code, and residue-name matches before sequence alignment. This is essential for
non-contiguous ligand-derived pockets; treating their sparse residue list as a
contiguous sequence can yield a wrong receptor transform. Sequence alignment
remains the fallback for predictions that legitimately renumber residues.
Context overlays are offered only where structures have a shared protein frame:
docking, refolding, interaction, pose-validation, rescoring, and generation.
Molecule-only qualification deliberately does not overlay the receptor or
reference ligand because its deterministic ETKDG conformer is rebuilt from
SMILES and has no bound-pose frame; doing so would imply a pose that has not yet
been established by docking or cofolding.

Follow-up display correction: prepared ligand SDF files may legally have a
blank molfile title line. The contextual-reference loader preserves that line;
the generic navigable-record trimming had shifted the molfile header and caused
py3Dmol to silently omit minimized T3. The reference is now drawn after the
generated model with opaque, thicker magenta sticks and translucent atom
markers. A separate `Show generated compound` switch lets the user isolate the
reference when the two structures overlap. The full receptor uses a stronger
grey opacity. Generation viewers also state pocket provenance inline: the grey
pocket is the immutable selected campaign pocket staged as `input/pocket.pdb`,
not geometry reconstructed from `pharmacophore.json`.

Task-specific ligand design update (2026-07-28): `Generate` now separates
`Ligand Redesign`, `Fragment Growing`, `Scaffold Hopping`, and `Ligand
Optimization`. These names describe distinct scientific tasks; inpainting is
the native method used by selected-region redesign rather than a second
high-level task. All pages use the bound ligand and bound-ligand coordinate
pocket belonging to the exact selected prepared complex. Pharmacophore
hypotheses remain optional downstream pose-review provenance unless the chosen
native path explicitly consumes them.

The redesign and fragment-growing pages use synchronized Mol* atom selection
and an auditable atom table. Every ligand atom is represented as one
pseudo-residue solely for transmitting exact atom selections from the existing
Mol* component. Redesign automatically derives retained atoms and boundary
anchors from molecular connectivity. Fragment growing instead interprets the
selected connected atoms as the fixed starting fragment. At native launch the
runner extracts those exact atoms, bonds, and coordinates into
`selected_fragment.sdf`; the full reference ligand is not silently passed as
the fragment.

PocketXMol fragment growing uses the official
`growing_fixed_frag.yml` MaskFill task with the extracted atoms in fixed part1,
`not_remove` set to every fragment atom, and output-size mean equal to fragment
size plus requested growth. FLOWR.root receives the same extracted fragment
with native `--fragment_growing --grow_size`. Scaffold Hopping is exposed only
for FLOWR.root because it has a dedicated native transform; the page previews
the RDKit Bemis–Murcko core and leaves native core extraction authoritative.
Ligand Optimization is exposed only for PocketXMol and uses official
`opt_mol.yml` full-molecule optimization with configurable initialization
strength. PocketXMol selected-region redesign and FLOWR.root substructure
inpainting remain on Ligand Redesign.

The two CUDA 12.8 images were refreshed with only their shared runner layer:
`ovolig-pocketxmol-cu128:latest` is
`sha256:6ced47b8213cf6054ea4cb3682dcee84b27095fce035d440de0375954678255d`
and `ovolig-flowr-root-cu128:latest` is
`sha256:2c409098b0fc52bd6a29bd875777500cba50a03cbcd9a93232cfbaf7a968f83e`.
Focused pytest coverage for page startup and immutable command handshakes
passes (4 tests). Container dry-runs against minimized 4LNW/T3 verified that
the selected eight-atom iodophenol fragment becomes `Oc1ccccc1I`, both growing
commands consume that derived SDF, scaffold hopping carries its dedicated
FLOWR.root flag, and PocketXMol optimization loads `opt_mol.yml`.
