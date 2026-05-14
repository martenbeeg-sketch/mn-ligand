# mn-ligand

Standalone Streamlit app for ligand-centric computational chemistry workflows.

The app runs local Docker containers directly and stores all runtime state in a
project-local directory by default. It does not require the external platform to
be installed.

## Workflows

- Structure preparation
- Molecular docking
- Batch docking
- Bound ligand MD
- Molecular dynamics
- ADMET prediction
- Boltz-2 prediction
- Quantum chemistry
- ABFE
- RBFE

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
```

Local image tags:

- `ovolig-structure:latest`
- `ovolig-docking:latest`
- `ovolig-md-cu128:latest`
- `ovolig-admet:latest`
- `ovoex-boltz2:latest` for Boltz2 by default
- `ovolig-qc:latest`
- `ovolig-abfe-cu128:latest`
- `ovolig-rbfe-cu128:latest`

## Run The App

```bash
mn-ligand app
```

The command creates runtime folders automatically. `mn-ligand init` is optional
and only pre-creates those folders.

Default runtime paths:

```text
app home: ./mn-ligand-workdir
runs:     ./mn-ligand-workdir/workdir/runs
tmp:      ./.tmp
```

You can override the runtime location:

```bash
mn-ligand app --app-home /path/to/mn-ligand-runtime
```

You can pass Streamlit options through:

```bash
mn-ligand app --server.address 127.0.0.1 --server.port 8501
```

## Boltz2 Models

Boltz2 defaults to these local model/reference paths:

```text
/mnt/db/reference_files/boltz_models
/mnt/db/reference_files/boltz_models/msa_repository
```

If another machine uses different paths, open the hidden Boltz2 settings in the
app and change the model/cache and MSA repository directories.

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
