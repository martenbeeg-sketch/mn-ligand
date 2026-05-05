# ovo-ligand

OVO plugin scaffold for ligand-centric computational chemistry workflows.

This first version focuses on direct Docker container usage. Each workflow is
exposed as an OVO Streamlit page that builds and runs a `docker run` command.

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

Most generic Docker pages still use lightweight smoke-test wrappers. The
`Bound ligand MD` page is the first real workflow: it downloads a PDB, repairs
the protein by default with Ligand-X staged PDBFixer cleaning, reinserts the
bound ligands, lists bound non-water/non-ion HETATM ligands from the repaired
complex, lets you select one, then runs the Ligand-X MD optimization service in
`ovolig-md-cu128:latest`.

Supported modified amino acids are mapped before repair. Currently `CAS` is
mapped to `CYS` by keeping the CYS-compatible atoms (`N`, `CA`, `C`, `O`, `CB`,
`SG`) and dropping the arsenic substituent atoms before Ligand-X/PDBFixer repair.
This prevents `CAS` from appearing as a selectable small-molecule ligand in
structures such as `4LNW`.

The structure viewer used by this workflow is vendored in
`ovo_ligand/app/components/molstar_viewer`, including the built Mol* frontend,
so `ovo-ligand` does not depend on OVO's internal viewer package. In the bound
ligand workflow, selected protein chains are shown as blue overlays and the
selected simulation ligand is shown in red. Canvas clicks are reported back in
the page as chain/residue selections.

The bound-ligand MD workflow is self-contained in this repository. The Docker
runner mounts `ovo-ligand` itself into the MD container and uses the vendored
Ligand-X-derived modules under `ovo_ligand/ligandx`.

The MD stages come from the migrated Ligand-X workflow: protein/ligand
preparation, minimization, thermal heating, NVT, NPT, and optional production.
MM/GBSA is shown as a pending analysis step because no MM/GBSA implementation
was found in the original Ligand-X service code.

## Install with Conda

```bash
cd /home/user/programs/git-projects/ovo-ligand

conda env create -f environment.yml
conda activate ovo-ligand

pip install -e /home/user/programs/ovo/original/src
pip install -e .
```

If the environment already exists:

```bash
conda activate ovo-ligand
conda env update -f environment.yml --prune
pip install -e /home/user/programs/ovo/original/src
pip install -e .
```

## Build containers

The repository builds local Docker images, OVO-container style:

```bash
docker compose build
```

Build one image:

```bash
docker compose build docking
```

Each container has its own `containers/<tool>/environment.yml`, copied from the
corresponding Ligand-X conda environment. ABFE and RBFE currently reuse the
Ligand-X `md.yml` environment because that file contains the OpenFE, Gufe,
Cinnabar, Kartograf, OpenMM, and OpenFF stack used by those workflows.

Local image tags:

- `ovolig-structure:latest`
- `ovolig-docking:latest`
- `ovolig-md-cu128:latest`
- `ovolig-admet:latest`
- `ovoex-boltz2:latest` for Boltz2 by default
- `ovolig-qc:latest`
- `ovolig-abfe-cu128:latest`
- `ovolig-rbfe-cu128:latest`

Then start the ligand-only Streamlit app:

```bash
ovo-ligand init
ovo-ligand app
```

`ovo-ligand init` creates a dedicated local runtime directory for this project
only. It does not reuse `/home/user/programs/ovo/original/ovo-primrose`, and it
does not modify your shell startup files. `ovo-ligand app` opens only the
Ligand-X workflows, not OVO's built-in RFdiffusion, BindCraft, or Designs pages.

```text
OVO_HOME=/home/user/programs/git-projects/ovo-ligand/.ovo-home
TMPDIR=/home/user/programs/git-projects/ovo-ligand/.tmp
```

You can still pass Streamlit options through:

```bash
ovo-ligand app --server.address 127.0.0.1 --server.port 8501
```

## Run a container directly

Example:

```bash
docker run --rm \
  -v "$PWD/examples:/input:ro" \
  -v "$PWD/output:/output" \
  ovolig-docking:latest \
  /bin/bash -lc 'cp /input/example-protein.pdb /output/protein.pdb'
```

## Container Images

Default image names are local tags. Boltz2 defaults to your existing
Blackwell-ready image, `ovoex-boltz2:latest`. Override image names in the wizard
if you later publish images to GHCR.

The included [containers/boltz2/Dockerfile](containers/boltz2/Dockerfile) is an
optional recipe for rebuilding that style of CUDA 12.8 Boltz2 image.

For the OpenMM/OpenFE images, validate CUDA at runtime on the target machine:

```bash
docker run --rm --gpus all ovolig-md-cu128:latest \
  python -m openmm.testInstallation
```
