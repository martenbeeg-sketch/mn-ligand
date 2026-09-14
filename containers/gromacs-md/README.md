# GROMACS MD image

`ovolig-gromacs-cu128:latest` contains GROMACS 2026.3 with CUDA 12.8,
a CPU double-precision binary for Roe–Brooks minimization, AmberTools,
ParmEd, MDTraj, MDAnalysis, DSSP, and the Python `g-mmpbsa` package.
It also contains a CPU-only GROMACS 2025.4 `grompp` utility used only to create
an endpoint TPR compatible with the GROMACS 2025 core embedded in
`g-mmpbsa` 3.0.13. Production MD remains on GROMACS 2026.3.

Build from the repository root:

```bash
docker build -t ovolig-gromacs-cu128:latest containers/gromacs-md
```

The worker mounts each immutable run directory at `/output`; published artifact
paths remain relative to that run directory.
