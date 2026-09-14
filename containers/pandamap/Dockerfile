FROM python:3.11-slim-bookworm

ARG PANDAMAP_COMMIT=000734721a3dfb22a585294cd90821f9260dab76
LABEL org.opencontainers.image.title="ovolig-pandamap" \
      org.opencontainers.image.description="PandaMap protein-ligand interaction adapter for mn-ligand" \
      org.opencontainers.image.revision="${PANDAMAP_COMMIT}" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    BLIS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    POLARS_MAX_THREADS=1 \
    RAYON_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends dssp \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/pandamap
COPY . /opt/pandamap
RUN python -m pip install --no-cache-dir . pandas gemmi rdkit
WORKDIR /workspace
ENTRYPOINT ["python", "/opt/pandamap/mn_ligand_runner.py"]
CMD ["--help"]
