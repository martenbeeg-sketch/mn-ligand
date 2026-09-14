FROM local-plip:latest

LABEL org.opencontainers.image.title="ovolig-plip" \
      org.opencontainers.image.description="PLIP protein-ligand interaction adapter for mn-ligand" \
      org.opencontainers.image.licenses="GPL-2.0"

ENV BLIS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    POLARS_MAX_THREADS=1 \
    RAYON_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3-pandas python3-gemmi python3-rdkit \
    && rm -rf /var/lib/apt/lists/*
COPY mn_ligand_runner.py /opt/mn-ligand/mn_ligand_runner.py
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/mn-ligand/mn_ligand_runner.py"]
CMD ["--help"]
