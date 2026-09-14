FROM python:3.11-slim-bookworm

ARG POSEBUSTERS_COMMIT=1a5f26aa7270fafba21b7fec8b3633f4c4e45ead

LABEL org.opencontainers.image.title="ovolig-posebusters" \
      org.opencontainers.image.description="CPU-parallel PoseBusters pose plausibility validation for mn-ligand" \
      org.opencontainers.image.source="https://github.com/maabuu/posebusters" \
      org.opencontainers.image.revision="${POSEBUSTERS_COMMIT}" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    POLARS_MAX_THREADS=1 \
    RAYON_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

WORKDIR /opt/posebusters
COPY . /opt/posebusters
RUN python -m pip install --no-cache-dir . gemmi

WORKDIR /workspace
ENTRYPOINT ["python", "/opt/posebusters/mn_ligand_runner.py"]
CMD ["--help"]
