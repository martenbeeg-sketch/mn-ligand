FROM mambaorg/micromamba:1.5.10

LABEL org.opencontainers.image.title="ovolig-qc"
LABEL org.opencontainers.image.description="Quantum chemistry helper container for ovo-ligand, built from Ligand-X qc.yml; mount ORCA separately"

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean -a -y \
    && rm -f /tmp/environment.yml

ENV ORCA_PATH=/opt/orca
WORKDIR /work
ENTRYPOINT []
