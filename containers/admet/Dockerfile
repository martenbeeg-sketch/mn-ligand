FROM mambaorg/micromamba:1.5.10

LABEL org.opencontainers.image.title="ovolig-admet"
LABEL org.opencontainers.image.description="ADMET prediction container for mn-ligand, built from Ligand-X admet.yml"

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean -a -y \
    && rm -f /tmp/environment.yml

WORKDIR /work
ENTRYPOINT []
