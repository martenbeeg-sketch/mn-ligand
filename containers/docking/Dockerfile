FROM mambaorg/micromamba:1.5.10

LABEL org.opencontainers.image.title="ovolig-docking"
LABEL org.opencontainers.image.description="AutoDock Vina/Open Babel docking container for ovo-ligand, built from Ligand-X docking.yml"

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean -a -y \
    && rm -f /tmp/environment.yml

WORKDIR /work
ENTRYPOINT []
