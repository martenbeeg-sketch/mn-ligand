FROM mnprot-biohub-esm-cu128:latest

ARG ENGINE
ARG SOURCE_DIR
ARG SOURCE_COMMIT

LABEL org.opencontainers.image.title="mn-ligand molecular generation engine" \
      org.opencontainers.image.description="CUDA 12.8 / Blackwell candidate runtime for pocket and pharmacophore-conditioned design" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.source="mn-ligand local pinned source checkout"

ENV ENGINE="${ENGINE}" \
    CUDA_HOME="/usr/local/cuda" \
    TORCH_CUDA_ARCH_LIST="12.0+PTX" \
    PYTHONPATH="/opt/engine" \
    PYTHONDONTWRITEBYTECODE="1" \
    PYTHONUNBUFFERED="1"

WORKDIR /opt/engine
COPY ${SOURCE_DIR}/ /opt/engine/
COPY containers/generation/runner.py /opt/mn-ligand/generation_runner.py

# Preserve the CUDA 12.8 PyTorch stack supplied by the Blackwell base instead
# of allowing the projects' historical environment files to downgrade it.
RUN python -m pip install --no-cache-dir \
      easydict einops hydra-core lmdb networkx pandas pyyaml scipy \
      scikit-learn tqdm biopython lightning torch-geometric rdkit \
    && if [ "${ENGINE}" = "omtra" ]; then \
         python -m pip install --no-cache-dir --no-deps .; \
       fi

ENTRYPOINT ["python", "/opt/mn-ligand/generation_runner.py"]
CMD ["--help"]
