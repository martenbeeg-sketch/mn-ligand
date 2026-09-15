# Copyright 2024 DeepMind Technologies Limited
# AlphaFold 3 source is licensed under CC BY-NC-SA 4.0. Model parameters are
# distributed separately and remain subject to their own terms of use.
#
# Reconstructed from the Dockerfile and source revision used to build the local
# alphafast:latest image, then updated to CUDA 12.8 for native Blackwell support.
# The small application-specific MMseqs patch is kept in this build context so a
# fresh clone can reproduce the installed behavior.

# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-base-ubuntu24.04

LABEL org.opencontainers.image.title="AlphaFast for mn-ligand" \
      org.opencontainers.image.description="AlphaFold 3 and MMseqs2-GPU runtime on CUDA 12.8 for NVIDIA Blackwell"

ARG ALPHAFAST_REPOSITORY=https://github.com/RomeroLab/alphafast.git
ARG ALPHAFAST_SOURCE_COMMIT=2cd8c52d9fe520c4cd3ac8f3bce95528603b38d0

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update --quiet \
    && apt-get install --yes --quiet python3.12 python3.12-dev \
    && apt-get install --yes --quiet git wget gcc g++ make zlib1g-dev zstd pigz \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=0 \
    UV_PROJECT_ENVIRONMENT=/alphafold3_venv \
    PATH="/hmmer/bin:/alphafold3_venv/bin:${PATH}"

RUN uv venv "${UV_PROJECT_ENVIRONMENT}"

RUN mkdir -p /hmmer_build /hmmer && \
    wget -q http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz -P /hmmer_build && \
    (cd /hmmer_build && echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 hmmer-3.4.tar.gz" | sha256sum --check) && \
    (cd /hmmer_build && tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz) && \
    (cd /hmmer_build/hmmer-3.4 && ./configure --prefix /hmmer) && \
    (cd /hmmer_build/hmmer-3.4 && make -j"$(nproc)") && \
    (cd /hmmer_build/hmmer-3.4 && make install) && \
    (cd /hmmer_build/hmmer-3.4/easel && make install) && \
    rm -rf /hmmer_build

RUN wget -q -O mmseqs.tar.gz https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz & \
    wget -q -O foldseek.tar.gz https://mmseqs.com/foldseek/foldseek-linux-gpu.tar.gz & \
    wait && \
    pigz -dc mmseqs.tar.gz | tar xf - && \
    pigz -dc foldseek.tar.gz | tar xf - && \
    cp mmseqs/bin/mmseqs foldseek/bin/foldseek /usr/local/bin/ && \
    rm -rf mmseqs foldseek *.tar.gz

WORKDIR /app/alphafold

RUN git clone "${ALPHAFAST_REPOSITORY}" . && \
    git checkout --detach "${ALPHAFAST_SOURCE_COMMIT}" && \
    rm -rf .git

COPY mmseqs-split-memory.patch /tmp/mmseqs-split-memory.patch
RUN git apply /tmp/mmseqs-split-memory.patch && \
    rm /tmp/mmseqs-split-memory.patch

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --all-groups

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-groups --no-editable

RUN uv run build_data

ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
    XLA_CLIENT_MEM_FRACTION=0.95

CMD ["uv", "run", "python3", "run_alphafold.py"]
