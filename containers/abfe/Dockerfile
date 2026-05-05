FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

LABEL org.opencontainers.image.title="ovolig-abfe"
LABEL org.opencontainers.image.description="Blackwell-oriented OpenFE ABFE container for ovo-ligand"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC \
    CONDA_DIR=/opt/conda \
    CONDA_ENV_NAME=ovolig-abfe \
    PATH=/opt/conda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgfortran5 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" && \
    rm -f /tmp/miniforge.sh

SHELL ["/bin/bash", "-lc"]

COPY environment-cu128.yml /tmp/environment.yml
RUN mamba env create -n "${CONDA_ENV_NAME}" -f /tmp/environment.yml && \
    mamba run -n "${CONDA_ENV_NAME}" python -c "import openmm; print('openmm', openmm.__version__)" && \
    mamba clean -a -y && \
    rm -f /tmp/environment.yml

RUN echo "source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV_NAME}" >> /root/.bashrc

ENV PATH=/opt/conda/envs/ovolig-abfe/bin:/opt/conda/bin:$PATH

WORKDIR /work
ENTRYPOINT []
