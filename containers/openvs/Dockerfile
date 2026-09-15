# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:$PATH \
    CONDA_ENV_NAME=openvs \
    ROSETTAHOME=/rosetta \
    DIMORPHITE=/dimorphite_dl

# Use bash for RUN so "conda run" and shell logic behave consistently
SHELL ["/bin/bash", "-lc"]

# ------------------------------------------------------------------
# System packages
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    mc \
    wget \
    git \
    curl \
    bash \
    build-essential \
    clang \
    clang-tidy \
    clang-format \
    ca-certificates \
    doxygen \
    graphviz \
    libcudnn8 \
    libcudnn8-dev \
    libgfortran5 \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev \
    libffi-dev \
    python3 \
    python3-dev \
    python3-numpy \
    python3-setuptools \
    python3-distutils \
    python3-openssl \
    python3-venv \
    python-dev-is-python3 \
    mpich \
    mpi-default-dev && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# Install Miniforge (conda-forge based)
# ------------------------------------------------------------------
RUN wget -O /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm -f /tmp/miniforge.sh

# Optional but useful
RUN conda init bash

# ------------------------------------------------------------------
# Conda configuration
# Keep this BEFORE creating environments
# ------------------------------------------------------------------
RUN cat > /root/.condarc <<'EOF'
channels:
  - conda-forge
  - https://conda.graylab.jhu.edu
channel_priority: strict
EOF

# ------------------------------------------------------------------
# Create conda environments
# Put mamba in both base and openvs env for convenience
# ------------------------------------------------------------------
RUN conda install -n base -y mamba && \
    conda create -n ${CONDA_ENV_NAME} -y python=3.9 mamba pip && \
    conda create -n csd -y python=3.9 mamba pip

# ------------------------------------------------------------------
# Build Rosetta from source
# ------------------------------------------------------------------
RUN git clone https://github.com/RosettaCommons/rosetta.git /rosetta

WORKDIR /rosetta/source

# Adjust -j4 if you want more parallelism
RUN ./scons.py -j4 mode=release bin

# ------------------------------------------------------------------
# Clone extra repositories
# ------------------------------------------------------------------
RUN git clone https://github.com/durrantlab/dimorphite_dl.git /dimorphite_dl && \
    git clone https://github.com/gfzhou/OpenVS.git /OpenVS

# ------------------------------------------------------------------
# Install OpenVS dependencies
# ------------------------------------------------------------------
WORKDIR /OpenVS

# Install conda dependencies from requirements.txt into openvs env
RUN conda run -n ${CONDA_ENV_NAME} mamba install -y --file requirements.txt

# Install OpenVS in editable mode
RUN conda run -n ${CONDA_ENV_NAME} pip install -e .

# ------------------------------------------------------------------
# Install CSD Python API in separate environment
# ------------------------------------------------------------------
RUN conda run -n csd mamba install -y \
    --channel=https://conda.ccdc.cam.ac.uk \
    csd-python-api

# ------------------------------------------------------------------
# Auto-activate openvs for interactive shells
# ------------------------------------------------------------------
RUN echo "conda activate ${CONDA_ENV_NAME}" >> /root/.bashrc

# ------------------------------------------------------------------
# Final working directory
# ------------------------------------------------------------------
RUN mkdir -p /workdir
WORKDIR /workdir

CMD ["bash"]
