# =============================================================
#  GNINA + UniDock-Pro + AutoDock-GPU + Vina + Meeko + OpenBabel + Jupyter
#  CUDA 12.8.1-devel on Ubuntu 22.04 with Ada and Blackwell targets
# =============================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CACHEBUST=1
ARG CUDA_ARCHITECTURES=89-real;120-real;120-virtual
ARG OPENBABEL_SOURCE_COMMIT=cbd4db43f8908b874864280fdc03bf92569eebc1
ARG GNINA_SOURCE_COMMIT=e76ee4ee59cb9932c3d0a3a97edd0ca980d10f6d
ARG UNIDOCK_SOURCE_COMMIT=a077b20e72ec00360b1e6578bbff7afc087b17c7

LABEL org.opencontainers.image.title="AVGU docking suite for mn-ligand" \
      org.opencontainers.image.description="GNINA, Uni-Dock Pro and AutoDock Vina on CUDA 12.8 for NVIDIA Blackwell"

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PATH="/opt/conda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}" \
    TZ=Europe/Rome

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
 && apt-get update && apt-get install -y --no-install-recommends \
    wget curl git ca-certificates bzip2 zip tar build-essential libgfortran5 \
    cmake swig libboost-all-dev libeigen3-dev libgoogle-glog-dev libprotobuf-dev protobuf-compiler \
    libhdf5-dev libatlas-base-dev python3-dev librdkit-dev python3-numpy python3-pip python3-pytest libjsoncpp-dev \
    software-properties-common gpg \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Micromamba installation
# ------------------------------------------------------------
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

SHELL ["/bin/bash", "-lc"]
RUN micromamba shell init -s bash -r /opt/conda

# ------------------------------------------------------------
# Base Conda environment
# ------------------------------------------------------------
RUN micromamba install -y -p /opt/conda -c conda-forge python=3.10 mamba \
 && micromamba clean -a -y

# ------------------------------------------------------------
# Conda scientific + visualization stack
# ------------------------------------------------------------
RUN micromamba install -y -p /opt/conda -c conda-forge \
      numpy scipy rdkit gemmi "meeko==0.6.1" vina autogrid cctbx-base==2024.9 \
      jupyterlab notebook ipywidgets jupytext ipykernel nb_conda_kernels \
      seaborn biopython openpyxl pymol-open-source pyarrow scikit-learn pip \
 && micromamba clean -a -y

RUN micromamba run -p /opt/conda python -m pip install --no-cache-dir py3Dmol molscrub pytest

# ------------------------------------------------------------
# ProDy installation
# ------------------------------------------------------------
WORKDIR /opt/src
RUN git clone --depth=1 https://github.com/phenix-project/geostd.git geostd
RUN git clone --depth=1 https://github.com/prody/ProDy.git prody \
 && cd prody && micromamba run -p /opt/conda python -m pip install . && cd .. \
 && rm -rf /root/.cache/pip

# ------------------------------------------------------------
# AutoDock-GPU (CUDA 12 binary)
# ------------------------------------------------------------
RUN set -euo pipefail; \
    pinned_release="1.6"; \
    url="https://github.com/ccsb-scripps/AutoDock-GPU/releases/download/v${pinned_release}/adgpu-v${pinned_release}_linux_x64_cuda12_128wi"; \
    wget -q --show-progress "${url}" -O /usr/local/bin/adgpu && chmod +x /usr/local/bin/adgpu

# ------------------------------------------------------------
# Build and install OpenBabel
# ------------------------------------------------------------
WORKDIR /opt/src
RUN git clone https://github.com/openbabel/openbabel.git \
 && cd openbabel \
 && git checkout --detach "${OPENBABEL_SOURCE_COMMIT}" \
 && git submodule update --init --recursive \
 && mkdir build && cd build \
 && cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS=-Wno-deprecated-declarations \
          -DWITH_MAEPARSER=OFF -DWITH_COORDGEN=OFF \
          -DPYTHON_BINDINGS=ON -DRUN_SWIG=ON .. \
 && make -j8 && make install && ldconfig \
 && cd /opt/src && rm -rf openbabel

# ------------------------------------------------------------
# Upgrade CMake (≥3.25 for Gnina)
# ------------------------------------------------------------
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
 && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' > /etc/apt/sources.list.d/kitware.list \
 && apt-get update && apt-get install -y --no-install-recommends cmake \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Build and install JSONCPP (modern v1.9.5)
# ------------------------------------------------------------
WORKDIR /opt/src
RUN git clone --branch 1.9.5 https://github.com/open-source-parsers/jsoncpp.git \
 && cd jsoncpp && mkdir build && cd build \
 && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON .. \
 && make -j$(nproc) && make install && ldconfig \
 && cd /opt/src && rm -rf jsoncpp

# ------------------------------------------------------------
# Build and install GNINA (CUDA enabled)
# ------------------------------------------------------------
WORKDIR /opt/src
RUN apt-get remove -y libjsoncpp-dev || true \
 && git clone https://github.com/gnina/gnina.git \
 && cd gnina \
 && git checkout --detach "${GNINA_SOURCE_COMMIT}" \
 && sed -i '/set(CMAKE_CUDA_ARCHITECTURES "all-major")/d' CMakeLists.txt \
 && mkdir build && cd build \
 && export CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
 && cmake -DUSE_CUDA=ON \
          -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
          -DUSE_SYSTEM_JSONCPP=ON \
          -DJSONCPP_INCLUDE_DIR=/usr/local/include/jsoncpp \
          -DJSONCPP_LIBRARY=/usr/local/lib/libjsoncpp.so \
          -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
          -DCMAKE_BUILD_TYPE=Release .. \
 && make -j$(nproc) && make install && ldconfig \
 && cuobjdump --list-elf /usr/local/bin/gnina | grep -q 'sm_120' \
 && cd /opt/src && rm -rf gnina

# ------------------------------------------------------------
# UniDock-Pro build (CUDA 12 compatible)
# ------------------------------------------------------------
WORKDIR /opt/unidock
RUN git clone https://github.com/NiBoyang/UniDock-Pro.git && \
    git -C UniDock-Pro checkout --detach "${UNIDOCK_SOURCE_COMMIT}"
WORKDIR /opt/unidock/UniDock-Pro
RUN cmake -B build -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" && \
    cmake --build build -j$(nproc) && \
    cuobjdump --list-elf build/udp | grep -q 'sm_120'
ENV PATH="/opt/unidock/UniDock-Pro/build:${PATH}"

# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /opt/src/*

# ------------------------------------------------------------
# Workspace & Jupyter setup
# ------------------------------------------------------------
WORKDIR /workspace
EXPOSE 8888

RUN printf '\n==============================================================\nContainer ready! ✅\n\nTo start JupyterLab inside the container, run:\n\n  jupyter lab --NotebookApp.token="" --NotebookApp.password="" \\\n              --ip=0.0.0.0 --no-browser --allow-root\n\nThen open http://localhost:8888 in your browser.\n==============================================================\n\n' \
  > /etc/motd

RUN printf 'eval "$(micromamba shell hook -s bash)"\nmicromamba activate -p /opt/conda\ncat /etc/motd\n' >> /etc/bash.bashrc

# Make the source-built Open Babel bindings visible to the Conda interpreter.
RUN printf '/usr/local/lib/python3.10/site-packages\n' \
  > /opt/conda/lib/python3.10/site-packages/openbabel-local.pth

CMD ["bash", "-l"]
