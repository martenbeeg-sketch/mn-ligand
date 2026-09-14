FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS gromacs-build

ARG GROMACS_VERSION=2026.3
ARG MMPBSA_GROMACS_VERSION=2025.4
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libfftw3-dev libhwloc-dev \
        libopenmpi-dev openmpi-bin python3-pip wget && \
    python3 -m pip install --no-cache-dir "cmake>=3.28,<4" ninja && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q "https://ftp.gromacs.org/gromacs/gromacs-${GROMACS_VERSION}.tar.gz" -O /tmp/gromacs.tar.gz && \
    tar -xzf /tmp/gromacs.tar.gz -C /tmp && \
    wget -q "https://ftp.gromacs.org/gromacs/gromacs-${MMPBSA_GROMACS_VERSION}.tar.gz" -O /tmp/gromacs-mmpbsa.tar.gz && \
    tar -xzf /tmp/gromacs-mmpbsa.tar.gz -C /tmp

RUN cmake -S "/tmp/gromacs-${GROMACS_VERSION}" -B /tmp/build-mixed -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/gromacs \
        -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF \
        -DGMX_INSTALL_NBLIB_API=OFF \
        -DGMX_GPU=CUDA -DGMX_MPI=OFF -DGMX_OPENMP=ON \
        -DGMX_BUILD_OWN_FFTW=OFF -DGMX_BUILD_UNITTESTS=OFF -DGMXAPI=OFF && \
    cmake --build /tmp/build-mixed --parallel "$(nproc)" && \
    cmake --install /tmp/build-mixed

RUN cmake -S "/tmp/gromacs-${GROMACS_VERSION}" -B /tmp/build-double -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/gromacs-double \
        -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF \
        -DGMX_INSTALL_NBLIB_API=OFF \
        -DGMX_DOUBLE=ON -DGMX_GPU=OFF -DGMX_MPI=OFF -DGMX_OPENMP=ON \
        -DGMX_BUILD_OWN_FFTW=OFF -DGMX_BUILD_UNITTESTS=OFF -DGMXAPI=OFF && \
    cmake --build /tmp/build-double --parallel "$(nproc)" && \
    cmake --install /tmp/build-double

RUN cmake -S "/tmp/gromacs-${MMPBSA_GROMACS_VERSION}" -B /tmp/build-mmpbsa -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/gromacs-mmpbsa \
        -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF \
        -DGMX_INSTALL_NBLIB_API=OFF \
        -DGMX_GPU=OFF -DGMX_MPI=OFF -DGMX_OPENMP=ON \
        -DGMX_BUILD_OWN_FFTW=OFF -DGMX_BUILD_UNITTESTS=OFF -DGMXAPI=OFF && \
    cmake --build /tmp/build-mmpbsa --parallel "$(nproc)" && \
    cmake --install /tmp/build-mmpbsa

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="ovolig-gromacs"
LABEL org.opencontainers.image.description="RTX 5090 GROMACS Roe-Brooks MD image for mn-ligand"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC \
    CONDA_DIR=/opt/conda \
    CONDA_ENV_NAME=ovolig-gromacs \
    PATH=/opt/conda/envs/ovolig-gromacs/bin:/opt/gromacs/bin:/opt/gromacs-double/bin:/opt/conda/bin:$PATH \
    GMX_BIN=/opt/gromacs/bin/gmx \
    GMX_DOUBLE_BIN=/opt/gromacs-double/bin/gmx_d \
    GMX_MMPBSA_BIN=/opt/gromacs-mmpbsa/bin/gmx \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash ca-certificates curl dssp libfftw3-double3 libfftw3-single3 \
        libgomp1 libhwloc15 libopenmpi3 wget && \
    rm -rf /var/lib/apt/lists/*

COPY --from=gromacs-build /opt/gromacs /opt/gromacs
COPY --from=gromacs-build /opt/gromacs-double /opt/gromacs-double
COPY --from=gromacs-build /opt/gromacs-mmpbsa /opt/gromacs-mmpbsa

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" && \
    rm -f /tmp/miniforge.sh

COPY environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml && \
    mamba clean -a -y && \
    rm -f /tmp/environment.yml && \
    gmx --version && \
    gmx_d --version && \
    "${GMX_MMPBSA_BIN}" --version && \
    command -v g_mmpbsa

WORKDIR /work
ENTRYPOINT []
