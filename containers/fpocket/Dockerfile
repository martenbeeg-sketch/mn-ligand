FROM debian:bookworm-slim AS builder

ARG FPOCKET_REPOSITORY=https://github.com/Discngine/fpocket.git
ARG FPOCKET_COMMIT=4bb0d8447f62fee77e2c3c29f54b5fcaf5e2c066

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        g++ \
        gcc \
        git \
        libnetcdf-dev \
        make \
    && rm -rf /var/lib/apt/lists/*

RUN git clone "${FPOCKET_REPOSITORY}" /src/fpocket \
    && git -C /src/fpocket checkout --detach "${FPOCKET_COMMIT}" \
    && make -C /src/fpocket

FROM debian:bookworm-slim

ARG FPOCKET_COMMIT=4bb0d8447f62fee77e2c3c29f54b5fcaf5e2c066

LABEL org.opencontainers.image.title="ovolig-fpocket"
LABEL org.opencontainers.image.description="Pinned CPU fpocket pocket-detection container for mn-ligand"
LABEL org.opencontainers.image.source="https://github.com/Discngine/fpocket"
LABEL org.opencontainers.image.revision="${FPOCKET_COMMIT}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libnetcdf19 \
        python-is-python3 \
        python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/fpocket/bin/fpocket /usr/local/bin/fpocket

WORKDIR /work
ENTRYPOINT []
CMD ["fpocket", "--help"]
