FROM eclipse-temurin:17-jdk-jammy@sha256:723151f3fc88ca2060153ee08ab8dbbea7983d6ed6f2622fe440acf178737c94 AS builder

ARG P2RANK_COMMIT=d8c8e0d870f79a36b7dbb176edf8c019b6cc789c

WORKDIR /src/p2rank
COPY . .
RUN ./gradlew --no-daemon clean assemble

FROM eclipse-temurin:17-jre-jammy@sha256:475d8e96b4b2bfe08999e5e854755c773af1581acdf959a4545d88f0696a2339

ARG P2RANK_COMMIT=d8c8e0d870f79a36b7dbb176edf8c019b6cc789c

LABEL org.opencontainers.image.title="ovolig-p2rank"
LABEL org.opencontainers.image.description="Pinned CPU P2Rank pocket-detection container for mn-ligand"
LABEL org.opencontainers.image.source="https://github.com/rdk/p2rank"
LABEL org.opencontainers.image.revision="${P2RANK_COMMIT}"
LABEL org.opencontainers.image.licenses="MIT"

COPY --from=builder /src/p2rank/distro /opt/p2rank
RUN chmod +x /opt/p2rank/prank \
    && ln -s /opt/p2rank/prank /usr/local/bin/prank

WORKDIR /work
ENTRYPOINT ["/opt/p2rank/prank"]
CMD ["help"]
