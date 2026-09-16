# syntax=docker/dockerfile:1

ARG BUILD_IMAGE=nvidia/cuda:13.1.2-devel-ubuntu24.04@sha256:b9f64abf7226fdb3463ca202bc99878ec847171e6c5f77bd34c8d1403fbf1eca
ARG RUNTIME_IMAGE=nvidia/cuda:13.1.2-runtime-ubuntu24.04@sha256:bff001d3257971cc4752e15ac2d354befa70995ded8e141741ade50569fc192e

FROM ${BUILD_IMAGE} AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG NINFER_REPO=https://github.com/practicorecoza/ninfer-engine.git
ARG NINFER_REF

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates cmake git libavcodec-dev libavformat-dev libavutil-dev \
        libcurl4-openssl-dev libswscale-dev ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN test -n "$NINFER_REF" \
    && git init \
    && git remote add origin "$NINFER_REPO" \
    && git fetch --depth 1 origin "$NINFER_REF" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$NINFER_REF"

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=120a \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build -j --target ninfer ninfer-serve

FROM ${RUNTIME_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG NINFER_REPO=https://github.com/practicorecoza/ninfer-engine.git
ARG NINFER_REF
ARG IMAGE_SOURCE=https://github.com/practicorecoza/ninfer

LABEL org.opencontainers.image.source="${IMAGE_SOURCE}" \
      org.opencontainers.image.revision="${NINFER_REF}" \
      org.opencontainers.image.description="NInfer sm120a quantized-KV and bounded-vision runtime" \
      org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates curl ffmpeg openssh-server python3 python3-venv tini \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/hf \
    && /opt/hf/bin/pip install --no-cache-dir --upgrade pip huggingface_hub

RUN mkdir -p /root/ninfer-5090/build-sm120a/apps /usr/share/doc/ninfer
COPY --from=build /build/apps/ninfer /root/ninfer-5090/build-sm120a/apps/ninfer
COPY --from=build /build/apps/ninfer-serve /root/ninfer-5090/build-sm120a/apps/ninfer-serve
COPY --from=build /src/LICENSE /usr/share/doc/ninfer/LICENSE
COPY docker/entrypoint /usr/local/bin/ninfer-entrypoint
COPY froggeric-chat-template.jinja /usr/share/doc/ninfer/froggeric-chat-template.jinja
RUN ln -s /root/ninfer-5090/build-sm120a/apps/ninfer /usr/local/bin/ninfer \
    && ln -s /root/ninfer-5090/build-sm120a/apps/ninfer-serve /usr/local/bin/ninfer-serve \
    && chmod 755 /usr/local/bin/ninfer-entrypoint \
    && printf 'PermitRootLogin prohibit-password\nPasswordAuthentication no\nKbdInteractiveAuthentication no\nPubkeyAuthentication yes\n' \
        > /etc/ssh/sshd_config.d/ninfer.conf \
    && printf 'source %s\nrevision %s\narchitecture sm_120a\nCUDA 13.1.2\n' \
        "$NINFER_REPO" "$NINFER_REF" > /usr/share/doc/ninfer/BUILDINFO \
    && rm -f /etc/ssh/ssh_host_*

ENV PATH="/opt/hf/bin:${PATH}"
WORKDIR /workspace
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/ninfer-entrypoint"]
