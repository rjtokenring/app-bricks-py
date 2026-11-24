# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# GenieX OpenAI-compatible server with only its QAIRT (Genie) backend, on the QNN and FastRPC libraries of
# qairt-common-base. It runs as the arduino user with the FastRPC and DMA heap devices, no privileged mode.

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
ARG GENIEX_VERSION=v0.11.0
ARG GENIEX_URL=https://github.com/qualcomm/GenieX/releases/download/${GENIEX_VERSION}/geniex-cli-linux-arm64-${GENIEX_VERSION}.tar.gz
ARG GENIEX_DIGEST=sha256:b80fbc2149784a5e01d2c48d4700a459d83ec340b1f28e9351eaa4549fb3d9f8

FROM scratch AS geniex
ARG GENIEX_URL
ARG GENIEX_DIGEST
ADD --checksum=${GENIEX_DIGEST} ${GENIEX_URL} /geniex.tar.gz

FROM ${REGISTRY}app-bricks/qairt-common-base:${BASE_IMAGE_VERSION}

ARG GENIEX_VERSION

COPY ./scripts/* /

# From the release only the CLI, the SDK and the QAIRT plugin: the llama.cpp plugin and the QNN runtime the
# plugin bundles (qairt/htp-files, an older QAIRT than the base) stay out.
#
# /opt/qairt is the QNN runtime GenieX loads (GENIEX_QAIRT_LIB): a flat folder of links to the base libraries
# that also holds the skels, because GenieX points ADSP_LIBRARY_PATH at that folder. Without the skels FastRPC
# falls back to the board's DSP payload, whose libQnnHtpV75Skel.so is another QNN version, and the device
# fails to open (QNN error 14001).
RUN --mount=from=geniex,source=/geniex.tar.gz,target=/mnt/geniex.tar.gz \
    set -ex; \
    top="geniex-cli-linux-arm64-${GENIEX_VERSION}"; \
    mkdir -p /opt/geniex; \
    tar -xzf /mnt/geniex.tar.gz -C /opt/geniex --strip-components=1 \
      "${top}/geniex" "${top}/libgeniex.so" \
      "${top}/qairt/libgeniex_plugin.so" "${top}/qairt/libgeniex_core.so" "${top}/qairt/libgeniex_vlm.so" \
      "${top}/qairt/libgeniex-proc.so" "${top}/qairt/libgeniex-proc-vision.so"; \
    chmod 755 /opt/geniex/geniex; \
    mkdir -p /opt/qairt; \
    for lib in libQnnHtp.so libQnnSystem.so libQnnHtpNetRunExtensions.so libQnnHtpV73Stub.so libQnnHtpV75Stub.so; do \
      ln -s "/usr/lib/${lib}" /opt/qairt/; \
    done; \
    for skel in /usr/lib/rfsa/adsp/libQnnHtpV7*Skel.so; do ln -s "${skel}" /opt/qairt/; done; \
    for f in /opt/qairt/*; do test -e "${f}" || { echo "ERROR: dangling ${f}"; exit 1; }; done; \
    chmod +x /*.sh; \
    mkdir -p /models

EXPOSE 18181

USER arduino
ENV HOME=/home/arduino
ENV USER=arduino
ENV LD_LIBRARY_PATH=/opt/geniex
ENV GENIEX_QAIRT_LIB=/opt/qairt
# The model store GenieX reads is a view of /models rebuilt at every start (see make_store.py)
ENV GENIEX_DATADIR=/tmp/geniex

ENTRYPOINT ["/qairt-entrypoint.sh", "/run-geniex.sh"]
