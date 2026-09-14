# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

FROM public.ecr.aws/g7a8t7v6/inference-container-qnn:v1.95.11

ARG DEBIAN_FRONTEND=noninteractive

# Create the user and group needed to run the container as non-root
RUN set -ex; \
    groupadd -g 1000 arduino || true; \
    useradd -u 1000 -g arduino -ms /bin/bash arduino

# Add all EI OOTB models to the container with specific ownership
COPY --chown=arduino:arduino ./models/ei-ootb-models/arm64/* /models/ootb/ei/
COPY --chown=arduino:arduino ./src/start-runner.sh /home/arduino/start-runner.sh

# Entrypoint wrapper: builds a merged /usr/share/qcom view (host DSP payload
# symlinked from the read-only /run/host-qcom mount, conf.d populated with host
# yamls plus the baked default) without ever writing to the host. Same script as
# the one qairt-common-base ships for the images that derive from it.
COPY --chmod=755 ./src/qairt-entrypoint.sh /qairt-entrypoint.sh

# Baked fastrpc DSP path config for all supported devices. /qairt-entrypoint.sh
# copies it to /usr/share/qcom/conf.d/00-arduino-dsp-binaries.yaml at runtime:
# the fastrpc parser keeps the last alphabetical match, so this file sorts first
# and acts as a fallback that any yaml provided by the host overrides.
COPY ./conf/hexagon-dsp-binaries.yaml /etc/fastrpc/hexagon-dsp-binaries.yaml

RUN set -ex; \
    apt-get update && apt-get install -y --no-install-recommends python3-pip && apt-get dist-clean && rm -rf /var/lib/apt/lists/* ; \
    pip3 install --no-cache-dir --break-system-packages requests;\
    chmod +x /home/arduino/start-runner.sh; \
    # Pre-create the merged view root, owned by arduino: the entrypoint runs as
    # the non-root user and has to populate it at startup.
    mkdir -p /usr/share/qcom/conf.d; \
    chown -R arduino:arduino /usr/share/qcom

WORKDIR /home/arduino
ENV HOME=/home/arduino

USER arduino
ENV USER=arduino

ENTRYPOINT ["/qairt-entrypoint.sh", "/home/arduino/start-runner.sh"]
