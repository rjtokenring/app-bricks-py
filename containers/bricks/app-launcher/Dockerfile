# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ${REGISTRY}app-bricks/python-apps-base:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

USER root

COPY ./scripts/launcher.sh /launcher.sh

# One container serves every app in /home/arduino/ArduinoApps. Each app still sees itself at /app:
# /app is a symlink to /home/app/.launcher/current, which arduino-app-launcher points at the running
# app, or at the empty .launcher/none folder when none runs. The second link lives in a folder the
# arduino user owns, the first one never changes.
RUN set -ex; \
    chmod 755 /launcher.sh; \
    rm -rf /app; \
    mkdir -p /home/app/.launcher/none /run/arduino-app-launcher; \
    ln -sfn /home/app/.launcher/none /home/app/.launcher/current; \
    ln -s /home/app/.launcher/current /app; \
    chown -R arduino:arduino /home/app/.launcher /run/arduino-app-launcher

USER arduino

# Not /app: it is a symlink that may point at an app folder removed meanwhile
WORKDIR /home/app

ENTRYPOINT [ "/launcher.sh" ]
