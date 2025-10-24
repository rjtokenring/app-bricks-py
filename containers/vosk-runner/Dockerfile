# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
FROM ${REGISTRY}app-bricks/python-base:${BASE_IMAGE_VERSION}

COPY --chown=arduino:arduino ./ws-api-server.py /home/app/ws-api-server.py
COPY --chown=arduino:arduino ./requirements.txt /home/app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt; \
    pip cache purge; \
    apt-get update -y; \
    apt-get install --no-install-recommends -y unzip; \
    apt-get autoclean -y; \
    apt-get autoremove -y; \
    apt-get dist-clean; \
    mkdirs /home/app/models; \
    wget -O /home/app/models/vosk-model-small-en-us-0.15.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip; \
    cd /home/app/models && unzip vosk-model-small-en-us-0.15.zip && rm vosk-model-small-en-us-0.15.zip; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall /usr/local/bin; \
    python -m compileall /usr/local/lib

USER arduino

WORKDIR /app

ENV HOME=/home/app
ENV USER=arduino

ENTRYPOINT [ "python", "/home/app/ws-api-server.py" ]
