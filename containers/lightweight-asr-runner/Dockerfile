# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
FROM ${REGISTRY}app-bricks/python-base:${BASE_IMAGE_VERSION}

COPY --chown=arduino:arduino ./src/server/* /home/app/
COPY --chown=arduino:arduino ./requirements.txt /home/app/requirements.txt

RUN pip install --no-cache-dir -r /home/app/requirements.txt; \
    pip cache purge; \
    apt-get update -y; \
    apt-get install --no-install-recommends -y unzip; \
    mkdirs /home/app/models; \
    # download Vosk small English US model as default STT model
    wget -O /home/app/models/vosk-model-small-en-us-0.15.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip; \
    # unzip model
    cd /home/app/models && unzip vosk-model-small-en-us-0.15.zip && rm vosk-model-small-en-us-0.15.zip; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall /usr/local/bin; \
    python -m compileall /usr/local/lib; \
    # clean up apt cache and unnecessary files/packages
    rm /home/app/requirements.txt; \
    apt-get purge unzip -y; \
    apt-get autoclean -y; \
    apt-get autoremove -y; \
    apt-get dist-clean

USER arduino

WORKDIR /app

ENV HOME=/home/app
ENV USER=arduino

ENTRYPOINT [ "python", "/home/app/ws-api-server.py" ]
