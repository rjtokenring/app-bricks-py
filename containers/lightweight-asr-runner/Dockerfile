# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# Manual build args:
#    docker buildx build --build-arg REGISTRY=ghcr.io/arduino/ --platform linux/arm64 -t lightweight-asr-runner:latest .
#    docker save -o lightweight-asr-runner.tar lightweight-asr-runner:latest
#    docker load -i lightweight-asr-runner.tar

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
FROM ${REGISTRY}app-bricks/python-base:${BASE_IMAGE_VERSION}

COPY --chown=arduino:arduino ./src/server/* /home/arduino/
COPY --chown=arduino:arduino ./requirements.txt /home/arduino/requirements.txt

RUN pip install --no-cache-dir -r /home/arduino/requirements.txt; \
    pip cache purge; \
    apt-get update -y; \
    apt-get install --no-install-recommends -y unzip; \
    mkdir -p /home/arduino/models; \
    # download Vosk small English US model as default STT model
    wget -O /home/arduino/models/vosk-model-small-en-us-0.15.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip; \
    # unzip model
    cd /home/arduino/models && unzip vosk-model-small-en-us-0.15.zip && rm vosk-model-small-en-us-0.15.zip; \
    chown -R arduino:arduino /home/arduino/*; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall /usr/local/bin; \
    python -m compileall /usr/local/lib; \
    # clean up apt cache and unnecessary files/packages
    rm /home/arduino/requirements.txt; \
    apt-get purge unzip -y; \
    apt-get autoclean -y; \
    apt-get autoremove -y; \
    apt-get dist-clean

USER arduino

WORKDIR /home/arduino

ENV HOME=/home/arduino
ENV USER=arduino

ENTRYPOINT [ "python", "/home/arduino/ws-api-server.py" ]
