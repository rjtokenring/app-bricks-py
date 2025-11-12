# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
FROM ${REGISTRY}app-bricks/python-base:${BASE_IMAGE_VERSION}

COPY ./requirements.txt /app/requirements.txt
COPY ./src/python/piper-server.py /app/piper-server.py

RUN pip install --no-cache-dir -r /app/requirements.txt; \
    rm /app/requirements.txt; \
    mkdir /app/models; \
    wget -O /app/models/en_US-lessac-low.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx?download=true; \
    wget -O /app/models/en_US-lessac-low.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx.json?download=true; \
    chown arduino:arduino -R /app; \
    chmod 755 -R /app; \
    # ensure permissions for /run.sh
    chmod 755 /run.sh; \
    # trim dependencies
    find /usr/local/lib/python3.13/site-packages -type d -name "tests" -exec rm -r {} +; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall /usr/local/bin; \
    python -m compileall /usr/local/lib

USER arduino

WORKDIR /app

ENV HOME=/app
ENV USER=arduino

ENTRYPOINT [ "python", "/app/piper-server.py" ]
