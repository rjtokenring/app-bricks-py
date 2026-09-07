# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ${REGISTRY}app-bricks/aihub-models-runner:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

# Code, ONNX models and - once compiled on the board with tools/compile_htp_context.py -
# the HTP context binaries next to them.
COPY . .

# requirements.txt is a hash-locked set for linux aarch64 / CPython 3.13 (see
# requirements.in): --require-hashes refuses anything not listed there, so the
# onnxruntime / onnxruntime-qnn (QAIRT) pair the HTP context binaries were compiled
# against cannot drift under a rebuild.
RUN python -m pip install --no-cache-dir --require-hashes -r requirements.txt \
    && python -m compileall /opt/venv \
    && rm requirements.txt requirements.in
