# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ${REGISTRY}app-bricks/aihub-models-runner:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

COPY . .

# requirements.txt is a hash-locked set for linux aarch64 / CPython 3.13 (see
# requirements.in): --require-hashes refuses anything not listed there, so the
# onnxruntime / onnxruntime-qnn (QAIRT) pair the models' HTP context binaries were
# compiled against cannot drift under a rebuild.
#
# The ONNX models are not tracked in git (98 MB); download_models.py fetches the
# ai-hub release zip and verifies it against the pinned hash and models/.../SHA256SUMS.
RUN python -m pip install --no-cache-dir --require-hashes -r requirements.txt \
    && python tools/download_models.py \
    && python -m compileall /opt/venv \
    && rm requirements.txt requirements.in
