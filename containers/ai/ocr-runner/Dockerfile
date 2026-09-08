# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ${REGISTRY}app-bricks/aihub-models-runner:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

# Code, ONNX models and the HTP context binaries compiled on the board with
# tools/compile_htp_context.py, next to them.
COPY . .

# requirements.txt is the complete, hash-locked package list for linux aarch64 /
# CPython 3.13 (see requirements.in): --no-deps --require-hashes installs exactly that and
# nothing else, so neither onnxruntime, onnxruntime-qnn nor a transitive dependency can
# drift under a rebuild, and the ~80 MB of sympy/mpmath/coloredlogs the wheels declare
# but never import stay out.
RUN python -m pip install --no-cache-dir --no-deps --require-hashes -r requirements.txt \
    && python -m compileall /opt/venv \
    && rm requirements.txt requirements.in

# The onnxruntime-qnn wheel bundles its own QAIRT (~190 MB of libQnn*.so) and that is what
# the EP uses: the base image's QAIRT (QNP_VER in qairt-common-base) is older than the one
# the EP is built against and is refused. Once QNP_VER matches the wheel's QAIRT
# (requirements.in, `# qairt-version`), switch to the system copy and drop the wheel's:
#
#   RUN find /opt/venv/lib/python3.*/site-packages/onnxruntime_qnn -name '*.so' \
#         ! -name 'libonnxruntime_providers_qnn.so' -delete
#   ENV EASYOCR_QNN_BACKEND_PATH=/usr/lib/libQnnHtp.so
#
# utils/onnx_ep.py already handles both layouts and warns when the backend library's QAIRT
# is not the one the plugin was built with.
