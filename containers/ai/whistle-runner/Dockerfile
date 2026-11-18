# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

# Whistle speech-to-text for the arduino:asr brick: the Cactus runtime library and the model, pinned by
# digest, behind a standard-library HTTP server. Nothing is downloaded at runtime. The runtime's Python
# client (cactus-compute) is not installed: it fetches the engine and the weights on first use and sends
# usage telemetry by default.
FROM ${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION} AS cactus
ARG CACTUS_WHL_URL=https://huggingface.co/Cactus-Compute/cactus/resolve/1a028f5a3972b67dd8d8752f882113d475e7ec4c/releases/4.0.0/python/cactus_runtime-4.0.0-py3-none-manylinux2014_aarch64.whl
ARG CACTUS_WHL_DIGEST=sha256:4125d3630028409a11ba460af9561e2dc2010d777e0ff7413528f6c5dd43a67e
ARG CACTUS_LICENSE_URL=https://huggingface.co/Cactus-Compute/cactus/resolve/1a028f5a3972b67dd8d8752f882113d475e7ec4c/releases/4.0.0/LICENSE
ARG WHISTLE_URL=https://huggingface.co/Cactus-Compute/whistle/resolve/ad73f1468dd5203791e0cc81d31544c553563645/whistle.cact
ARG WHISTLE_DIGEST=sha256:b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb
ARG WHISTLE_LICENSE_URL=https://huggingface.co/Cactus-Compute/whistle/resolve/ad73f1468dd5203791e0cc81d31544c553563645/LICENSE
# Both are Apache-2.0, the same file
ARG APACHE_LICENSE_DIGEST=sha256:cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30
ADD --checksum=${CACTUS_WHL_DIGEST} ${CACTUS_WHL_URL} /tmp/cactus.whl
ADD --checksum=${APACHE_LICENSE_DIGEST} ${CACTUS_LICENSE_URL} /out/lib/LICENSE
ADD --checksum=${WHISTLE_DIGEST} ${WHISTLE_URL} /out/model/whistle.cact
ADD --checksum=${APACHE_LICENSE_DIGEST} ${WHISTLE_LICENSE_URL} /out/model/LICENSE
# The wheel holds only the shared library, which needs nothing but glibc. The modes are set here and the
# final stage copies the trees without --chmod: BuildKit applies --chmod to the parent directories it
# creates too, and a 644 directory is not traversable by the arduino user the server runs as.
RUN set -ex; \
    python -c "import sys, zipfile; open(sys.argv[2], 'wb').write(zipfile.ZipFile(sys.argv[1]).read('cactus/libcactus.so'))" /tmp/cactus.whl /out/lib/libcactus.so; \
    chmod -R u=rwX,go=rX /out

FROM ${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION}

COPY --from=cactus /out/lib/ /usr/local/lib/cactus/
COPY --from=cactus /out/model/ /models/
COPY src/whistle_server.py /app/

USER arduino

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/whistle_server.py"]
