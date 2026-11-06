# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

# Whistle speech-to-text, the arduino:asr default on UNO Q: the Cactus runtime library and the model,
# pinned by digest. The brick loads both in the app process through ctypes and nothing is downloaded at
# runtime. The runtime's Python client (cactus-compute) is not installed: it fetches the engine and the
# weights on first use and sends usage telemetry by default.
FROM ${REGISTRY}app-bricks/python-base:${BASE_IMAGE_VERSION} AS cactus
ARG CACTUS_WHL_URL=https://huggingface.co/Cactus-Compute/cactus/resolve/1a028f5a3972b67dd8d8752f882113d475e7ec4c/releases/4.0.0/python/cactus_runtime-4.0.0-py3-none-manylinux2014_aarch64.whl
ARG CACTUS_WHL_DIGEST=sha256:4125d3630028409a11ba460af9561e2dc2010d777e0ff7413528f6c5dd43a67e
ADD --checksum=${CACTUS_WHL_DIGEST} ${CACTUS_WHL_URL} /tmp/cactus.whl
# The wheel holds only the shared library, which needs nothing but glibc
RUN python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extract('cactus/libcactus.so', sys.argv[2])" /tmp/cactus.whl /out

FROM ${REGISTRY}app-bricks/python-base:${BASE_IMAGE_VERSION}

ARG CACTUS_LICENSE_URL=https://huggingface.co/Cactus-Compute/cactus/resolve/1a028f5a3972b67dd8d8752f882113d475e7ec4c/releases/4.0.0/LICENSE
ARG WHISTLE_URL=https://huggingface.co/Cactus-Compute/whistle/resolve/ad73f1468dd5203791e0cc81d31544c553563645/whistle.cact
ARG WHISTLE_DIGEST=sha256:b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb
ARG WHISTLE_LICENSE_URL=https://huggingface.co/Cactus-Compute/whistle/resolve/ad73f1468dd5203791e0cc81d31544c553563645/LICENSE
# Both are Apache-2.0, the same file
ARG APACHE_LICENSE_DIGEST=sha256:cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30
COPY --from=cactus --chmod=644 /out/cactus/libcactus.so /usr/local/lib/cactus/libcactus.so
ADD --chmod=644 --checksum=${APACHE_LICENSE_DIGEST} ${CACTUS_LICENSE_URL} /usr/local/lib/cactus/LICENSE
ADD --chmod=644 --checksum=${WHISTLE_DIGEST} ${WHISTLE_URL} /models/ootb/cactus/whistle.cact
ADD --chmod=644 --checksum=${APACHE_LICENSE_DIGEST} ${WHISTLE_LICENSE_URL} /models/ootb/cactus/LICENSE

# The compose files provisioned at runtime reference the images published with this one
ARG BASE_IMAGE_VERSION
ENV BRICKS_RELEASE_VERSION=${BASE_IMAGE_VERSION}

# Copy the Streamlit configuration file with customized settings in home app shared directory
COPY ./streamlit/config.toml /home/app/.streamlit/config.toml

COPY ./scripts/*.sh /
# The wheel and the lock it was built from, from the bake context filled by `task build:bricks`
COPY --from=wheel arduino*.whl pyproject.toml uv.lock /app/
COPY ./patches/ /tmp/patches/
# OpenCV is left out, python-base ships its own GStreamer build of it
RUN set -ex; \
    uv export --frozen --project /app --extra all --no-dev --no-emit-project --no-emit-package opencv-python-headless | uv pip install --system --no-cache-dir --require-hashes -r -; \
    uv pip install --system --no-cache-dir --no-deps /app/*.whl; \
    rm /app/*.whl /app/pyproject.toml /app/uv.lock; \
    # Every patch in patches/ (see its README), in name order: one that no longer applies exactly fails the build.
    # Applied before compileall, which compiles the patched sources. GNU patch comes in through dpkg: apt-get
    # install refuses to run on the packages python-base removes with --force-depends, dpkg checks only the
    # dependencies of patch
    set -- /tmp/patches/*.patch; \
    if [ -e "$1" ]; then \
        apt-get update -y; \
        (cd /tmp && apt-get download patch); \
        dpkg -i /tmp/patch_*.deb; \
        for p in "$@"; do patch -p1 --forward --fuzz=0 --no-backup-if-mismatch -d /usr/local/lib/python3.13/site-packages -i "$p"; done; \
        dpkg --purge patch; \
        apt-get dist-clean; \
        rm -rf /var/lib/apt/lists/* /var/lib/dpkg/*-old /tmp/patch_*.deb; \
    fi; \
    rm -rf /tmp/patches; \
    # ALSA dsnoop/dmix wrappers: the file is static, generate it once here instead of at every app start
    HOME=/home/app bash /provision-alsa-devices.sh; \    
    # fix permissions for /home/app
    chown arduino:arduino -R /home/app; \
    chmod 777 -R /home/app; \
    # ensure permissions for /run.sh
    chmod 755 /*.sh; \
    # trim dependencies
    find /usr/local/lib/python3.13/site-packages -type d -name "tests" -exec rm -r {} +; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall /usr/local/bin; \
    python -m compileall /usr/local/lib

USER arduino

WORKDIR /app

ENV HOME=/home/app
ENV USER=arduino
ENV PIPEWIRE_REMOTE=/run/user/1000/pipewire-0

ENTRYPOINT [ "/run.sh" ]
