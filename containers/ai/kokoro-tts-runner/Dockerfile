# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Kokoro-82M text-to-speech on ONNX Runtime, CPU only, behind a small HTTP API (src/server.py).
# The model and its voices are part of the image: the runner starts with no download.

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
# Kokoro v1.0 as exported to ONNX by the onnx-community organization of Hugging Face from
# hexgrad/Kokoro-82M (Apache-2.0), at a fixed revision: the full-precision weights and
# one file of style vectors per voice. Not model_fp16.onnx: on the CPU it generates
# nothing but NaN for some sentences.
ARG KOKORO_REPO_URL=https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/1939ad2a8e416c0acfeecc08a694d14ef25f2231
ARG KOKORO_MODEL_DIGEST=sha256:8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb

FROM ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 AS uv

FROM scratch AS model
ARG KOKORO_REPO_URL
ARG KOKORO_MODEL_DIGEST
ADD --chmod=644 --checksum=${KOKORO_MODEL_DIGEST} ${KOKORO_REPO_URL}/onnx/model.onnx /models/kokoro.onnx
# The voices of the languages espeak-ng phonemizes (src/text.py): the Japanese and Mandarin ones are left out
ADD --chmod=644 --checksum=sha256:c4a6b876047fd7fb472edf4ebd63cfac7c3b958a7cae7c106e8f038ca6308c45 ${KOKORO_REPO_URL}/voices/af_alloy.bin /models/af_alloy.bin
ADD --chmod=644 --checksum=sha256:4a004c33430762e2461eedb2013fad808ef4ab3121f5300f554476caf58d8361 ${KOKORO_REPO_URL}/voices/af_aoede.bin /models/af_aoede.bin
ADD --chmod=644 --checksum=sha256:f69d836209b78eb8c66e75e3cda491e26ea838a3674257e9d4e5703cbaf55c8b ${KOKORO_REPO_URL}/voices/af_bella.bin /models/af_bella.bin
ADD --chmod=644 --checksum=sha256:d583ccff3cdca2f7fae535cb998ac07e9fcb90f09737b9a41fa2734ec44a8f0b ${KOKORO_REPO_URL}/voices/af_heart.bin /models/af_heart.bin
ADD --chmod=644 --checksum=sha256:a240a5e3c15b43563d6e923bdca8ef5613a23471d9b77653694012435df23bd8 ${KOKORO_REPO_URL}/voices/af_jessica.bin /models/af_jessica.bin
ADD --chmod=644 --checksum=sha256:9be5221b6a941c04b561959b8ff0b06e809444dcc4ab7e75a7b23606f691819e ${KOKORO_REPO_URL}/voices/af_kore.bin /models/af_kore.bin
ADD --chmod=644 --checksum=sha256:cd2191ab31b914ed7b318416b0e4440fdf392ddad9106a060819aa600a64f59a ${KOKORO_REPO_URL}/voices/af_nicole.bin /models/af_nicole.bin
ADD --chmod=644 --checksum=sha256:18778272caa0d0eebaea251c35fd635f038434f9eee5e691d02a174bd328414f ${KOKORO_REPO_URL}/voices/af_nova.bin /models/af_nova.bin
ADD --chmod=644 --checksum=sha256:00a2bcf82b1d86e8f19902ede58c65ccf6c0e43b44b7d74fad54e5d8933c9c30 ${KOKORO_REPO_URL}/voices/af_river.bin /models/af_river.bin
ADD --chmod=644 --checksum=sha256:4409fbc125afabacc615d94db5398d847006a737b0247d6892b7a9a0007a2f0a ${KOKORO_REPO_URL}/voices/af_sarah.bin /models/af_sarah.bin
ADD --chmod=644 --checksum=sha256:4435255c9744f3f31659e0d714ab7689bf65d9e77ec1cce060f083912614f0b9 ${KOKORO_REPO_URL}/voices/af_sky.bin /models/af_sky.bin
ADD --chmod=644 --checksum=sha256:162b035ed91cfc48b6046982184c645f72edcdd1b82843347f605d7bf7b15716 ${KOKORO_REPO_URL}/voices/am_adam.bin /models/am_adam.bin
ADD --chmod=644 --checksum=sha256:3968b92c3c4cd1c4416dbded36c13eaa388a90d5788d02a13e4d781f5f8cf3c3 ${KOKORO_REPO_URL}/voices/am_echo.bin /models/am_echo.bin
ADD --chmod=644 --checksum=sha256:e8b5be17edd1e3636901ce7598baafe2dc8dd8ff707a0c23bf9e461add7e2832 ${KOKORO_REPO_URL}/voices/am_eric.bin /models/am_eric.bin
ADD --chmod=644 --checksum=sha256:c27989f741f7ee34d273a39d8a595cc0837d35f5ced9a29b7cc162614616df43 ${KOKORO_REPO_URL}/voices/am_fenrir.bin /models/am_fenrir.bin
ADD --chmod=644 --checksum=sha256:52403be32fd047c6a44517cb0bcd6b134f2a18baa73e70ef41651e0eab921ade ${KOKORO_REPO_URL}/voices/am_liam.bin /models/am_liam.bin
ADD --chmod=644 --checksum=sha256:1d1f21dd8da39c30705cd4c75d039d265e9bc4a2a93ed09bc9e1b1225eb95ba1 ${KOKORO_REPO_URL}/voices/am_michael.bin /models/am_michael.bin
ADD --chmod=644 --checksum=sha256:da5d135b424164916d75a68ffb4c2abce3d7d5ccc82dd1ee6cf447ce286145e6 ${KOKORO_REPO_URL}/voices/am_onyx.bin /models/am_onyx.bin
ADD --chmod=644 --checksum=sha256:fcf73c989033e9233e0b98713eca600c8c74dcc1614b37009d5450ff4a2274a0 ${KOKORO_REPO_URL}/voices/am_puck.bin /models/am_puck.bin
ADD --chmod=644 --checksum=sha256:61150cf726ab6c5ed7a99f90a304f91f5a72c00c592e89ec94e5df11c319227a ${KOKORO_REPO_URL}/voices/am_santa.bin /models/am_santa.bin
ADD --chmod=644 --checksum=sha256:08afa6ba24da61ea5e8efa139e5aadc938d83f0a6da5a900adaf763ac1da5573 ${KOKORO_REPO_URL}/voices/bf_alice.bin /models/bf_alice.bin
ADD --chmod=644 --checksum=sha256:669fe0647f9dd04fcab92f1439a40eeb4c8b4ab1f82e4996fe3d918ce4a63b73 ${KOKORO_REPO_URL}/voices/bf_emma.bin /models/bf_emma.bin
ADD --chmod=644 --checksum=sha256:3754352c4aaa46d17f27654ab7518d65b62ad6163a0f55a5f4330c2da2c4e94f ${KOKORO_REPO_URL}/voices/bf_isabella.bin /models/bf_isabella.bin
ADD --chmod=644 --checksum=sha256:5e0ee32ebe64a467124976b14e69590746f1c4ce41a12b587a50c862edfea335 ${KOKORO_REPO_URL}/voices/bf_lily.bin /models/bf_lily.bin
ADD --chmod=644 --checksum=sha256:6b3194bbceffb746733cbc22c8f593dd44e401a71d53895a2dca891bc595a1e8 ${KOKORO_REPO_URL}/voices/bm_daniel.bin /models/bm_daniel.bin
ADD --chmod=644 --checksum=sha256:f889083196807b4adb15e9204252165f503b8d33d3982e681c52443c49d798f1 ${KOKORO_REPO_URL}/voices/bm_fable.bin /models/bm_fable.bin
ADD --chmod=644 --checksum=sha256:c4b235a4c1f2cd3b939fed08b899ce9385638b763f7b73a59616c4fc9bd6c9bc ${KOKORO_REPO_URL}/voices/bm_george.bin /models/bm_george.bin
ADD --chmod=644 --checksum=sha256:b8f671cef828c30e66fdf0b0756a76bba58f6bb3398cbbf27058642acbcedb97 ${KOKORO_REPO_URL}/voices/bm_lewis.bin /models/bm_lewis.bin
ADD --chmod=644 --checksum=sha256:f66ec66bd295acb18372e37008533a9a3228483ccd294e7538d5d9294ac9a532 ${KOKORO_REPO_URL}/voices/ef_dora.bin /models/ef_dora.bin
ADD --chmod=644 --checksum=sha256:27809e9eafdcbcfff90a3016c697568676531de2a2c39cee29c96c7bd6b83e95 ${KOKORO_REPO_URL}/voices/em_alex.bin /models/em_alex.bin
ADD --chmod=644 --checksum=sha256:ad43b774e1ca24d05c6161297d8aeb770ac3d29bb95daf516727af5f7d543683 ${KOKORO_REPO_URL}/voices/em_santa.bin /models/em_santa.bin
ADD --chmod=644 --checksum=sha256:a35f5675ad08948e326ae75fd0ea16ba5d0042e4f76b5f3d1df77d0a48c54861 ${KOKORO_REPO_URL}/voices/ff_siwis.bin /models/ff_siwis.bin
ADD --chmod=644 --checksum=sha256:040be6a4425411cc01fda5fd06693c76bfa78572632852bc8cda9c99232ffb56 ${KOKORO_REPO_URL}/voices/hf_alpha.bin /models/hf_alpha.bin
ADD --chmod=644 --checksum=sha256:cd83ae0bb9b2e4e4fb92b4973bd8d1822ca0036d3c498bf4fc89aa8e33917cc7 ${KOKORO_REPO_URL}/voices/hf_beta.bin /models/hf_beta.bin
ADD --chmod=644 --checksum=sha256:b02d9222d9ed00ce26b302173a862c2c93f96cc40b5c422b8d14910b9ff34137 ${KOKORO_REPO_URL}/voices/hm_omega.bin /models/hm_omega.bin
ADD --chmod=644 --checksum=sha256:644daf88ba8aeb7bd08950bbdcd4453bb280864e49dc4df93fabc6be32e03f37 ${KOKORO_REPO_URL}/voices/hm_psi.bin /models/hm_psi.bin
ADD --chmod=644 --checksum=sha256:409b69248798fcdc2542330c76953d230710f19b057e59cb82fdc3c4cf71265c ${KOKORO_REPO_URL}/voices/if_sara.bin /models/if_sara.bin
ADD --chmod=644 --checksum=sha256:bc578e510d52a96d6940d46f12e96d7b3df00905dbea075113226d100e6e1ab0 ${KOKORO_REPO_URL}/voices/im_nicola.bin /models/im_nicola.bin
ADD --chmod=644 --checksum=sha256:3da7b5b2d91847ebf5646f57631af6ececae3c29a89cd300f06edf9aa6cfe9ee ${KOKORO_REPO_URL}/voices/pf_dora.bin /models/pf_dora.bin
ADD --chmod=644 --checksum=sha256:0175c753f59c54e7fd5a995bedef0c5ff2fb67e0043dd3dcb2ae74ec2acbeb2a ${KOKORO_REPO_URL}/voices/pm_alex.bin /models/pm_alex.bin
ADD --chmod=644 --checksum=sha256:8b012db3185778afe2e45a62cbad69db73021774fe68dda634bcc748a982eede ${KOKORO_REPO_URL}/voices/pm_santa.bin /models/pm_santa.bin

FROM ${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

COPY ./pyproject.toml ./uv.lock /tmp/deps/

# uv is only mounted while installing, it is not part of the image
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    set -ex; \
    apt-get update -y; \
    # espeak-ng turns text into the phonemes the model reads. It is a GPL-3.0 program the
    # server only runs as a child process, text on stdin and phonemes on stdout.
    apt-get install -y --no-install-recommends --no-install-suggests espeak-ng; \
    apt-get dist-clean; \
    # --no-deps: the export is the whole locked closure, and without it uv would resolve the
    # dependencies the wheels declare again, the ones overridden out of the lock included
    uv export --frozen --project /tmp/deps | uv pip install --system --no-cache-dir --no-deps --require-hashes -r -; \
    # cleanup unnecessary files to reduce image size
    rm -rf /tmp/* /var/tmp/*; \
    rm -rf /var/lib/apt/lists/*; \
    rm -fr /var/cache/apt/archives/*.deb \
        /var/cache/apt/archives/partial/*.deb \
        /usr/share/man/* \
        /usr/share/locale/* \
        /var/cache/apt/*.bin; \
    find /usr/share/doc -type f ! -name 'copyright' -delete; \
    find /usr/share/doc -type d -empty -delete; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall -q /usr/local/lib

COPY --link --from=model /models/ /models/

COPY ./src /app
RUN python -m compileall -q /app

# Text of the license the model and its voices are distributed under
COPY --from=licenses Apache-2.0.txt /app/LICENSES/

USER arduino

WORKDIR /app

ENV HOME=/home/arduino
ENV USER=arduino
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 9100

ENTRYPOINT ["python", "/app/server.py"]
