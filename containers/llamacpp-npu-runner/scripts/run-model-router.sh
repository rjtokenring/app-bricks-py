#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

echo "Generating models.ini..."
python3 /generate_models_ini.py /models

echo "Starting LLama server..."
export LD_LIBRARY_PATH=/opt/pkg-snapdragon/lib
# ADSP_LIBRARY_PATH must include BOTH the HTP skel libraries (libggml-htp-v*.so)
export ADSP_LIBRARY_PATH="/opt/pkg-snapdragon/lib;/lib/dsp/cdsp"

# Debug: surface Hexagon backend arch/skel URI and load failures.
# HEX_VERBOSE/arch lines are INFO/DEBUG level, so raise the log level too.
export GGML_HEXAGON_VERBOSE=1
export LLAMA_LOG_LEVEL=debug

# Build --device argument from GGML_HEXAGON_NDEV (default: 1)
NDEV="${GGML_HEXAGON_NDEV:-1}"
echo "Configuring ${NDEV} session(s)..."
DEVICE_LIST=""
for ((i=0; i<NDEV; i++)); do
  if [ -z "$DEVICE_LIST" ]; then
    DEVICE_LIST="HTP${i}"
  else
    DEVICE_LIST="${DEVICE_LIST},HTP${i}"
  fi
done

LLAMA_ARGS=(
  --device "$DEVICE_LIST"
  -ngl 100
  --no-mmap
  --models-preset /models/models.ini
  --verbose
)

if [ "${LLAMA_SERVER_SILENT}" = "1" ]; then
  LLAMA_ARGS+=(--log-disable)
fi

exec /opt/pkg-snapdragon/bin/llama-server "${LLAMA_ARGS[@]}"
