#!/bin/bash

HOST="0.0.0.0"

# Args: --model, --gpu-layers, --port
MODEL_ARG=""
GPU_LAYERS_ARG="16"
PORT_ARG="9000"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model=*)
      MODEL_ARG="${1#*=}"
      shift
      ;;
    --model)
      MODEL_ARG="$2"
      shift 2
      ;;
    --gpu-layers=*)
      GPU_LAYERS_ARG="${1#*=}"
      shift
      ;;
    --gpu-layers)
      GPU_LAYERS_ARG="$2"
      shift 2
      ;;
    --port=*)
      PORT_ARG="${1#*=}"
      shift
      ;;
    --port)
      PORT_ARG="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -z "$MODEL_ARG" ]]; then
  echo "Error: --model argument not provided. Exiting." >&2
  exit 1
fi

/usr/local/bin/llama-server \
  --model "/models/$MODEL_ARG" \
  --host "$HOST" \
  --port "$PORT_ARG" \
  --gpu-layers "$GPU_LAYERS_ARG"
    