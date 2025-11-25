#!/bin/bash

NODE_COMMAND=("node" "/app/linux/node/build/cli/linux/runner.js" "$@")

while true; do
  echo "🚀 Starting runner..."

  "${NODE_COMMAND[@]}"

  EXIT_CODE=$?

  # Check the exit code
  if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Application exited successfully (Exit Code: 0). Stopping restart loop."
    break
  else
    echo "⚠️ Application exited with error (Exit Code: $EXIT_CODE). Restarting in 1 seconds..."
    sleep 1
  fi
done
