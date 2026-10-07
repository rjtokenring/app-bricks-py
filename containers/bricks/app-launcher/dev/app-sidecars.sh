#!/bin/sh

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Start or stop the sidecars of an app for the app-launcher container, on the board, as
# arduino-app-cli would: the services of the compose file the CLI generated for the app, all but
# `main`, whose part the launcher plays. The launcher joins the app network, so the bricks reach
# the sidecars by service name as they do today.
#
#   app-sidecars.sh up detect-objects-on-camera
#   docker exec app-launcher arduino-app-launcher start detect-objects-on-camera
#   ...
#   docker exec app-launcher arduino-app-launcher stop
#   app-sidecars.sh down detect-objects-on-camera

set -eu

usage() {
  echo "usage: $0 up|down <app folder name>" >&2
  exit 2
}

[ $# -eq 2 ] || usage
action="$1"
app="$2"
apps_dir="${APPS_DIR:-/home/arduino/ArduinoApps}"
launcher="${LAUNCHER_CONTAINER:-app-launcher}"
cache="$apps_dir/$app/.cache"

[ -f "$cache/app-compose.yaml" ] || {
  echo "$cache/app-compose.yaml not found: start the app once with arduino-app-cli to generate it" >&2
  exit 1
}

set -- -f "$cache/app-compose.yaml"
if [ -f "$cache/app-compose-overrides.yaml" ]; then
  set -- "$@" -f "$cache/app-compose-overrides.yaml"
fi

project="$(sed -n 's/^name: *//p' "$cache/app-compose.yaml" | head -n 1)"
network="${project}_default"
sidecars="$(docker compose "$@" config --services | grep -vx main || true)"

case "$action" in
  up)
    if [ -z "$sidecars" ]; then
      echo "$app has no sidecars"
      exit 0
    fi
    # shellcheck disable=SC2086 # one word per service
    docker compose "$@" up -d --wait --no-deps $sidecars
    docker network connect "$network" "$launcher" 2>/dev/null || true
    ;;
  down)
    docker network disconnect "$network" "$launcher" 2>/dev/null || true
    docker compose "$@" down
    ;;
  *)
    usage
    ;;
esac
