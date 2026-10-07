#!/bin/sh

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Baseline for `arduino-app-launcher bench`: start an app with arduino-app-cli, the way it is
# started today, and time it until its web UI answers. Stop the app-launcher container first,
# both publish port 7000.
#
#   bench-classic.sh debug-start-time 5 http://127.0.0.1:7000/

set -eu

app="${1:?usage: $0 <app folder name> [runs] [url]}"
runs="${2:-3}"
url="${3:-http://127.0.0.1:7000/}"
path="${APPS_DIR:-/home/arduino/ArduinoApps}/$app"

now() { date +%s.%N; }
ms() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.0f", (b - a) * 1000 }'; }

arduino-app-cli app stop "$path" >/dev/null 2>&1 || true
i=1
while [ "$i" -le "$runs" ]; do
  t0="$(now)"
  arduino-app-cli app start "$path" >/dev/null 2>&1 &
  until curl -s -o /dev/null --max-time 1 "$url"; do sleep 0.02; done
  t1="$(now)"
  wait
  t2="$(now)"
  arduino-app-cli app stop "$path" >/dev/null 2>&1
  t3="$(now)"
  echo "run $i: to_http_ms=$(ms "$t0" "$t1") cli_start_returned_ms=$(ms "$t0" "$t2") stop_ms=$(ms "$t2" "$t3")"
  i=$((i + 1))
done
