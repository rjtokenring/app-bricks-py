#!/bin/sh

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Entrypoint of the app-launcher container: what run.sh does once per container,
# then the supervisor, which starts the apps (see arduino.app_tools.launcher).

# Disable core dumps: inherited by the supervisor, every worker and every app.
# Set ENABLE_CORE_DUMPS=1 to keep them (e.g. to debug a native crash).
if [ "${ENABLE_CORE_DUMPS:-0}" != "1" ]; then
  ulimit -c 0 2>/dev/null || true
fi

if [ -z "$PYTHONUNBUFFERED" ]; then
  export PYTHONUNBUFFERED=1
fi

# No app runs yet: /app points at the empty folder until the first start
ln -sfn /home/app/.launcher/none /home/app/.launcher/current

# Device provisioning, per container as with run.sh, but once for all the apps:
# the launcher prepares each app with SKIP_DEVICE_PROVISIONING=1
bash /provision-alsa-devices.sh
sh /provision-fastrpc-dsp.sh || echo "Warning: fastrpc DSP provisioning failed"

case "$1" in
  provision|prepare)
    # The run.sh modes stay available, on the app APP_DIR names
    exec /run.sh "$@"
    ;;
esac

exec arduino-app-launcher serve
