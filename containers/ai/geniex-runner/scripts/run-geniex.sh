#!/bin/sh

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Exposes the QAIRT bundles of /models as the GenieX model store, then serves them.
# geniex replaces this shell so it gets SIGTERM directly and can release the model
# before exiting: a SIGKILL with a model loaded leaves the DSP memory mapped until reboot.
set -eu

python3 /make_store.py /models "${GENIEX_DATADIR}"

# --skip-update: no release probe at start, apps run offline
exec /opt/geniex/geniex serve --skip-update "$@"
