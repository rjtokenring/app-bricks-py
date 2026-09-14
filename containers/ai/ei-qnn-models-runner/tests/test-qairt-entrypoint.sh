#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Checks the copy of qairt-entrypoint.sh this image ships (it derives from the
# Edge Impulse QNN runtime, not from qairt-common-base, and each container is
# built with its own directory as build context, so the shared script cannot be
# COPYied from here):
#   1. it has not drifted from the canonical script, comments aside, and the
#      baked DSP yaml still maps the same machines as the other images
#   2. it builds the merged /usr/share/qcom view and execs the command
# The full behavioural suite lives with the canonical script; check 1 is what
# makes it cover this copy too.
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COPY_SCRIPT="$SCRIPT_DIR/../src/qairt-entrypoint.sh"
CANONICAL_SCRIPT="$SCRIPT_DIR/../../../base/qairt-common-base/qairt-entrypoint.sh"
COPY_YAML="$SCRIPT_DIR/../conf/hexagon-dsp-binaries.yaml"
CANONICAL_YAML="$SCRIPT_DIR/../../../base/python-base/conf/hexagon-dsp-binaries.yaml"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

FAILURES=0

fail() {
  echo "FAIL: $1"
  FAILURES=$((FAILURES + 1))
}

pass() {
  echo "ok:   $1"
}

check() { # <description> <condition...>
  local desc="$1"
  shift
  if "$@"; then pass "$desc"; else fail "$desc"; fi
}

# Strips comments, blank lines and CR (the repo is checked out with CRLF on
# Windows): what is left is the executable code, which must be identical.
code_only() {
  sed -e 's/\r$//' -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$1"
}

# The "other" permission digit must carry the read bit: the merged view has to
# stay readable by every uid whatever umask the entrypoint inherited. Only the
# bit is checked, not the exact mode: cp derives it from the baked yaml, whose
# mode depends on how the repo was checked out (a Windows bind mount reports
# 777, a Linux checkout 644).
world_readable() { # <path>
  local mode
  mode="$(stat -c %a "$1")"
  case "${mode#"${mode%?}"}" in
    [4567]) return 0 ;;
    *) echo "  ($1 is mode $mode)"; return 1 ;;
  esac
}

# Drops comments and blank lines but keeps the machine -> DSP path mapping.
yaml_body() {
  sed -e 's/\r$//' -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' "$1"
}

echo "== case 1: no drift from the canonical script and baked yaml"
check "canonical script is where it is expected" test -f "$CANONICAL_SCRIPT"
check "canonical baked yaml is where it is expected" test -f "$CANONICAL_YAML"
if diff -u <(code_only "$CANONICAL_SCRIPT") <(code_only "$COPY_SCRIPT"); then
  pass "entrypoint copy matches qairt-common-base/qairt-entrypoint.sh"
else
  fail "entrypoint copy drifted from qairt-common-base/qairt-entrypoint.sh"
fi
if diff -u <(yaml_body "$CANONICAL_YAML") <(yaml_body "$COPY_YAML"); then
  pass "baked yaml maps the same machines as python-base"
else
  fail "baked yaml drifted from python-base/conf/hexagon-dsp-binaries.yaml"
fi

echo "== case 2: merged view built from the host mount, then exec"
HOST="$WORK/host"
QCOM="$WORK/qcom"
mkdir -p "$HOST/qcs8300/Qualcomm/QCS8300-RIDE/dsp" "$HOST/conf.d"
echo bin > "$HOST/qcs8300/Qualcomm/QCS8300-RIDE/dsp/fastrpc_shell_3"
cat > "$HOST/conf.d/hexagon-dsp-binaries.yaml" <<'YAML'
machines:
  Host Machine:
    DSP_LIBRARY_PATH: host/dsp
YAML

# The script reads the baked yaml from its absolute in-image path: point it at
# the file this build context ships.
sed "s|/etc/fastrpc/hexagon-dsp-binaries.yaml|$COPY_YAML|" "$COPY_SCRIPT" > "$WORK/entrypoint.sh"
out=$(umask 077; HOST_QCOM="$HOST" QCOM="$QCOM" sh "$WORK/entrypoint.sh" echo hello-from-cmd)

check "payload dir is a symlink to the host mount" test -L "$QCOM/qcs8300"
check "payload files resolve through the symlink" test -f "$QCOM/qcs8300/Qualcomm/QCS8300-RIDE/dsp/fastrpc_shell_3"
check "host yaml copied as a regular file (not symlink)" test -f "$QCOM/conf.d/hexagon-dsp-binaries.yaml"
if [ -L "$QCOM/conf.d/hexagon-dsp-binaries.yaml" ]; then fail "host yaml must not be a symlink"; else pass "host yaml is not a symlink"; fi
check "baked fallback present" test -f "$QCOM/conf.d/00-arduino-dsp-binaries.yaml"
check "conf.d dir is world-traversable (755)" test "$(stat -c %a "$QCOM/conf.d")" = "755"
check "baked fallback is world-readable" world_readable "$QCOM/conf.d/00-arduino-dsp-binaries.yaml"
check "copied host yaml is world-readable" world_readable "$QCOM/conf.d/hexagon-dsp-binaries.yaml"
check "command after setup is exec'd with its args" test "$out" = "hello-from-cmd"
check "host mount was not written to" test ! -e "$HOST/conf.d/00-arduino-dsp-binaries.yaml"

echo "== case 3: the image entrypoint chains the wrapper before the runner"
DOCKERFILE="$SCRIPT_DIR/../Dockerfile"
entrypoint=$(sed -n -e 's/\r$//' -e 's/^ENTRYPOINT //p' "$DOCKERFILE")
check "ENTRYPOINT runs /qairt-entrypoint.sh first" \
  test "$entrypoint" = '["/qairt-entrypoint.sh", "/home/arduino/start-runner.sh"]'
check "the wrapper is installed at /qairt-entrypoint.sh" \
  grep -q '^COPY --chmod=755 \./src/qairt-entrypoint\.sh /qairt-entrypoint\.sh$' "$DOCKERFILE"
check "the baked yaml is installed where the wrapper reads it" \
  grep -q '^COPY \./conf/hexagon-dsp-binaries\.yaml /etc/fastrpc/hexagon-dsp-binaries\.yaml$' "$DOCKERFILE"
check "the merged view root is writable by the non-root user" \
  grep -q 'chown -R arduino:arduino /usr/share/qcom' "$DOCKERFILE"

echo
if [ "$FAILURES" -gt 0 ]; then
  echo "$FAILURES check(s) failed"
  exit 1
fi
echo "All checks passed"
