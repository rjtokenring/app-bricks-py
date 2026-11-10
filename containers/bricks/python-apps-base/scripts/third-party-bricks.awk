# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Select the mounted third-party bricks an app declares.
#
# Usage: awk -f third-party-bricks.awk <app.yaml> <brick_config.yaml>...
#
# Starts from the `bricks:` list of app.yaml. The `arduino:` ids are the bricks
# of the library and are dropped unchecked; every other id is looked up in the
# `id` of the brick_config.yaml files, each one at
# <name>-<version>/src/<package>/brick_config.yaml. Prints the `src` folders of
# the bricks found, joined by ':' as PYTHONPATH wants them. With no id left to
# look up, no brick_config.yaml is read.
#
# One process and no YAML library: an app start pays for this. The list is read
# in block style (`- id`, `- id: {...}`, `- id:` followed by its options) and
# as a flow list of ids on the `bricks:` line.

function scalar(text) { # a YAML scalar without quotes, comment and outer blanks
  gsub(/["\047]/, "", text)
  sub(/[ \t]+#.*$/, "", text)
  sub(/^[ \t]+/, "", text)
  sub(/[ \t]+$/, "", text)
  return text
}

function declare(entry,    id) { # <entry of the bricks list>
  id = scalar(entry)
  sub(/:([ \t].*)?$/, "", id) # the options of the brick
  if (id != "" && id !~ /^arduino:/) {
    declared[id] = 1
    wanted++
  }
}

{ sub(/\r$/, "") } # files written with CRLF line endings

FNR == 1 {
  sub(/^\357\273\277/, "") # UTF-8 byte order mark
  id_seen = 0
  if (FILENAME != ARGV[1] && !wanted) exit
}

FILENAME == ARGV[1] {
  if ($0 ~ /^bricks:/) {
    in_bricks = 1
    if (match($0, /\[.*\]/)) {
      count = split(substr($0, RSTART + 1, RLENGTH - 2), entries, ",")
      for (i = 1; i <= count; i++) declare(entries[i])
    }
  } else if ($0 ~ /^[^ \t#-]/) {
    in_bricks = 0
  } else if (in_bricks && match($0, /^[ \t]*-[ \t]*/)) {
    # Entries of the list only, not the lists nested in the options of a brick
    if (!dash) dash = index($0, "-")
    if (index($0, "-") == dash) declare(substr($0, RLENGTH + 1))
  }
  next
}

!id_seen && /^id:/ {
  id_seen = 1
  if (scalar(substr($0, 4)) in declared) {
    src = FILENAME
    sub(/\/[^\/]+\/brick_config\.yaml$/, "", src)
    if (!(src in selected)) {
      selected[src] = 1
      sources = (sources == "" ? "" : sources ":") src
    }
  }
}

END {
  if (sources != "") print sources
}
