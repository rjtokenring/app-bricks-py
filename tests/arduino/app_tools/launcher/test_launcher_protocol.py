# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import re
from pathlib import Path

import pytest

from arduino.app_tools.launcher import protocol

WORKER_PY = Path(protocol.__file__).with_name("worker.py")


def test_a_message_is_one_line():
    line = protocol.encode({"cmd": "start", "app": "a\nb"})
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    assert protocol.decode(line.rstrip(b"\n")) == {"cmd": "start", "app": "a\nb"}


@pytest.mark.parametrize("line", [b"not json", b"[1, 2]", b'"text"'])
def test_only_json_objects_are_messages(line: bytes):
    with pytest.raises(protocol.ProtocolError):
        protocol.decode(line)


def test_the_splitter_joins_chunks_and_splits_lines():
    splitter = protocol.LineSplitter()
    assert splitter.feed(b'{"a"') == []
    assert splitter.feed(b': 1}\n{"b": 2}\n{"c"') == [b'{"a": 1}', b'{"b": 2}']
    assert splitter.feed(b": 3}\n") == [b'{"c": 3}']
    assert splitter.rest() == b""


def test_the_splitter_keeps_the_unterminated_tail():
    splitter = protocol.LineSplitter()
    splitter.feed(b"first\nlast")
    assert splitter.rest() == b"last"
    assert splitter.rest() == b""


def test_the_splitter_refuses_an_endless_line():
    splitter = protocol.LineSplitter(max_line=8)
    with pytest.raises(protocol.ProtocolError):
        splitter.feed(b"0123456789")
    assert splitter.feed(b"ok\n") == [b"ok"], "it recovers on the next line"


def test_the_worker_speaks_the_same_protocol_version():
    # worker.py runs on the app interpreter and cannot import this package: it repeats the version
    match = re.search(r"^PROTOCOL_VERSION = (\d+)$", WORKER_PY.read_text(), re.MULTILINE)
    assert match is not None
    assert int(match.group(1)) == protocol.PROTOCOL_VERSION


def test_the_worker_imports_only_the_standard_library():
    import ast
    import sys

    tree = ast.parse(WORKER_PY.read_text())
    imported = {alias.name.partition(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.partition(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module and node.level == 0}
    assert imported <= set(sys.stdlib_module_names), imported - set(sys.stdlib_module_names)
