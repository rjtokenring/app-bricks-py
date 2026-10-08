# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import subprocess
import sys
from typing import Any

import pytest
import requests
from requests.adapters import HTTPAdapter

from arduino.app_utils import HttpClient


def test_importing_app_utils_does_not_import_requests():
    code = "import sys, arduino.app_utils; print(sorted(m for m in ('requests', 'urllib3') if m in sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def _session(client: HttpClient) -> requests.Session:
    return getattr(client, "_HttpClient__http_session")


def test_the_client_retries_with_the_configured_strategy():
    client = HttpClient(total_retries=3, backoff_factor=2, status_forcelist=(503,))
    adapter = _session(client).get_adapter("https://example.com")
    assert isinstance(adapter, HTTPAdapter)
    assert adapter.max_retries.total == 3
    assert adapter.max_retries.backoff_factor == 2
    assert adapter.max_retries.status_forcelist == [503]
    client.close()


def test_a_failed_request_returns_none(monkeypatch: pytest.MonkeyPatch):
    def fail(*args: Any, **kwargs: Any) -> None:
        raise requests.exceptions.ConnectionError("down")

    client = HttpClient()
    monkeypatch.setattr(_session(client), "request", fail)
    assert client.request_with_retry("http://localhost:1") is None
    assert client.request_with_retry("") is None
