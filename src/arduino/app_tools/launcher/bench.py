# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Start-time measurements of an app through the launcher, warm and immediate."""

import queue
import statistics
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from .client import Client
from .protocol import Message


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _wait_http(url: str, deadline: float) -> float | None:
    """Time the URL first answers, with any HTTP status, or None at the deadline."""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0):
                return time.time()
        except urllib.error.HTTPError:
            return time.time()
        except (urllib.error.URLError, OSError):
            time.sleep(0.02)
    return None


def _wait_ready(client: Client, app: str, deadline: float) -> bool:
    while time.time() < deadline:
        for entry in client.call("status").get("apps", []):
            if entry.get("name") == app and (entry.get("worker") or {}).get("state") == "ready":
                return True
        time.sleep(0.2)
    return False


def run_bench(client: Client, app: str, runs: int, mode: str, http_url: str | None, settle_s: float, timeout_s: float = 120.0) -> Message:
    """Start and stop the app `runs` times; time each start from the request to App.run(), and to the first HTTP answer.

    In "warm" mode each run waits for the app worker to be ready first; "immediate" starts a worker without warm-up.
    """
    events: queue.Queue[Message] = queue.Queue()

    def follow() -> None:
        for event in client.stream("events"):
            events.put(event)

    threading.Thread(target=follow, daemon=True, name="bench-events").start()
    time.sleep(0.2)
    client.call("stop")
    samples: list[dict[str, Any]] = []
    for index in range(runs):
        if mode == "warm" and not _wait_ready(client, app, time.time() + timeout_s):
            raise TimeoutError(f"the worker of {app} did not become ready")
        while not events.empty():
            events.get_nowait()
        t0 = time.time()
        reply = client.call("start", app=app, mode="immediate" if mode == "immediate" else "auto")
        if not reply.get("ok"):
            raise RuntimeError(f"start failed: {reply}")
        sample: dict[str, Any] = {"run": index + 1, "path": reply.get("path"), "was_ready": reply["worker"]["was_ready"]}
        t = reply["t"]
        sample["to_started_ms"] = (t["started"] - t0) * 1000
        if http_url:
            answered = _wait_http(http_url, time.time() + timeout_s)
            if answered is not None:
                sample["to_http_ms"] = (answered - t0) * 1000
        # Events carry their own time: reading them after the HTTP wait loses nothing
        deadline = time.time() + (5.0 if http_url else timeout_s)
        while time.time() < deadline:
            try:
                event = events.get(timeout=max(0.0, deadline - time.time()))
            except queue.Empty:
                break
            if event.get("event") == "app_run" and event.get("run_id") == reply["run_id"]:
                sample["to_app_run_ms"] = (event["t"] - t0) * 1000
                break
            if event.get("event") == "app_exited" and event.get("run_id") == reply["run_id"]:
                break
        samples.append(sample)
        time.sleep(settle_s)
        stop = client.call("stop")
        sample["stop_ms"] = (stop.get("stopped") or {}).get("stop_ms")
    summary: Message = {"app": app, "mode": mode, "runs": samples}
    for key in ("to_started_ms", "to_app_run_ms", "to_http_ms", "stop_ms"):
        values = [float(s[key]) for s in samples if s.get(key) is not None]
        if values:
            summary[key] = {"median": round(statistics.median(values), 1), "p90": round(_percentile(values, 0.9), 1), "min": round(min(values), 1)}
    return summary
