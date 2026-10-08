# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The arduino-app-launcher command: `serve` runs the supervisor, the other commands talk to it."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .protocol import Message

DEFAULT_SOCKET = "/run/arduino-app-launcher/launcher.sock"


def _parse_env(pairs: list[str] | None) -> dict[str, str] | None:
    if not pairs:
        return None
    env: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--env expects KEY=VALUE, got {pair!r}")
        env[key] = value
    return env


def _print(message: Message) -> None:
    print(json.dumps(message, indent=2, default=str))


def _print_status(status: Message) -> None:
    active = status.get("active")
    mem = status.get("mem", {})
    available = mem.get("MemAvailable")
    print(f"launcher {status.get('version')}, up {status.get('uptime_s')}s, MemAvailable {available // 1024 if available else '?'} MB")
    if active:
        print(f"running: {active['app']} (run {active['run_id']}, pid {active['pid']}, {active['path']}, up {active['uptime_s']}s)")
    else:
        print("running: none")
    print(f"{'APP':32} {'WORKER':10} {'PID':>7} {'WARM ms':>8} {'USS MB':>7}  NOTES")
    for app in status.get("apps", []):
        worker = app.get("worker") or {}
        uss = worker.get("uss_kb")
        notes: list[str] = []
        if worker.get("suspended"):
            notes.append("suspended while an app starts")
        if not app.get("selected"):
            notes.append("not selected")
        if app.get("pending_warm"):
            notes.append("queued")
        if app.get("warm_error"):
            notes.append(f"error: {app['warm_error']}")
        if app.get("last_exit"):
            last = app["last_exit"]
            notes.append(f"last exit {last.get('signal') or last.get('exit_code')}")
        print(
            f"{app['name']:32} {worker.get('state', '-'):10} {worker.get('pid') or '-':>7} {worker.get('warm_ms') or '-':>8} "
            f"{round(uss / 1024, 1) if uss else '-':>7}  {'; '.join(notes)}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="arduino-app-launcher", description=__doc__)
    parser.add_argument("--socket", default=os.environ.get("APP_LAUNCHER_SOCKET", DEFAULT_SOCKET), help="control socket of the supervisor")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("serve", help="run the supervisor (the container entrypoint)")
    ping = commands.add_parser("ping", help="exit 0 when the supervisor answers: the container healthcheck")
    ping.add_argument("--all-ready", action="store_true", help="exit 0 only once every selected app has a warm worker")
    ping.add_argument("-q", "--quiet", action="store_true")
    start = commands.add_parser("start", help="start an app, stopping the running one")
    start.add_argument("app", help="app folder name, or path inside the apps directory")
    start.add_argument("--env", action="append", metavar="KEY=VALUE", help="app environment; default: the one of the CLI compose file")
    start.add_argument("--no-prepare", action="store_true", help="do not run run.sh prepare even if the dependencies changed")
    start.add_argument("--immediate", action="store_true", help="ignore the warm worker, start a fresh one (for measurements)")
    commands.add_parser("stop", help="stop the running app")
    restart = commands.add_parser("restart", help="start an app again, the running one by default")
    restart.add_argument("app", nargs="?")
    commands.add_parser("reload", help="run the edited main.py of the running app again in its process; restarts it if it cannot")
    prepare = commands.add_parser("prepare", help="run run.sh prepare on an app: venv and dependencies")
    prepare.add_argument("app")
    prepare.add_argument("--env", action="append", metavar="KEY=VALUE")
    warm = commands.add_parser("warm", help="give an app a warm worker now")
    warm.add_argument("app")
    commands.add_parser("rescan", help="pick up app folders added or removed")
    status = commands.add_parser("status", help="running app and workers")
    status.add_argument("--json", action="store_true")
    logs = commands.add_parser("logs", help="output of the running app")
    logs.add_argument("--tail", type=int, default=200)
    logs.add_argument("-f", "--follow", action="store_true")
    commands.add_parser("events", help="follow the launcher events, one JSON object per line")
    commands.add_parser("shutdown", help="stop the running app and the supervisor")
    bench = commands.add_parser("bench", help="measure the start time of an app")
    bench.add_argument("app")
    bench.add_argument("--runs", type=int, default=5)
    bench.add_argument("--mode", choices=("warm", "immediate"), default="warm")
    bench.add_argument("--http-url", help="also time until this URL answers, e.g. http://127.0.0.1:7000/")
    bench.add_argument("--settle", type=float, default=2.0, help="seconds the app runs before it is stopped")

    args = parser.parse_args(argv)
    socket_path = Path(args.socket)

    if args.command == "serve":
        import asyncio

        from .server import Config, serve

        config = Config.from_env()
        config.socket_path = socket_path
        asyncio.run(serve(config))
        return

    from .client import Client, LauncherUnavailable

    client = Client(
        socket_path, timeout=None if args.command in ("start", "restart", "reload", "prepare", "bench") else (2.0 if args.command == "ping" else 60.0)
    )
    try:
        reply: Message
        if args.command == "start":
            request: dict[str, Any] = {"app": args.app}
            env = _parse_env(args.env)
            if env is not None:
                request["env"] = env
            if args.no_prepare:
                request["prepare"] = "never"
            if args.immediate:
                request["mode"] = "immediate"
            reply = client.call("start", **request)
        elif args.command == "restart":
            reply = client.call("restart", **({"app": args.app} if args.app else {}))
        elif args.command == "reload":
            reply = client.call("reload")
        elif args.command == "prepare":
            env = _parse_env(args.env)
            reply = client.call("prepare", app=args.app, **({"env": env} if env is not None else {}))
        elif args.command == "warm":
            reply = client.call("warm", app=args.app)
        elif args.command == "ping":
            reply = client.call("ping")
            readiness = reply.get("readiness", {})
            healthy = bool(reply.get("ok")) and (not args.all_ready or not readiness.get("not_ready"))
            if not args.quiet:
                print(f"ok, {len(readiness.get('ready', []))}/{readiness.get('apps', 0)} apps ready" if healthy else f"not ready: {readiness}")
            raise SystemExit(0 if healthy else 1)
        elif args.command == "status":
            reply = client.call("status")
            if not args.json and reply.get("ok"):
                _print_status(reply)
                return
        elif args.command == "logs":
            if args.follow:
                for message in client.stream("logs", follow=True, tail=args.tail):
                    if "line" in message:
                        print(message["line"], flush=True)
                return
            reply = client.call("logs", tail=args.tail)
            if reply.get("ok"):
                print("\n".join(reply.get("lines", [])))
                return
        elif args.command == "events":
            for message in client.stream("events"):
                print(json.dumps(message, default=str), flush=True)
            return
        elif args.command == "bench":
            from .bench import run_bench

            reply = run_bench(client, args.app, args.runs, args.mode, args.http_url, args.settle)
        else:
            reply = client.call(args.command)
    except (LauncherUnavailable, OSError) as e:
        if not getattr(args, "quiet", False):
            print(e, file=sys.stderr)
        raise SystemExit(2) from e
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    _print(reply)
    if reply.get("ok") is False:
        raise SystemExit(1)
