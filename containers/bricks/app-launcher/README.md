# app-launcher

One long-lived container for every app in `/home/arduino/ArduinoApps`, instead of one `python-apps-base`
container created per app start. Its entrypoint runs `arduino-app-launcher serve`
(`src/arduino/app_tools/launcher/`), which keeps a **warm worker per app** and runs the requested app in it.

Status: proof of concept. arduino-app-cli does not drive it yet; [`dev/app-sidecars.sh`](dev/app-sidecars.sh)
and the `arduino-app-launcher` client stand in for it.

## How an app starts

```
container app-launcher (tini as PID 1)
└─ supervisor   arduino-app-launcher serve       system Python, no app code, no heavy import
   ├─ worker[app-a]  <app-a>/.cache/.venv/bin/python …/launcher/worker.py   ready
   ├─ worker[app-b]  <app-b>/.cache/.venv/bin/python …/launcher/worker.py   running /app/python/main.py
   └─ worker[app-b]' its next worker, warming for the next start
/app → /home/app/.launcher/current → /home/arduino/ArduinoApps/<running app>
```

- **One worker per app venv.** Each app has its own venv (`.cache/.venv`, created and filled by `run.sh` as
  today). A worker is started with that venv's interpreter, so imports resolve as for `python main.py` in
  `run.sh`: app venv first, image site-packages after. Before the app runs, the worker imports the heavy
  libraries (`APP_LAUNCHER_PRELOAD`), the bricks the app declares and the third-party modules its sources
  import. It never imports the app's own modules.
- **Start.** The running app gets SIGTERM, and SIGKILL to its whole process group after 2.5 s
  (`APP_LAUNCHER_STOP_TIMEOUT_S`; the app shutdown budgets shrink to fit, see `APP_SHUTDOWN_GRACE_PERIOD_S`
  in `app_utils/app.py`). Once nothing of it is left, `/app` is pointed at the new app and its worker runs
  `/app/python/main.py` as `__main__`: cwd `/app`, `sys.path[0]` `/app/python`, `/app/bricks` on the path,
  as `run.sh` does.
- **Clean context.** A worker runs one app once and ends with it: threads, router connection, devices and
  port 7000 are released by the process exit. A new worker for that app starts warming right after.
- **Dependencies.** When an app has no venv yet, or its `requirements.txt`, brick requirements or private
  wheels changed, the launcher runs `APP_DIR=<app> SKIP_DEVICE_PROVISIONING=1 /run.sh prepare` before giving
  it a worker.
- **Fallbacks.** An app module named like a module imported in advance (e.g. its own `utils.py`) makes the
  worker re-exec a fresh `python /app/python/main.py`. Streamlit apps run `python -m streamlit run` in the
  worker.

## Contract for arduino-app-cli

1. Run this container once (see [`compose.example.yaml`](compose.example.yaml)): the `main` service of today
   without the app mount on `/app`, with `/home/arduino/ArduinoApps` mounted at the same path, the control
   socket folder mounted, `init: true`.
2. Generate the app compose file without the `main` service; its sidecars as today.
3. To start app B while A runs:
   1. `{"cmd": "stop"}` to the launcher, then `compose down` of A's sidecars;
   2. `compose up --wait` of B's sidecars and connect the launcher to B's network;
   3. `{"cmd": "start", "app": "B", "env": {...}}`, `env` being the environment the `main` service of B
      would have had (`APP_HOME`, `BOARD_NAME`, `HOST_IP`, brick variables). Without `env` the launcher
      reads it from `B/.cache/app-compose.yaml`.
4. Logs: the app output goes to the container log (`docker logs app-launcher`), the `logs` request returns
   the running app's; `events` streams `app_started`, `app_run`, `app_exited` with exit code and signal.

## Control protocol

Unix socket `/run/arduino-app-launcher/launcher.sock`, one JSON object per line. The client is the same
command:

| Request | Client | Reply |
|---|---|---|
| `{"cmd": "start", "app", "env"?, "prepare"?: "auto"\|"never", "mode"?: "auto"\|"immediate"}` | `arduino-app-launcher start APP [--env K=V]` | `run_id`, `pid`, `path` (warm, exec, streamlit), timings |
| `{"cmd": "stop"}` | `stop` | exit code, signal, `killed`, `stop_ms` |
| `{"cmd": "restart", "app"?}` | `restart [APP]` | as start |
| `{"cmd": "prepare", "app"}` | `prepare APP` | runs `run.sh prepare` |
| `{"cmd": "warm", "app"}`, `{"cmd": "rescan"}` | `warm APP`, `rescan` | |
| `{"cmd": "status"}` | `status [--json]` | running app, worker of each app with state and memory |
| `{"cmd": "logs", "tail"?, "follow"?}` | `logs [-f]` | running app output |
| `{"cmd": "events"}` | `events` | stream of events |

Errors are `{"ok": false, "error": {"code", "message"}}`, codes `bad_request`, `not_found`, `prepare_failed`,
`spawn_failed`, `busy`, `internal`.

## Settings

| Variable | Default | |
|---|---|---|
| `APP_LAUNCHER_APPS_DIR` | `/home/arduino/ArduinoApps` | |
| `APP_LAUNCHER_WARM_APPS` | `all` | or a comma-separated list of app folders |
| `APP_LAUNCHER_PRELOAD` | `numpy,cv2,PIL.Image,requests,yaml,arduino.app_utils` | imported by every worker |
| `APP_LAUNCHER_STOP_TIMEOUT_S` | `2.5` | SIGTERM to SIGKILL |
| `APP_LAUNCHER_WARM_CONCURRENCY` | `2` | workers warming at once |
| `APP_LAUNCHER_REPLACEMENT_DELAY_S` | `5` | delay before warming the next worker of an app just started |
| `APP_LAUNCHER_MEM_RESERVE_MB` | `400` | no new worker below this MemAvailable |

## Trying it on a board

```sh
docker compose -f compose.example.yaml up -d
docker exec app-launcher arduino-app-launcher status
docker exec app-launcher arduino-app-launcher start debug-start-time
dev/app-sidecars.sh up detect-objects-on-camera
docker exec app-launcher arduino-app-launcher start detect-objects-on-camera
docker exec app-launcher arduino-app-launcher bench debug-start-time --runs 5 --http-url http://127.0.0.1:7000/
```

Port 7000 is published by this container: stop it before starting apps with arduino-app-cli, and the other
way round.
