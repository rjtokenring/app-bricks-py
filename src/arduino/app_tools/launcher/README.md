# arduino-app-launcher: design notes

Context for whoever extends or fixes the launcher. How to run it, and the contract with arduino-app-cli, are in
[containers/bricks/app-launcher/README.md](../../../../containers/bricks/app-launcher/README.md).

Status: proof of concept (branch `poc-application-launcher`). arduino-app-cli does not drive it yet.

## Why

Today arduino-app-cli generates a compose file per app and creates every container of the app at each start.
The main container (`python-apps-base`) runs `/run.sh`, which checks the app venv, provisions ALSA and the DSP,
then `exec python /app/python/main.py`. That `python` imports numpy, cv2, fastapi... from scratch. On an UNO Q,
the `debug-start-time` app (web_ui + camera + sqlstore) answers HTTP about 6.7 s after
`arduino-app-cli app start`.

The launcher keeps **one container up for all the apps**. For every app it keeps a Python process that has
already done the expensive imports. A start then only runs `main.py`: the same app answers HTTP in about 0.7 s,
and reaches `App.run()` 50 ms after the request.

The goal is to make this *the* way apps start and restart, even with a single app. A restart after an edit
must be as fast and as correct as a first start.

## Constraints that shaped the design

These come from the people who own the feature. Keep them unless they say otherwise.

1. **Every app has its own venv**, `<app>/.cache/.venv`, created by `run.sh` with `--system-site-packages`. An
   app may pin other versions of numpy, cv2 or the library itself. So there can be **no shared warm parent
   process and no fork** (zygote): a warm context belongs to one venv. This is a pool of processes, each bound
   to one app venv.
2. **Venv creation and dependency installs stay in `run.sh`.** The launcher calls it (`run.sh prepare`), it
   does not reimplement it.
3. **The running app sees its root as `/app`**, exactly as in the classic container. Do not rely on
   `APP_HOME` alone: the library and user code hardcode `/app/...` paths, e.g. `/app/assets`, `/app/data`,
   `/app/certs`.
4. **Stop:** SIGTERM, then SIGKILL to the whole process group after 2.5 s.
5. **For now every app is warmed.** Choosing a subset comes later, through `select_apps_to_warm()`.
6. **No fork at all**, not even between apps whose venvs add nothing to the image. A shared parent per
   identical environment was proposed to stop repeating the common imports across workers, and declined.
   Every worker imports on its own.
7. **Two warm-ups at a time** (`APP_LAUNCHER_WARM_CONCURRENCY=2`). Measured on an UNO Q with 6 apps, all ready
   after:

   | concurrency | all apps ready | first, most recent app ready |
   |---|---|---|
   | 1 | 39 s | 7.1 s |
   | 2 | 26 s | 7.4 s |
   | 3 | 21 s | 8.0 s |
   | 4 | 19 s | 9.1 s |
   | 6 | 17 s | 10.8 s |

   Higher values make every warm-up slower through contention and delay the app most likely to start next.

## Processes

```
container app-launcher        init: true, so tini is PID 1 and reaps orphans
└─ supervisor                 `arduino-app-launcher serve`, system Python, stdlib + yaml, asyncio
   ├─ worker[app-a]           <app-a>/.cache/.venv/bin/python /usr/local/.../launcher/worker.py  READY
   ├─ worker[app-b]           <app-b>/.cache/.venv/bin/python .../worker.py  RUNNING main.py
   └─ worker[app-b]'          the next worker of app-b, warming
/app → /home/app/.launcher/current → /home/arduino/ArduinoApps/<running app>   (or → .launcher/none)
```

- **Supervisor.** Never imports app code or heavy libraries, and never imports `arduino.app_utils`. It owns
  the control socket, the pool, the `/app` link, `run.sh prepare`, log relay and events.
- **Worker.** One OS process, **used for at most one run of its app**. It is never reused: the "clean context"
  for the next start is a new process. Thread leaks, the router connection with its `provide`d methods,
  devices, port 7000, module state: the OS cleans all of it when the process ends.

## Life of a worker

1. **Spawn** (`process.Worker.spawn`):
   - started by path with the venv interpreter;
   - cwd is the app's real path;
   - its own session, so its process group is the unit for SIGKILL;
   - stdout and stderr go to one pipe read by the supervisor;
   - the control channel is a `socketpair` end passed as `--control-fd`.

   The env is complete at spawn: container env + app env + `VIRTUAL_ENV`/`PATH` +
   `APP_SHUTDOWN_GRACE_PERIOD_S`. Some modules read env at import (loggers, `cloud_llm`/`cloud_asr` `API_KEY`,
   `tps`), so it cannot be set later.
2. **Warm** (`worker.warm`): the supervisor sends `warm{modules}`. Modules are imported one by one. Between two
   imports the worker polls the channel: a `run` cuts the warm-up short, `quit` or EOF ends the worker. Then
   it sends `ready`. During warm-up **no app code runs** and `/app` is not touched: it may point at another
   app.
3. **Run** (`worker.run`), sent once `/app` points at this app:
   1. check that `realpath(/app)` is this app, otherwise exit 70;
   2. `chdir('/app')`, `sys.path[0] = '/app/python'`, `/app/bricks` on `sys.path` and `PYTHONPATH`,
      `sys.argv = ['/app/python/main.py']`;
   3. send `started` and print the run.sh banner;
   4. install the `App.run()` trace;
   5. `exec` main.py into a fresh `ModuleType('__main__')`.
4. **End.** The interpreter shuts down normally (threading shutdown, atexit) and the process exits.

### Why main.py runs the way it does (do not "simplify")

- **Not `runpy.run_path`.** It puts the previous `__main__` back as soon as the top-level code returns, but
  threads and atexit handlers of the app run after that and must still see the app's `__main__`.
- **The module is permanent** and carries `__file__`, `__loader__` (`SourceFileLoader`), `__spec__=None`,
  `__package__=None`, `__cached__=None` and `__builtins__`: the dunders of a script run as `python main.py`.
- **When main.py returns, the worker pops `__file__` and `__cached__`.** CPython does the same for scripts
  (`_PyRun_SimpleFileObject`): atexit handlers of `python main.py` do not see `__file__`.
- **Uncaught exceptions.** Frames of `worker.py` are stripped, and the hook is called with
  `exc.with_traceback(tb)`. The default hook of Python 3.12+ prints `exc.__traceback__` and ignores its `tb`
  argument. Exit code 1, as for an uncaught exception.
- **The worker never installs a SIGTERM handler.** Idle, SIGTERM kills it. Running, `App.run()` installs its
  own handler (`app.py`, exit 143). This also stays compatible with the import-time handler of PR #554, which
  installs only over `SIG_DFL`.
- **Shadowed module names.** If an app module has the name of a module already imported, e.g. a local
  `fractions.py` or `utils.py` after a preload, the worker `execv`s a fresh `python /app/python/main.py`
  (path `exec`). Correct, just not warm. A fresh interpreter sets `sys.path[0]` to the *real* folder of
  main.py, because CPython realpaths the script folder.
- **Streamlit.** `runpy.run_module('streamlit', alter_sys=True)` with `streamlit run --server.port 7000 main.py`,
  which is the console script in-process. If Streamlit does not import, it falls back to
  `execv python -m streamlit run`.

### What gets imported in advance

`server._warm_modules` → `imports.warm_candidates`, in this order:
1. `APP_LAUNCHER_PRELOAD`: default numpy, cv2, PIL.Image, yaml, arduino.app_utils;
2. the bricks of `app.yaml`. The id → module mapping comes from the `id:` of the installed
   `brick_config.yaml` files and is **not** the folder name: `arduino:video_object_detection` →
   `arduino.app_bricks.video_objectdetection`;
3. module-level absolute imports found by an AST scan of `python/` and `bricks/`, skipping function bodies and
   `if TYPE_CHECKING:` blocks.

Names whose top-level package matches an app module are dropped. The web stack (`imports.WEB_UI_MODULES`:
`arduino.app_bricks.web_ui`, fastapi, fastapi_socketio, starlette, uvicorn, socketio, engineio, with their
submodules) is dropped from all three groups unless `app.yaml` declares `arduino:web_ui`: an app that imports
fastapi without the brick gets it at run time. A library from the scan that imports fastapi itself still brings it
in. The worker additionally skips anything whose `find_spec` origin lies inside the app folder.

## Supervisor

`server.Supervisor`. Each known app has a `Slot`: its `AppInfo`, at most one standby worker, the env of the last
request, a failure count and a prepare lock. There is a single active run (`ActiveRun`). Lifecycle commands
(start, stop, restart, prepare) are serialized by one `asyncio.Lock`; status, logs and events are not.

### start B while A runs (`Supervisor.start`)

1. Resolve B inside `APP_LAUNCHER_APPS_DIR`, with no path traversal, and reload its `app.yaml`.
2. If `needs_prepare(B)`, run `run.sh prepare`. If B is the running app, stop it first: a venv must not change
   under the app using it.
3. Take B's standby worker. Discard it if it is dead, if its fingerprint differs, or in `mode=immediate`.
4. **In parallel:** stop A (`Worker.stop`), and spawn a worker for B if there is none: warm when there is an
   app to stop, immediate otherwise.
5. `Worker.stop`: SIGTERM to the leader → wait `stop_timeout_s` → `killpg(SIGKILL)` → `killpg(SIGKILL)` again
   for children left behind → poll `killpg(pgid, 0)` until the group is gone. Only then is A finished.
6. Short settle (`settle_s`), so the router drops A's provided methods.
7. Repoint `/home/app/.launcher/current` atomically: temp symlink + `os.replace`.
8. Send `run` to B's worker and wait for `started`.
9. Queue the next workers: A's after `replacement_delay_s`, B's after `replacement_delay_s` too, so warming
   does not take CPU from B's own start.

A restart is the same path with A == B: the stop does not queue a worker, the start does.

### Pool

- `rescan()` finds the app folders, those with `app.yaml` and `python/main.py`, and queues a warm-up for every
  selected app without a live worker, in `pool.warm_order` (most recently started first, from
  `/home/app/.launcher/state.json`).
- `APP_LAUNCHER_WARM_CONCURRENCY` consumers process the queue.
- A warm-up runs `run.sh prepare` if needed, spawns the worker and waits for `ready`.
- `Slot.pending_warm` (queued) and `Slot.warming` (preparing or spawning) keep **one warm-up per app at a
  time**. Without them a `warm` request arriving during the prepare spawned a second worker and orphaned the
  first one (seen on the board).
- If a start took the worker meanwhile, the warm-up just returns. `wait_ready` also returns on `started`.
- Failures: a standby that dies is counted once, in `_on_exit`, which also prints what the worker wrote. A
  warm-up timeout or a failed prepare is counted in `_warm_failed`. After `APP_LAUNCHER_MAX_FAILURES` the app
  is not warmed again until a `warm` or `prepare` request; its start then uses an immediate worker.
- No new worker below `APP_LAUNCHER_MEM_RESERVE_MB` of MemAvailable. Idle workers set `oom_score_adj` to 1000
  and restore it at run.

### Readiness and start priority

- **Container health is "the supervisor answers"** (`ping`, the compose healthcheck), about 3 s after
  `docker compose up` on an UNO Q. Health does not wait for the workers: from then on a start always works,
  only its speed depends on the worker. A worker still warming is cut short by the `run`, and the app imports
  the rest itself. An app with no worker yet gets an immediate one. Waiting for a worker to be ready would not
  be faster: the remaining imports have to happen anyway.
- **Readiness per app** is separate: `ping` replies `readiness {apps, ready, not_ready}`, `status` shows each
  worker's state, and `worker_ready` events report it. `arduino-app-launcher ping --all-ready` exits 0 only
  when every selected app would start warm.
- **Start priority** (`_prioritize`). When a start sends `run`, the other workers still warming get SIGSTOP
  (to their process group) and the warm-up consumers wait on `_warm_gate`. Everything resumes, with SIGCONT
  and the gate set, `start_priority_tail_s` (1 s) after the app reaches `App.run()`, when the app ends, or after
  `start_priority_s` (15 s) at most. `run`, `stop` and `quit` resume a suspended worker first: a stopped
  process cannot read its channel. Ready workers are not touched; they sleep on their channel.

### Fingerprints (`appinfo`)

- **`deps_fingerprint`** covers `python/requirements.txt`, `bricks/*/requirements.txt` and the private wheels
  by name and size. `*.whl.installed` counts as `*.whl`: run.sh renames installed wheels, and without this
  normalization every prepare would invalidate itself. A successful prepare records it in
  `<app>/.cache/launcher/deps.sha`, and `needs_prepare` compares against that record.
- **`app_fingerprint`** decides whether a standby worker is still good. It covers `app.yaml`, the deps
  fingerprint, the *list of installed distributions* in the venv (`*.dist-info`, `*.pth`...), the interpreter,
  the env, and launcher version + preload + stop timeout.
  - It does **not** use the modification time of `site-packages`. The worker's own interpreter writes
    `__pycache__` there at start, so every worker looked stale (seen on the board).
  - main.py and the app's own modules are not part of it: they are read at run time, which is what makes
    "edit, then restart" correct.

### App environment

The env comes from the `start`/`prepare` request (`env`). Without one, it comes from `services.main.environment`
of the compose files arduino-app-cli left in `<app>/.cache/` (`app-compose.yaml`, then
`app-compose-overrides.yaml`). This is a stand-in until the CLI sends it. `APP_HOME` defaults to the app's
real path, which is also the host path, since the apps folder is mounted at the same path.

### Logs and events

- **Logs.** Output of the running app goes to the supervisor stdout, so to `docker logs`, and to a 2000-line
  ring buffer (`logs`, `logs -f`). What a worker printed while warming is held, up to 256 kB, and replayed at
  its run, or printed if the worker dies while waiting.
- **Events** (`events` stream): `worker_ready`, `worker_failed`, `app_started`, `app_run` (when the app
  reaches `App.run()`, traced by wrapping `AppController.run`), `app_exited`.

## Modules

| File | Runs in | Role |
|---|---|---|
| `worker.py` | the app's venv interpreter, **stdlib only** | warm-up, run of main.py. Repeats `PROTOCOL_VERSION`; a test checks both match and that it imports only the stdlib |
| `server.py` | supervisor | `Config` (APP_LAUNCHER_* env), `Supervisor`: slots, pool, start/stop, link, control socket |
| `process.py` | supervisor | `Worker`: spawn, channel, output, `stop()` with process-group kill |
| `prepare.py` | supervisor | `run.sh prepare` with `APP_DIR` and `SKIP_DEVICE_PROVISIONING=1` |
| `appinfo.py` | supervisor | app folders, `app.yaml`, brick id mapping, compose env, fingerprints |
| `imports.py` | supervisor | AST scan, local names, warm candidates |
| `pool.py` | supervisor | pure policy: selection, order, memory guard |
| `protocol.py` | both sides | JSON-lines framing, error codes |
| `client.py`, `cli.py`, `bench.py` | anywhere | client, `arduino-app-launcher` command, start-time bench |

`server.py`, `process.py`, `prepare.py` and `worker.py` raise ImportError on Windows. The other modules, and
their tests, are cross-platform.

### Changes outside this package

- `containers/bricks/python-apps-base/scripts/run.sh`, still compatible with the classic flow:
  - `BASE_DIR="${APP_DIR:-/app}"`;
  - export `VIRTUAL_ENV`/`PATH` instead of sourcing `bin/activate`. The activate script hardcodes
    `/app/.cache/.venv`, which in the launcher is *the running app's* venv, so another app's requirements
    would have gone into it;
  - `uv venv --relocatable` for new venvs;
  - `python -m streamlit` instead of the `streamlit` script, whose shebang is absolute;
  - `SKIP_DEVICE_PROVISIONING`: the ALSA script rewrites `~/.asoundrc` under a running app;
  - cleanup of the prepare temp folder.
- `src/arduino/app_utils/app.py`: `SHUTDOWN_GRACE_PERIOD_S` is read from `APP_SHUTDOWN_GRACE_PERIOD_S`
  (default 5 s), and the peripherals budget is 30% of it. At 2.5 s: 0.75 s for peripherals, 0.5 s headroom,
  1.25 s for bricks.
- `containers/bricks/app-launcher/`:
  - the image: `/app` is a symlink, `WORKDIR /home/app` because `/app` may dangle;
  - `launcher.sh`: device provisioning once, then the supervisor;
  - `compose.example.yaml`, derived from the `main` service the CLI generates;
  - `dev/app-sidecars.sh`, `dev/bench-classic.sh`.

## Sidecars

The bricks reach their sidecar containers (e.g. the Edge Impulse runner `ei-video-obj-detection-runner`) by
compose service name, on the app's compose network. The launcher container does not start sidecars.
Whoever starts the app has to:
1. bring the app's sidecars up and wait until they are healthy;
2. connect the launcher container to `<app>_default`, so the service names resolve;
3. send `start`.

Before the next app, in reverse: stop, disconnect, `compose down`.

`dev/app-sidecars.sh up|down <app>` does this by hand from the compose file the CLI already generated
(`.cache/app-compose.yaml` + overrides, every service but `main`). In production this is arduino-app-cli's
job. Recreating the launcher container drops its network connections: the sidecars must be connected again.

## Tests

`tests/arduino/app_tools/launcher/`:
- **Cross-platform:** `test_launcher_protocol`, `_appinfo`, `_imports`, `_pool`.
- **POSIX only:** `test_launcher_worker` (parity with `python main.py`: dunders, paths, signals, exit codes,
  atexit, tracebacks, shadowing, immediate mode) and `test_launcher_supervisor` (real workers and a fake
  run.sh: switch, restart after an edit, SIGKILL with children, self-exit, prepare, crash while warming,
  double warm request, env, control socket).
- `tests/containers/test_run_sh_launcher.py`: run.sh with fake tools.
- `tests/arduino/app_utils/test_app_grace_period.py`

On Windows the POSIX tests skip. Run them in Linux, e.g. a `python:3.13-slim` container with `--init` (orphans
must be reaped) and a non-root user.

## Measured on an UNO Q (4 cores, 3.6 GB)

`debug-start-time`, medians, from the request:

| | to `App.run()` | to HTTP |
|---|---|---|
| arduino-app-cli today | | ~6.75 s |
| launcher, immediate worker | 3.75 s | 4.36 s |
| launcher, warm worker | 51 ms | 681 ms |

- Container start, until the socket and healthcheck answer: about 3.1–3.3 s. With 6 apps every worker is
  ready after about 26 s, or 30 s when a start during the boot suspends the warm-ups.
- A start sent as soon as the socket answers, its worker still warming, reaches HTTP after 5.5 s without start
  priority and 4.4 s with it.
- Warm-up: 2–10 s per app. Private memory: 34–87 MB per idle worker; 7 apps took about 450 MB.
- Stop: 0.4–1.7 s.
- `CameraCodeDetection.stop()` does not fit the 1.25 s brick budget of a 2.5 s stop. It is abandoned, and the
  camera is still released by the peripherals step.

## Known gaps and next steps

- arduino-app-cli integration: generate the compose file without `main`, sidecars + network, send `start` with
  `env`, read events and logs.
- The classic flow with the new run.sh is not yet tested on a board.
- Streamlit is not tested on a board.
- Selection of the apps to warm (`APP_LAUNCHER_WARM_APPS`, `pool.select_apps_to_warm`).
- Learned preloads: snapshot `sys.modules` at `App.run()` and replay it at the next warm-up.
- Trigger through the Bridge instead of, or next to, the unix socket.
- Port 7000 is published by the launcher container: it cannot run next to classic app containers.
- PR #554 (lazy imports, `perf/app-startup-time`) changes run.sh too: expect a merge to reconcile.
