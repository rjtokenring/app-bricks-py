# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""arduino-app-launcher: one long-lived container that keeps a warm Python process ready for every app.

The supervisor (`arduino-app-launcher serve`) runs on the system interpreter and never imports app code.
For every app it keeps one worker: a process started with the interpreter of that app's own venv, which
imports the heavy libraries in advance and then waits. Starting the app tells its worker to run
/app/python/main.py; when the app stops the worker process ends with it, and a fresh one takes its place.
"""
