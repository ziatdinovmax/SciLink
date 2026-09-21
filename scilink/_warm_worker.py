"""The child side of :class:`scilink.executors.WarmScriptExecutor`.

Run as a plain file (``python _warm_worker.py``), never imported: it must start
without importing ``scilink``, so the only import cost it ever pays is what the
scripts themselves import, once.

Protocol: one JSON request per line on stdin, one JSON reply per line on a
duplicate of the original stdout. File descriptors 1 and 2 are pointed at
per-run files while a script runs, so output from C extensions is captured too
and can never corrupt the protocol.
"""

import io
import json
import os
import runpy
import sys
import traceback


def main() -> int:
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    home = os.getcwd()
    proto.write(json.dumps({"ready": True, "pid": os.getpid()}) + "\n")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        if req.get("op") == "quit":
            break
        proto.write(json.dumps(_run(req, devnull, home)) + "\n")
    return 0


def _run(req, devnull, home):
    cwd, script = req["cwd"], req["script"]
    out_path, err_path = os.path.join(cwd, ".warm_stdout"), os.path.join(cwd, ".warm_stderr")
    saved_env = dict(os.environ)
    saved_argv, saved_path = list(sys.argv), list(sys.path)
    code = 0
    try:
        os.environ.update({str(k): str(v) for k, v in (req.get("env") or {}).items()})
        os.chdir(cwd)
        out_fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        err_fd = os.open(err_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(out_fd, 1)
        os.dup2(err_fd, 2)
        os.close(out_fd)
        os.close(err_fd)
        sys.stdout = io.TextIOWrapper(os.fdopen(os.dup(1), "wb"), encoding="utf-8", line_buffering=True)
        sys.stderr = io.TextIOWrapper(os.fdopen(os.dup(2), "wb"), encoding="utf-8", line_buffering=True)
        sys.argv = [script]
        try:
            runpy.run_path(script, run_name="__main__")
        except SystemExit as e:                       # a script may end with sys.exit()
            c = e.code
            code = 0 if c in (None, 0) else (c if isinstance(c, int) else 1)
            if code and not isinstance(c, int):
                print(c, file=sys.stderr)
        except BaseException:                         # noqa: BLE001 - reported like a failed process
            traceback.print_exc()
            code = 1
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
                stream.close()
            except Exception:                         # noqa: BLE001
                pass
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        sys.argv, sys.path[:] = saved_argv, saved_path
        os.environ.clear()
        os.environ.update(saved_env)
        try:
            os.chdir(home)
        except OSError:
            pass
        _tidy()
    reply = {"returncode": code}
    for key, path in (("stdout", out_path), ("stderr", err_path)):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                reply[key] = fh.read()
            os.remove(path)
        except OSError:
            reply[key] = ""
    return reply


def _tidy():
    """What a script leaves behind in a process that lives on: open figures."""
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        try:
            plt.close("all")
        except Exception:                             # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(main())
