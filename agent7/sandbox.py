"""Run untrusted Python (agent-written solutions and hooks) in a subprocess.

What this does: a fresh `python -I` process per call, an empty temp working
directory, a scrubbed environment (no API keys), CPU / memory / file-size
rlimits, a per-test alarm and a hard wall-clock kill. The payload arrives on
stdin and is never written to disk, so a solution cannot read the hidden tests
from a file.

What this does NOT do: block network access or filesystem reads. It is a
guard against runaway or sloppy code, not a security boundary against
hostile code. The container this runs in is the real boundary. Before this
runs anywhere persistent (host01), wrap it in a container or nsjail with no
network.
"""
import json
import re
import os
import secrets
import subprocess
import sys
import tempfile

_HARNESS = r'''
import json, os, signal, sys, traceback
payload = json.loads(sys.stdin.read())
nonce = payload.pop("nonce")
out_fd = os.dup(1)
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 1)
sys.stdout = open(os.devnull, "w")

def emit(obj):
    os.write(out_fd, (nonce + json.dumps(obj) + "\n").encode())
    os._exit(0)

class _Timeout(BaseException):
    pass

def _alarm(*_):
    raise _Timeout()

signal.signal(signal.SIGALRM, _alarm)
per_test = payload["per_test_timeout"]

def short(e):
    return (type(e).__name__ + ": " + str(e))[:300]

if payload["mode"] == "hook":
    ns = {"__name__": "agent7_hook"}
    try:
        signal.setitimer(signal.ITIMER_REAL, per_test)
        exec(compile(payload["hook_src"], "<hook>", "exec"), ns)
        res = ns["postprocess"](payload["code"])
        signal.setitimer(signal.ITIMER_REAL, 0)
        if not isinstance(res, str):
            raise TypeError("postprocess must return str")
        emit({"ok": True, "code": res})
    except BaseException as e:
        emit({"ok": False, "error": short(e)})

src = payload["source"]
tests = payload["tests"]
ns = {"__name__": "solution"}
try:
    code = compile(src, "<solution>", "exec")
except SyntaxError as e:
    emit({"load": "syntax_error", "error": short(e), "results": []})
try:
    signal.setitimer(signal.ITIMER_REAL, per_test)
    exec(code, ns)
    signal.setitimer(signal.ITIMER_REAL, 0)
except _Timeout:
    emit({"load": "timeout", "error": "timed out while importing", "results": []})
except BaseException as e:
    emit({"load": "load_error", "error": short(e), "results": []})
if payload["entry"] not in ns:
    emit({"load": "missing_entry", "error": "no name " + payload["entry"], "results": []})

results = []
for t in tests:
    g = dict(ns)
    g["SOLUTION_SOURCE"] = src
    try:
        signal.setitimer(signal.ITIMER_REAL, per_test)
        exec(compile(t, "<test>", "exec"), g)
        signal.setitimer(signal.ITIMER_REAL, 0)
        results.append({"ok": True})
    except _Timeout:
        results.append({"ok": False, "kind": "timeout", "error": "exceeded %ss" % per_test})
    except AssertionError as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        results.append({"ok": False, "kind": "assertion", "error": short(e)})
    except BaseException as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        results.append({"ok": False, "kind": "exception", "error": short(e)})
emit({"load": "ok", "results": results})
'''

MEM_BYTES = 1024 * 1024 * 1024

# Solutions share a process with the test runner, so a devious one could
# reach into the harness (stack frames, gc, __main__) and forge a pass. These
# patterns have no legitimate use in the benchmark tasks; a match fails the
# task as "tamper" so reward hacking shows up in the logs instead of the score.
# A determined attacker can obfuscate past a regex: this raises the bar, it is
# not proof.
_TAMPER = re.compile(r"_getframe|f_globals|f_back|gc\.get_(objects|referrers)|inspect\.(stack|currentframe)"
                     r"|sys\.modules|os\.write|os\.dup|_exit|signal\.|setitimer|builtins")


def _limits(cpu_seconds):
    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024,) * 2)
        os.setsid()
    return apply


def _run(payload, wall_timeout):
    nonce = "@@" + secrets.token_hex(16) + "@@"
    payload = dict(payload, nonce=nonce)
    with tempfile.TemporaryDirectory(prefix="agent7-") as cwd:
        env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "HOME": cwd}
        try:
            p = subprocess.run(
                [sys.executable, "-I", "-c", _HARNESS],
                input=json.dumps(payload), capture_output=True, text=True,
                cwd=cwd, env=env, timeout=wall_timeout,
                preexec_fn=_limits(int(wall_timeout) + 1),
            )
        except subprocess.TimeoutExpired:
            return None, "wall-clock timeout"
    for line in p.stdout.splitlines():
        if line.startswith(nonce):
            return json.loads(line[len(nonce):]), None
    return None, ("crashed (exit %s): %s" % (p.returncode, p.stderr[-300:])).strip()


def run_tests(source, entry, tests, per_test_timeout=5.0):
    """Run `tests` (assert snippets) against `source`. Returns a dict with
    `passed` (bool), `n_passed`, `n_total`, `load`, and per-test `results`."""
    hit = _TAMPER.search(source or "")
    if hit:
        return {"load": "tamper", "error": f"forbidden construct {hit.group(0)!r}", "results": [],
                "n_passed": 0, "n_total": len(tests), "passed": False}
    wall = min(120.0, per_test_timeout * (len(tests) + 1) + 5)
    out, err = _run({"mode": "tests", "source": source, "entry": entry, "tests": list(tests),
                     "per_test_timeout": per_test_timeout}, wall)
    if out is None:
        out = {"load": "crash", "error": err, "results": []}
    res = out["results"]
    n_ok = sum(r["ok"] for r in res)
    out.update(n_passed=n_ok, n_total=len(tests),
               passed=out["load"] == "ok" and n_ok == len(tests) and len(res) == len(tests))
    return out


def run_hook(hook_src, code, timeout=5.0):
    """Apply an agent-written `postprocess(code) -> str` hook in the sandbox.
    Returns the transformed code, or the original code if the hook fails."""
    if not hook_src or not hook_src.strip():
        return code, None
    out, err = _run({"mode": "hook", "hook_src": hook_src, "code": code,
                     "per_test_timeout": timeout}, timeout + 5)
    if out is None:
        return code, err
    if not out["ok"]:
        return code, out["error"]
    return out["code"], None
