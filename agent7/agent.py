"""The developer agent. Everything it is allowed to change about itself lives
in a Genome; everything it is not allowed to touch (tasks, hidden tests,
sandbox, scoring, acceptance rule) lives outside it.
"""
import json
import re
from dataclasses import asdict, dataclass, field, replace

from .sandbox import run_hook, run_tests

EFFORTS = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class Genome:
    system_prompt: str
    playbook: tuple = ()          # short rules appended to the system prompt
    self_check: bool = True       # run the visible examples before answering
    repair_rounds: int = 1        # retries after a failed self-check
    write_own_tests: bool = False # ask the model for extra edge-case asserts to self-check against
    effort: str = "medium"
    hook_code: str = ""           # optional `postprocess(code: str) -> str`, run sandboxed
    generation: int = 0
    lineage: tuple = ()           # human-readable history of accepted changes

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        d["playbook"] = tuple(d.get("playbook", ()))
        d["lineage"] = tuple(d.get("lineage", ()))
        return cls(**d)

    def system(self):
        if not self.playbook:
            return self.system_prompt
        return self.system_prompt + "\n\nRules you have learned:\n" + "\n".join(f"- {r}" for r in self.playbook)


BASELINE = Genome(
    system_prompt=(
        "You are a highly experienced Python developer. You read specifications "
        "carefully, handle edge cases the spec implies, and write correct, "
        "efficient, dependency-free code. Answer with a single ```python code "
        "block containing the complete implementation and nothing else."
    ),
)

_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text):
    blocks = _BLOCK.findall(text or "")
    if blocks:
        return max(blocks, key=len)
    return text or ""


def task_prompt(task):
    ex = "\n".join(task.examples)
    return (f"TASK_ID: {task.id}\n\n{task.prompt}\n\n"
            f"The function or class must be named `{task.entry}`. Standard library only.\n\n"
            f"Examples (these are also run as a check):\n```python\n{ex}\n```")


class DeveloperAgent:
    def __init__(self, backend, genome):
        self.backend = backend
        self.g = genome

    def _ask(self, messages):
        return self.backend.complete(self.g.system(), messages, effort=self.g.effort)

    def _own_tests(self, task):
        """Ask the model for extra asserts. They are the agent's guesses, not
        ground truth, so a failure only triggers a repair attempt."""
        msg = (f"{task_prompt(task)}\n\nDo not implement it. Write 4-8 extra single-line "
               f"`assert` statements that probe edge cases of this specification. Only "
               f"include asserts you are certain are correct. Reply with one ```python block.")
        text = self._ask([{"role": "user", "content": msg}])
        return tuple(l.strip() for l in extract_code(text).splitlines() if l.strip().startswith("assert "))

    def solve(self, task):
        """Returns (code, trace). trace records what the agent did, for logs."""
        trace = {"attempts": 0, "self_check": []}
        checks = tuple(task.examples)
        if self.g.write_own_tests:
            try:
                checks += self._own_tests(task)
            except Exception as e:  # a bad guess must not sink the task
                trace["own_tests_error"] = str(e)[:200]
        messages = [{"role": "user", "content": task_prompt(task)}]
        code = ""
        for attempt in range(1 + max(0, self.g.repair_rounds)):
            trace["attempts"] = attempt + 1
            reply = self._ask(messages)
            code = extract_code(reply)
            if self.g.hook_code:
                code, err = run_hook(self.g.hook_code, code)
                if err:
                    trace["hook_error"] = err
            if not self.g.self_check:
                break
            res = run_tests(code, task.entry, checks, per_test_timeout=5.0)
            trace["self_check"].append(res["n_passed"])
            if res["passed"]:
                break
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": "Your code failed its checks:\n" + _feedback(res, checks)
                          + "\nFix it. Some of these checks may have been written by you and could be wrong; "
                            "trust the specification over them. Reply with the full corrected ```python block."}]
        return code, trace


def _feedback(res, checks):
    if res["load"] != "ok":
        return f"{res['load']}: {res.get('error', '')}"
    lines = []
    for chk, r in zip(checks, res["results"]):
        if not r["ok"]:
            lines.append(f"- `{chk.splitlines()[-1][:200]}` -> {r.get('kind')}: {r.get('error', '')}")
    return "\n".join(lines[:8])


def mutate(genome, change):
    """Apply one validated change dict {"field", "value"} and return a new Genome.
    Raises ValueError for anything outside the allowed surface."""
    f, v = change.get("field"), change.get("value")
    if f == "system_prompt":
        if not isinstance(v, str) or not 50 <= len(v) <= 4000:
            raise ValueError("system_prompt must be a 50-4000 char string")
        return replace(genome, system_prompt=v)
    if f == "playbook_add":
        if not isinstance(v, str) or not 5 <= len(v) <= 400:
            raise ValueError("playbook rule must be 5-400 chars")
        if len(genome.playbook) >= 20:
            raise ValueError("playbook full; remove a rule first")
        return replace(genome, playbook=genome.playbook + (v,))
    if f == "playbook_remove":
        if not isinstance(v, int) or not 0 <= v < len(genome.playbook):
            raise ValueError("playbook_remove needs a valid index")
        return replace(genome, playbook=genome.playbook[:v] + genome.playbook[v + 1:])
    if f in ("self_check", "write_own_tests"):
        if not isinstance(v, bool):
            raise ValueError(f"{f} must be bool")
        return replace(genome, **{f: v})
    if f == "repair_rounds":
        if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 3:
            raise ValueError("repair_rounds must be 0-3")
        return replace(genome, repair_rounds=v)
    if f == "effort":
        if v not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}")
        return replace(genome, effort=v)
    if f == "hook_code":
        if not isinstance(v, str) or len(v) > 4000:
            raise ValueError("hook_code must be a string under 4000 chars")
        if v.strip() and "def postprocess" not in v:
            raise ValueError("hook_code must define postprocess(code)")
        return replace(genome, hook_code=v)
    raise ValueError(f"field {f!r} is not mutable")
