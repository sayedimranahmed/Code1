"""Self-improvement loop.

Each generation the agent reads its own genome and its train failures,
proposes exactly one change, and the change is kept only if the train score
rises by at least `min_gain` tasks on two independent runs. After the last
generation the result is scored once on the held-out split, which the
optimizer never sees. The new genome is promoted to `current.json` only if it
does not do worse than its parent on held-out.

The agent can change its prompt, its playbook, its strategy knobs and a
sandboxed post-processing hook. It cannot change the tasks, tests, sandbox,
scoring or this acceptance rule: those live outside the genome.
"""
import json
import time
from dataclasses import replace
from pathlib import Path

from .agent import EFFORTS, Genome, extract_code, mutate
from .evaluate import evaluate
from .tasks import get_tasks

META_SYSTEM = (
    "You are improving an autonomous coding agent that you are also running as. "
    "You will see its current configuration (its genome) and the tasks it failed. "
    "Propose exactly ONE change that you expect to fix a class of failures, not a "
    "single task. Generic, transferable rules beat task-specific hints: the change "
    "will be judged on unseen tasks. Reply with only a JSON object."
)

MUTABLE_HELP = f"""Allowed changes (pick one):
- {{"field": "system_prompt", "value": "<full new prompt, 50-4000 chars>"}}
- {{"field": "playbook_add", "value": "<one short rule>"}}
- {{"field": "playbook_remove", "value": <index>}}
- {{"field": "self_check", "value": true|false}}  (run visible examples before answering)
- {{"field": "write_own_tests", "value": true|false}}  (model writes extra edge-case asserts to self-check)
- {{"field": "repair_rounds", "value": 0-3}}
- {{"field": "effort", "value": one of {list(EFFORTS)}}}  (more effort costs more tokens)
- {{"field": "hook_code", "value": "<python defining postprocess(code: str) -> str>"}}  (runs on every answer, sandboxed; empty string removes it)
Add a "rationale" key explaining which failures this targets."""


def _failure_report(res, tasks_by_id, limit=10):
    out = []
    for r in res.failures()[:limit]:
        t = tasks_by_id[r.task_id]
        out.append(
            f"### {t.id}\nSpec: {t.prompt[:600]}\n"
            f"Outcome: load={r.load}, hidden tests passed {r.n_passed}/{r.n_total}, failure kinds={list(r.fail_kinds)}, "
            f"attempts={r.trace.get('attempts')}, self-check passes per attempt={r.trace.get('self_check')}\n"
            f"Agent's code:\n```python\n{r.code[:1500]}\n```")
    return "\n\n".join(out) or "(no failures)"


def propose(backend, genome, res, tasks_by_id, rejected):
    msg = (f"Current genome:\n```json\n{genome.to_json()}\n```\n\n"
           f"Train score: {res.n_passed}/{len(res.results)}.\n\nFailed train tasks (hidden test "
           f"contents are withheld on purpose):\n\n{_failure_report(res, tasks_by_id)}\n\n"
           f"Changes already tried and rejected this run (do not repeat):\n"
           + ("\n".join(f"- {json.dumps(c)}" for c in rejected) or "(none)")
           + f"\n\n{MUTABLE_HELP}")
    text = backend.complete(META_SYSTEM, [{"role": "user", "content": msg}], effort="high")
    return _parse_json(text)


def _parse_json(text):
    text = (text or "").strip()
    if "```" in text:
        text = extract_code(text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


def evolve(backend, genome, generations=5, min_gain=2, workers=4, run_dir=None,
           promote_to=None, log=print):
    run_dir = Path(run_dir or f"runs/{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    train, heldout = get_tasks("train"), get_tasks("heldout")
    by_id = {t.id: t for t in train}
    events = run_dir / "events.jsonl"

    def record(**kw):
        kw["t"] = time.time()
        with events.open("a") as f:
            f.write(json.dumps(kw, default=str) + "\n")

    start = genome
    parent_res = evaluate(backend, genome, train, workers)
    log(f"gen {genome.generation}: train {parent_res.n_passed}/{len(train)}")
    record(kind="start", genome=json.loads(genome.to_json()), train=parent_res.summary())
    rejected = []

    for _ in range(generations):
        try:
            change = propose(backend, genome, parent_res, by_id, rejected)
            child = mutate(genome, change)
        except Exception as e:
            log(f"  proposal invalid: {e}")
            record(kind="invalid_proposal", error=str(e))
            rejected.append({"invalid": str(e)[:200]})
            continue
        child_res = evaluate(backend, child, train, workers)
        need = parent_res.n_passed + min_gain
        verdict = "rejected"
        confirm_res = None
        if child_res.n_passed >= need:
            # Confirm on a second run: one lucky sample is not an improvement.
            confirm_res = evaluate(backend, child, train, workers)
            if confirm_res.n_passed >= need:
                verdict = "accepted"
        log(f"  {change.get('field')}: train {child_res.n_passed}"
            + (f"/{confirm_res.n_passed}" if confirm_res else "") + f" vs parent {parent_res.n_passed} -> {verdict}")
        record(kind="candidate", change=change, train=child_res.summary(),
               confirm=confirm_res.summary() if confirm_res else None, verdict=verdict)
        if verdict == "accepted":
            note = f"{change.get('field')}: {str(change.get('rationale', ''))[:160]}"
            genome = replace(child, generation=genome.generation + 1, lineage=genome.lineage + (note,))
            # Use the lower of the two runs as the new bar, so noise cannot ratchet it up.
            parent_res = min(child_res, confirm_res, key=lambda r: r.n_passed)
            rejected = []
        else:
            rejected.append({k: change.get(k) for k in ("field", "value")})

    (run_dir / "final_genome.json").write_text(genome.to_json())
    report = {"generations_accepted": genome.generation - start.generation, "train_final": parent_res.summary()}
    if genome != start:
        base_h = evaluate(backend, start, heldout, workers)
        new_h = evaluate(backend, genome, heldout, workers)
        report.update(heldout_start=base_h.summary(), heldout_final=new_h.summary())
        promoted = new_h.n_passed >= base_h.n_passed
        report["promoted"] = promoted
        log(f"held-out: start {base_h.n_passed}/{len(heldout)} -> final {new_h.n_passed}/{len(heldout)}"
            + ("" if promoted else " (regressed: NOT promoted)"))
        if promoted and promote_to:
            Path(promote_to).write_text(genome.to_json())
    else:
        report["promoted"] = False
        log("no change accepted; held-out not spent")
    record(kind="end", **report)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    return genome, report
