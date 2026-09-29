"""CLI:  python -m agent7 {validate,eval,baseline,evolve}"""
import argparse
import json
import sys
from pathlib import Path

from .agent import BASELINE, Genome
from .backends import make_backend
from .evaluate import evaluate
from .sandbox import run_tests
from .tasks import get_tasks

GENOME_DIR = Path(__file__).parent / "genome"
CURRENT = GENOME_DIR / "current.json"


def load_genome(name):
    if name == "baseline":
        return BASELINE
    p = CURRENT if name == "current" else Path(name)
    return Genome.from_json(p.read_text()) if p.exists() else BASELINE


def cmd_validate(a):
    """No model needed. Proves the benchmark is sound: every reference solution
    passes its hidden tests and examples, every hidden suite rejects a stub, and
    the full agent pipeline scores 100% with the oracle backend."""
    bad = []
    for t in get_tasks():
        ok = run_tests(t.reference, t.entry, t.tests + t.examples)
        if not ok["passed"]:
            bad.append((t.id, "reference fails", ok))
        stub = f"def {t.entry}(*a, **k):\n    return None\n"
        if t.entry[0].isupper():
            stub = f"class {t.entry}:\n    def __init__(self, *a, **k): pass\n    def __getattr__(self, n): return lambda *a, **k: None\n"
        if run_tests(stub, t.entry, t.tests)["passed"]:
            bad.append((t.id, "stub passes hidden tests", None))
    res = evaluate(make_backend("oracle"), BASELINE, get_tasks(), workers=a.workers)
    print(f"tasks: {len(get_tasks())} (train {len(get_tasks('train'))}, held-out {len(get_tasks('heldout'))})")
    print(f"oracle pipeline score: {res.n_passed}/{len(res.results)}")
    for b in bad:
        print("PROBLEM:", b[0], b[1], json.dumps(b[2])[:400] if b[2] else "")
    ok = not bad and res.n_passed == len(res.results)
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


def cmd_eval(a):
    res = evaluate(make_backend(a.backend, a.model), load_genome(a.genome), get_tasks(a.split), a.workers)
    print(json.dumps(res.summary(), indent=2))
    return 0


def cmd_baseline(a):
    """Freeze the baseline: score the fixed BASELINE genome on both splits and
    write the numbers to results/. Run this once per model before evolving."""
    be = make_backend(a.backend, a.model)
    out = {"model": getattr(be, "model", a.backend), "genome": json.loads(BASELINE.to_json())}
    for split in ("train", "heldout"):
        out[split] = [evaluate(be, BASELINE, get_tasks(split), a.workers).summary() for _ in range(a.repeats)]
    out["usage"] = getattr(be, "usage", None)
    Path("results").mkdir(exist_ok=True)
    path = Path("results") / f"baseline_{out['model']}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "genome"}, indent=2))
    print("wrote", path)
    return 0


def cmd_evolve(a):
    from .evolve import evolve
    be = make_backend(a.backend, a.model)
    genome, report = evolve(be, load_genome(a.genome), generations=a.generations, min_gain=a.min_gain,
                            workers=a.workers, promote_to=CURRENT if a.promote else None)
    print(json.dumps(report, indent=2))
    print("usage:", getattr(be, "usage", None))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="agent7")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "eval", "baseline", "evolve"):
        s = sub.add_parser(name)
        s.add_argument("--workers", type=int, default=4)
        if name != "validate":
            s.add_argument("--backend", default="anthropic")
            s.add_argument("--model", default=None)
        if name in ("eval", "evolve"):
            s.add_argument("--genome", default="current", help="baseline | current | path to json")
    sub.choices["eval"].add_argument("--split", default="train", choices=["train", "heldout", "all"])
    sub.choices["baseline"].add_argument("--repeats", type=int, default=2)
    ev = sub.choices["evolve"]
    ev.add_argument("--generations", type=int, default=5)
    ev.add_argument("--min-gain", type=int, default=2)
    ev.add_argument("--no-promote", dest="promote", action="store_false")
    a = p.parse_args(argv)
    return {"validate": cmd_validate, "eval": cmd_eval, "baseline": cmd_baseline, "evolve": cmd_evolve}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
