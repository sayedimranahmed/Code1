"""Mutation testing for the benchmark itself: how often do the hidden tests
catch a small bug planted in the reference solution? A low kill rate means a
task's tests are too weak to tell a correct solution from a subtly wrong one,
which would let the optimizer "improve" by getting lucky.
"""
import ast
import random

from .sandbox import run_tests
from .tasks import get_tasks

_SWAPS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
          ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Add: ast.Sub, ast.Sub: ast.Add,
          ast.Mult: ast.FloorDiv, ast.And: ast.Or, ast.Or: ast.And}


def _sites(tree):
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            sites += [(node, "cmp", i) for i, op in enumerate(node.ops) if type(op) in _SWAPS]
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in _SWAPS:
            sites.append((node, "op", None))
        elif isinstance(node, ast.BoolOp):
            sites.append((node, "bool", None))
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            sites.append((node, "int", None))
    return sites


def mutants(src, n, seed=0):
    base = ast.parse(src)
    count = len(_sites(base))
    rng = random.Random(seed)
    out = []
    for idx in rng.sample(range(count), min(n, count)):
        tree = ast.parse(src)
        node, kind, i = _sites(tree)[idx]
        if kind == "cmp":
            node.ops[i] = _SWAPS[type(node.ops[i])]()
        elif kind in ("op", "bool"):
            node.op = _SWAPS[type(node.op)]()
        else:
            node.value += 1
        out.append(ast.unparse(tree))
    return out


def kill_rates(per_task=6):
    rows = []
    for t in get_tasks():
        ms = mutants(t.reference, per_task)
        killed = sum(not run_tests(m, t.entry, t.tests, per_test_timeout=3.0)["passed"] for m in ms)
        rows.append((t.id, killed, len(ms)))
    return rows


if __name__ == "__main__":
    rows = kill_rates()
    tk = sum(r[1] for r in rows)
    tn = sum(r[2] for r in rows)
    for tid, k, n in rows:
        if n and k < n:
            print(f"{tid:22s} {k}/{n} mutants killed")
    print(f"overall: {tk}/{tn} = {tk / tn:.0%}")
