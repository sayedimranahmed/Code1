"""Score a genome on a set of tasks against the hidden tests."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .agent import DeveloperAgent
from .sandbox import run_tests


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    n_passed: int
    n_total: int
    load: str
    fail_kinds: tuple
    code: str
    trace: dict


@dataclass
class EvalResult:
    results: list

    @property
    def score(self):
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def n_passed(self):
        return sum(r.passed for r in self.results)

    def failures(self):
        return [r for r in self.results if not r.passed]

    def summary(self):
        return {"score": round(self.score, 4), "passed": self.n_passed, "total": len(self.results),
                "failed": [r.task_id for r in self.failures()]}


def evaluate(backend, genome, tasks, workers=4):
    agent = DeveloperAgent(backend, genome)

    def one(task):
        try:
            code, trace = agent.solve(task)
        except Exception as e:  # backend failure counts as a failed task, and is logged
            code, trace = "", {"error": f"{type(e).__name__}: {e}"[:300]}
        res = run_tests(code, task.entry, task.tests)
        kinds = tuple(sorted({r.get("kind", "") for r in res["results"] if not r["ok"]}))
        return TaskResult(task.id, res["passed"], res["n_passed"], res["n_total"], res["load"], kinds, code, trace)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return EvalResult(list(ex.map(one, tasks)))
