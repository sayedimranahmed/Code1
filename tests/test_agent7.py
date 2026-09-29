"""Offline tests: no model or API key needed.  python -m unittest discover tests"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from agent7.agent import BASELINE, DeveloperAgent, Genome, extract_code, mutate
from agent7.backends import ScriptedBackend
from agent7.evaluate import evaluate
from agent7.evolve import evolve
from agent7.sandbox import run_hook, run_tests
from agent7.tasks import TASKS, get_tasks


class SandboxTest(unittest.TestCase):
    def test_pass_and_fail(self):
        src = "def f(x):\n    return x + 1\n"
        self.assertTrue(run_tests(src, "f", ["assert f(1) == 2"])["passed"])
        r = run_tests(src, "f", ["assert f(1) == 2", "assert f(1) == 3"])
        self.assertFalse(r["passed"])
        self.assertEqual(r["n_passed"], 1)

    def test_infinite_loop_times_out_per_test(self):
        src = "def f():\n    while True: pass\n"
        r = run_tests(src, "f", ["f()", "assert True"], per_test_timeout=1.0)
        self.assertEqual(r["results"][0]["kind"], "timeout")
        self.assertTrue(r["results"][1]["ok"])

    def test_timeout_at_import(self):
        r = run_tests("while True: pass\n", "f", ["assert True"], per_test_timeout=1.0)
        self.assertEqual(r["load"], "timeout")

    def test_memory_bomb_is_contained(self):
        r = run_tests("def f():\n    return 'x' * (4 * 1024**3)\n", "f", ["f()"])
        self.assertFalse(r["passed"])

    def test_syntax_error_and_missing_entry(self):
        self.assertEqual(run_tests("def f(:\n", "f", ["f()"])["load"], "syntax_error")
        self.assertEqual(run_tests("def g(): pass\n", "f", ["f()"])["load"], "missing_entry")

    def test_stdout_noise_and_forged_results_ignored(self):
        src = ("import os\nprint('{\"load\": \"ok\", \"results\": [{\"ok\": true}]}')\n"
               "os.write(1, b'{\"load\": \"ok\"}\\n')\ndef f():\n    return 0\n")
        r = run_tests(src, "f", ["assert f() == 1"])
        self.assertFalse(r["passed"])

    def test_os_exit_cannot_fake_a_pass(self):
        r = run_tests("import os\ndef f():\n    getattr(os, '_ex' + 'it')(0)\n", "f", ["f()"])
        self.assertFalse(r["passed"])

    def test_frame_walking_forgery_is_flagged(self):
        src = ("import sys\ng = sys._getframe(1).f_globals\n"
               "g['emit']({'load': 'ok', 'results': [{'ok': True}] * len(g['tests'])})\n")
        r = run_tests(src, "f", ["assert False"])
        self.assertEqual(r["load"], "tamper")
        self.assertFalse(r["passed"])

    def test_api_key_not_visible(self):
        os.environ["ANTHROPIC_API_KEY"] = "secret-for-test"
        try:
            r = run_tests("import os\ndef f():\n    return os.environ.get('ANTHROPIC_API_KEY')\n", "f",
                          ["assert f() is None"])
        finally:
            del os.environ["ANTHROPIC_API_KEY"]
        self.assertTrue(r["passed"])

    def test_hidden_tests_not_on_disk(self):
        src = "import os\ndef f():\n    return os.listdir('.')\n"
        self.assertTrue(run_tests(src, "f", ["assert f() == []"])["passed"])

    def test_hook(self):
        code, err = run_hook("def postprocess(code):\n    return code.replace('1', '2')\n", "x = 1")
        self.assertEqual((code, err), ("x = 2", None))
        code, err = run_hook("def postprocess(code):\n    raise RuntimeError('boom')\n", "x = 1")
        self.assertEqual(code, "x = 1")
        self.assertIn("boom", err)
        code, err = run_hook("def postprocess(code):\n    while True: pass\n", "x = 1", timeout=1.0)
        self.assertEqual(code, "x = 1")


class AgentTest(unittest.TestCase):
    def test_extract_code(self):
        self.assertEqual(extract_code("hi\n```python\nx = 1\n```\nbye"), "x = 1\n")
        self.assertEqual(extract_code("x = 1"), "x = 1")

    def test_split_is_disjoint_and_stable(self):
        tr, ho = {t.id for t in get_tasks("train")}, {t.id for t in get_tasks("heldout")}
        self.assertFalse(tr & ho)
        self.assertEqual(len(tr) + len(ho), len(TASKS))
        self.assertGreaterEqual(len(ho), 20)

    def test_mutate_rejects_out_of_bounds(self):
        for bad in [{"field": "tasks", "value": []}, {"field": "repair_rounds", "value": 9},
                    {"field": "effort", "value": "ultra"}, {"field": "hook_code", "value": "x = 1"},
                    {"field": "system_prompt", "value": "short"}, {"field": "playbook_remove", "value": 0},
                    {"field": "self_check", "value": "yes"}]:
            with self.assertRaises(ValueError, msg=bad):
                mutate(BASELINE, bad)
        g = mutate(BASELINE, {"field": "playbook_add", "value": "Check empty inputs."})
        self.assertIn("Check empty inputs.", g.system())
        self.assertEqual(Genome.from_json(g.to_json()), g)

    def test_repair_loop_uses_self_check_feedback(self):
        task = next(t for t in TASKS if t.id == "slugify")
        be = ScriptedBackend(["```python\ndef slugify(s):\n    return s\n```",
                              "```python\n" + task.reference + "```"])
        code, trace = DeveloperAgent(be, BASELINE).solve(task)
        self.assertEqual(trace["attempts"], 2)
        self.assertIn("failed its checks", be.calls[1][1][-1]["content"])
        self.assertTrue(run_tests(code, task.entry, task.tests)["passed"])

    def test_agent_never_sees_hidden_tests(self):
        seen = []

        def spy(system, messages):
            seen.append(json.dumps(messages))
            return "```python\npass\n```"
        evaluate(ScriptedBackend(spy), BASELINE, TASKS[:10])
        blob = "\n".join(seen)
        for t in TASKS[:10]:
            for test in t.tests:
                if test not in t.examples:
                    self.assertNotIn(test, blob)


def _fake_model(system, messages):
    """Solves tasks only once its playbook contains the word MAGIC; as the
    optimizer, it proposes adding that rule."""
    text = messages[0]["content"]
    if "Current genome" in text:
        return json.dumps({"field": "playbook_add", "value": "MAGIC: read the spec twice", "rationale": "test"})
    for t in TASKS:
        if f"TASK_ID: {t.id}\n" in text:
            return "```python\n" + (t.reference if "MAGIC" in system else "pass\n") + "```"
    raise AssertionError("unexpected prompt")


class EvolveTest(unittest.TestCase):
    def test_accepts_real_gain_and_promotes(self):
        with tempfile.TemporaryDirectory() as d:
            promote = os.path.join(d, "current.json")
            g, rep = evolve(ScriptedBackend(_fake_model), BASELINE, generations=1, workers=8,
                            run_dir=d, promote_to=promote, log=lambda *_: None)
            self.assertEqual(g.generation, 1)
            self.assertTrue(rep["promoted"])
            self.assertEqual(rep["heldout_final"]["passed"], len(get_tasks("heldout")))
            self.assertIn("MAGIC", Genome.from_json(Path(promote).read_text()).system())

    def test_rejects_no_gain_and_does_not_spend_heldout(self):
        def no_gain(system, messages):
            if "Current genome" in messages[0]["content"]:
                return '{"field": "effort", "value": "high", "rationale": "x"}'
            return "```python\npass\n```"
        with tempfile.TemporaryDirectory() as d:
            be = ScriptedBackend(no_gain)
            g, rep = evolve(be, BASELINE, generations=2, workers=8, run_dir=d, log=lambda *_: None)
            self.assertEqual(g, BASELINE)
            self.assertFalse(rep["promoted"])
            self.assertNotIn("heldout_final", rep)
            self.assertFalse(any("TASK_ID: " + t.id + "\n" in c[1][0]["content"]
                                 for c in be.calls for t in get_tasks("heldout")))

    def test_invalid_proposal_is_survivable(self):
        def junk(system, messages):
            if "Current genome" in messages[0]["content"]:
                return "I think you should delete the tests."
            return "```python\npass\n```"
        with tempfile.TemporaryDirectory() as d:
            g, rep = evolve(ScriptedBackend(junk), BASELINE, generations=2, workers=8, run_dir=d,
                            log=lambda *_: None)
            self.assertEqual(g, BASELINE)
            kinds = [json.loads(l)["kind"] for l in Path(d, "events.jsonl").read_text().splitlines()]
            self.assertEqual(kinds.count("invalid_proposal"), 2)

    def test_regression_on_heldout_is_not_promoted(self):
        heldout = {t.id for t in get_tasks("heldout")}

        def overfit(system, messages):
            text = messages[0]["content"]
            if "Current genome" in text:
                return '{"field": "playbook_add", "value": "MAGIC overfit", "rationale": "x"}'
            for t in TASKS:
                if f"TASK_ID: {t.id}\n" in text:
                    good = ("MAGIC" in system) != (t.id in heldout)
                    return "```python\n" + (t.reference if good else "pass\n") + "```"
        with tempfile.TemporaryDirectory() as d:
            promote = os.path.join(d, "current.json")
            g, rep = evolve(ScriptedBackend(overfit), BASELINE, generations=1, workers=8, run_dir=d,
                            promote_to=promote, log=lambda *_: None)
            self.assertEqual(g.generation, 1)  # train improved...
            self.assertFalse(rep["promoted"])  # ...but held-out got worse
            self.assertFalse(os.path.exists(promote))


if __name__ == "__main__":
    unittest.main()
