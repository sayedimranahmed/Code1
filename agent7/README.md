# agent7: a developer agent that edits itself and has to prove it

A coding agent that proposes changes to its own configuration and keeps a
change only when a scored benchmark says it got better. The evaluation came
first and the agent second, because a self-improving agent with a weak
evaluator just learns to fool the evaluator.

## What's here

| Piece | File | What it does |
|---|---|---|
| Benchmark | `tasks.py` | 54 Python tasks (32 train, 22 held-out). The agent sees each spec and a visible example; the hidden tests are never sent to the model. |
| Sandbox | `sandbox.py` | Runs agent-written code in a fresh `python -I` subprocess with CPU/memory/file limits, per-test timeouts, a scrubbed environment and no test files on disk. It also flags attempts to tamper with the harness. |
| Agent | `agent.py` | Writes a solution, runs it against the visible examples (and optionally tests it wrote itself), then repairs it. Everything it may change about itself is a `Genome`. |
| Self-improvement | `evolve.py` | Each generation: read own genome + train failures, propose one change, re-score on train. Keep it only if it gains at least 2 tasks on two separate runs. At the end, score once on held-out, and promote to `genome/current.json` only if held-out did not get worse. |
| Benchmark audit | `mutants.py` | Plants small bugs in the reference solutions and checks the hidden tests catch them. Currently 213/234 (91%); most survivors are equivalent mutants. |

What the agent can change about itself: its system prompt, a playbook of
learned rules, whether it self-checks, how many repair rounds it gets,
whether it writes its own edge-case tests, its effort level, and a Python
`postprocess(code)` hook that runs sandboxed on every answer.

What it cannot change: the tasks, the hidden tests, the sandbox, the scoring,
or the acceptance rule. Those live outside the genome on purpose.

## Run it

```bash
python -m unittest discover -s tests            # offline, no key needed
python -m agent7 validate                       # benchmark sanity: oracle must score 54/54
python -m agent7.mutants                        # how strong are the hidden tests?

pip install anthropic                           # needed for a real model
export ANTHROPIC_API_KEY=...                    # never commit this
python -m agent7 baseline                       # freeze baseline scores -> results/
python -m agent7 evolve --generations 5         # self-improvement run -> runs/<timestamp>/
python -m agent7 eval --genome current --split heldout
```

Defaults: model `claude-opus-5-5` (override with `--model` or `AGENT7_MODEL`),
effort `medium`, 4 parallel workers. Every run writes `runs/<ts>/events.jsonl`
listing each proposal, its scores and the verdict, and `usage` prints the tokens
spent.

## Known limits (read before trusting a number)

- **No model score exists yet.** Everything above was verified with an oracle
  and scripted backends. Whether the loop improves a real model is untested
  until a key is supplied.
- **Headroom.** These are function-level tasks. A current frontier model may
  score 28+/32 on the baseline, leaving too little room for the loop to show
  a real gain above noise. If the baseline comes back that high, the next step
  is harder tasks (multi-file repos, bug fixing in existing code), not more
  generations.
- **Noise.** 32 train tasks is a coarse ruler. Gains are required to hold on two
  runs, and held-out gets one look per run, but a +2 on 32 is still weak
  evidence. Treat held-out, not train, as the headline.
- **The sandbox is not a security boundary.** It stops runaway code, not a
  hostile program: there is no network isolation and the tamper check is a
  regex. Before this runs unattended anywhere persistent (host01), put it in a
  container or nsjail with networking off.
- **"Experienced developer" is measured narrowly.** This scores correctness on
  well-specified functions. It does not measure design, reading unfamiliar
  code, or judgement.
