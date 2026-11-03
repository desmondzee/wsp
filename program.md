# wsp

This is an experiment to have the LLM program neural network weights directly.

## Setup

To set up a new run, work with the user to:

1. **Agree on a run tag**: based on the task and today's date (e.g. `shakespeare-nov3`). The branch `wsp/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b wsp/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `harness.py` — forward-pass budget, scoring, held-out evaluation. Do not modify.
   - `tools.py` — helpers for reading and editing weights. Do not modify.
   - `edit.py` — the file you modify. Takes the starting weights, returns the edited weights.
   - The task's `task.md` (in `$WSP_TASK`) — the goal, what inputs and outputs mean, where the weights came from, and what one forward pass gives you.
4. **Verify the task loads**: `uv run harness.py check`. If it fails, tell the human.
5. **Initialize results.tsv** with just the header row.
6. **Confirm and go.**

## Experimentation

You change what the network does by reading its weights, working out what its parts compute, and writing new weights directly. There is no training loop. `tools.py` gives you the network's structure, weight statistics, products of weight matrices along paths, comparison against reference weights, and edit operations; `harness.start()`, `harness.reference()` and `harness.edited()` give you the starting, reference and currently edited weights.

**What you CAN do:**
- Modify `edit.py`. Anything in weight space is fair game: change values, add or remove directions, prune units, add new paths, rewrite layers.
- Read and analyse the weights freely.
- Run the network through `harness.run(w, capture=[...], ...)`, which returns the task's output and the activations of the named modules. This spends one forward pass from the task's budget.

**What you CANNOT do:**
- Set weights by minimising an error on data: no gradient steps, no regression or least-squares fits, no automated search over weights. You may measure activations and use what you learn to choose an edit, but the edit must come from your reasoning.
- Run the network outside `harness.run`, including by re-implementing it.
- Modify `harness.py`, `tools.py`, the task, or the ledger, or run the held-out evaluation.
- Change tensor names or shapes.

**The goal: the best task score without breaking what already works.** The task defines both.

**Noise**: the baseline scores the starting weights several times to set a noise floor. Gains below it don't count.

**Simplicity criterion**: a smaller edit you can explain beats a larger one with the same score.

**The first run**: understand the network before changing it, then record the baseline with `uv run harness.py baseline > run.log 2>&1`.

## Output format

`uv run harness.py score` prints a summary like this:

```
---
score:            0.412
noise:            0.031
fraction_changed: 0.0016
passes_used:      14
passes_left:      36
```

The task may print more metrics above `noise`. Extract the key ones with `grep "^score:\|^fraction_changed:\|^passes_left:" run.log`.

## Logging results

Log every scored edit to `results.tsv` (tab-separated). Do not commit it.

```
commit	score	fraction_changed	passes_used	status	description
```

1. git commit hash (short, 7 chars)
2. score (0.000000 for crashes)
3. fraction of weights changed
4. passes used so far
5. status: `keep`, `discard`, or `crash`
6. what the edit changes and why

## The experiment loop

LOOP until the budget is spent:

1. Form a hypothesis about which part of the network is responsible.
2. Edit `edit.py` to test it, predicting what will change.
3. git commit
4. Score it: `uv run harness.py score > run.log 2>&1`
5. Record the result in the tsv.
6. If the score beats the parent by more than the noise floor and nothing broke, keep the commit. Otherwise git reset.

**Never stop and never wait**: once the loop has begun, don't pause to ask the human. If something hangs, kill it and move on. If you run out of ideas, re-read the weights.
