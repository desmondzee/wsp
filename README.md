# wsp

> "No human is involved in writing this code because there are a lot of weights (typical networks might have millions), and coding directly in weights is kind of hard (I tried)."
> — Andrej Karpathy, [Software 2.0](https://karpathy.medium.com/software-2-0-a64152b37c35) (2017)

An agent reads a neural network's weights, works out what each part computes, and edits them directly to meet a goal, checking its work with a fixed budget of forward passes. No optimiser, no training loop. [Weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi et al., 2026) in the style of [autoresearch](https://github.com/karpathy/autoresearch).

![progress](progress.png)

## Files

- `harness.py` — loads the task, meters forward passes, scores edits, runs the held-out evaluation. Not modified.
- `tools.py` — read and edit weights of any architecture. Not modified.
- `edit.py` — the file the agent edits: starting weights in, edited weights out.
- `program.md` — agent instructions. Edited by the human.

## Tasks

Tasks live outside the repo. Point `WSP_TASK` at a directory with:

- `task.md` — the goal, what inputs and outputs mean, where the weights came from, what one forward pass returns, what must not break.
- `task.py` — defines `BUDGET`, `REPEATS`, `SCORE_PASSES`, and `weights()`, `reference()` (optional), `model(w)`, `forward(model, **kw)`, `score(model)`, `heldout(model)`.

`score` returns a dict with `score` (higher is better). The held-out evaluation must not share inputs with anything the agent can see. A task that needs extra packages lists them in `task.md`.

## Quick start

    uv sync
    export WSP_TASK=/path/to/task
    uv run harness.py check
    uv run harness.py baseline

Then point your agent at `program.md`:

    Hi have a look at program.md and let's kick off a new run! let's do the setup first.

`uv run harness.py heldout` is for the human, on checkpoints.

## Citation

This repo builds on:

    @misc{takehi2026weightspace,
      title        = {{Sparks of Weight-Space Programming: LLMs can natively read \& write neural network weights}},
      author       = {Takehi, Rikiya and Teoh, Jayden and Zhang, Alex and Khattab, Omar},
      year         = {2026},
      month        = nov,
      howpublished = {\url{https://rikiyat.github.io/blog/weight-space-programming/}},
      note         = {Blog post}
    }
