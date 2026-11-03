import argparse
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import torch

import edit
import tools

TASK_DIR = Path(os.environ.get("WSP_TASK", "task")).resolve()
LEDGER = Path(os.environ.get("WSP_LEDGER", "ledger.jsonl"))


def _load_task():
    spec = importlib.util.spec_from_file_location("task", TASK_DIR / "task.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


task = _load_task()
BUDGET = task.BUDGET
REPEATS = getattr(task, "REPEATS", 5)


def start():
    return {k: v.detach().clone() for k, v in task.weights().items()}


def reference():
    return task.reference() if hasattr(task, "reference") else None


def edited():
    w0 = start()
    w = edit.edit({k: v.clone() for k, v in w0.items()})
    assert set(w) == set(w0), "tensor names changed"
    for k in w0:
        assert w[k].shape == w0[k].shape, f"{k}: shape changed"
    return w


def _ledger():
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def _record(entry):
    entry["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def used():
    return sum(e.get("passes", 0) for e in _ledger())


def _spend(n, kind):
    if used() + n > BUDGET:
        raise SystemExit(f"forward-pass budget spent ({used()}/{BUDGET})")
    _record({"kind": kind, "passes": n})


def _hook(acts, name):
    def fn(module, inputs, output):
        out = output if torch.is_tensor(output) else output[0]
        acts.setdefault(name, []).append(out.detach().clone())
    return fn


def run(w, capture=(), **kwargs):
    _spend(1, "run")
    model = task.model(w)
    acts = {}
    handles = [model.get_submodule(name).register_forward_hook(_hook(acts, name)) for name in capture]
    try:
        with torch.no_grad():
            out = task.forward(model, **kwargs)
    finally:
        for h in handles:
            h.remove()
    return (out, acts) if capture else out


def _score(w, kind):
    passes = getattr(task, "SCORE_PASSES", 1)
    if used() + passes > BUDGET:
        raise SystemExit(f"forward-pass budget spent ({used()}/{BUDGET})")
    with torch.no_grad():
        result = task.score(task.model(w))
    _record({"kind": kind, "passes": passes, "result": result})
    return result


def noise():
    base = [e["result"]["score"] for e in _ledger() if e["kind"] == "baseline"]
    return statistics.stdev(base) if len(base) > 1 else None


def _print(result, w=None):
    print("---")
    for k, v in result.items():
        print(f"{k + ':':<18}{round(v, 6) if isinstance(v, float) else v}")
    n = noise()
    print(f"{'noise:':<18}{n if n is None else round(n, 6)}")
    if w is not None:
        print(f"{'fraction_changed:':<18}{tools.diff(start(), w)['fraction']:.6g}")
    print(f"{'passes_used:':<18}{used()}")
    print(f"{'passes_left:':<18}{BUDGET - used()}")


def cmd_check():
    w = edited()
    print(task.model(start()))
    print(tools.structure(w))
    print(f"task: {TASK_DIR}")
    print(f"budget: {BUDGET}, used: {used()}, left: {BUDGET - used()}")


def cmd_baseline():
    w = start()
    results = [_score(w, "baseline") for _ in range(REPEATS)]
    mean = statistics.mean(r["score"] for r in results)
    _print({"score": round(mean, 6)})


def cmd_score():
    w = edited()
    _print(_score(w, "score"), w)


def cmd_heldout():
    w = edited()
    with torch.no_grad():
        result = task.heldout(task.model(w))
    _record({"kind": "heldout", "passes": 0, "result": result})
    _print(result, w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check", "baseline", "score", "heldout"])
    {"check": cmd_check, "baseline": cmd_baseline, "score": cmd_score, "heldout": cmd_heldout}[parser.parse_args().command]()
