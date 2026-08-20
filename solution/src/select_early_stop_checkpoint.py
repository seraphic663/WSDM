#!/usr/bin/env python3
"""Select an early-stop checkpoint from ordered dev-only evaluation files."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path


REQUIRED_METRICS = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr")


def load_eval(path: Path, expected_rows: int) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("rows") != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, got {value.get('rows')}")
    input_name = Path(str(value.get("input", ""))).name.casefold()
    if "test" in input_name:
        raise ValueError(f"{path}: locked test input is forbidden for early-stop selection")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path}: missing metrics")
    for name in REQUIRED_METRICS:
        metric = metrics.get(name)
        if not isinstance(metric, (int, float)) or not math.isfinite(metric):
            raise ValueError(f"{path}: invalid {name}")
    return value


def parse_eval_spec(spec: str) -> tuple[int, Path]:
    step_text, separator, path_text = spec.partition("=")
    if not separator or not step_text.isdigit() or int(step_text) < 1 or not path_text:
        raise ValueError(f"invalid eval spec {spec!r}; expected STEP=PATH")
    return int(step_text), Path(path_text)


def select(
    eval_specs: list[tuple[int, Path]],
    *,
    expected_rows: int = 1500,
    min_steps: int = 2000,
    min_delta: float = 0.001,
    patience: int = 2,
    baseline_eval: Path | None = None,
    guardrail_tolerance: float = 0.005,
) -> dict:
    if min_steps < 1 or min_delta < 0 or patience < 1 or guardrail_tolerance < 0:
        raise ValueError("invalid early-stop parameters")
    steps = [step for step, _ in eval_specs]
    if steps != sorted(set(steps)):
        raise ValueError("eval steps must be unique and strictly increasing")
    if not eval_specs:
        raise ValueError("at least one eval is required")
    baseline = load_eval(baseline_eval, expected_rows) if baseline_eval else None
    history = []
    best_index = 0
    best_mrr = -math.inf
    stale = 0
    stop_step = None
    for index, (step, path) in enumerate(eval_specs):
        result = load_eval(path, expected_rows)
        metrics = {name: float(result["metrics"][name]) for name in REQUIRED_METRICS}
        improved = metrics["mrr"] >= best_mrr + min_delta
        if improved:
            best_index = index
            best_mrr = metrics["mrr"]
            stale = 0
        else:
            stale += 1
        history.append(
            {
                "step": step,
                "eval_path": str(path),
                "model": result.get("model"),
                "input": result.get("input"),
                "metrics": metrics,
                "improved_by_min_delta": improved,
                "stale_evaluations": stale,
            }
        )
        if step >= min_steps and stale >= patience:
            stop_step = step
            break

    selected = history[best_index]
    guardrails = None
    if baseline:
        guardrails = {}
        for name in ("recall_at_1", "recall_at_5", "recall_at_10"):
            delta = selected["metrics"][name] - float(baseline["metrics"][name])
            guardrails[name] = {"delta_vs_m00": delta, "passed": delta >= -guardrail_tolerance}
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "policy": {
            "dataset_role": "dev only; locked test forbidden",
            "primary_metric": "mrr",
            "expected_rows": expected_rows,
            "min_steps": min_steps,
            "min_delta": min_delta,
            "patience": patience,
            "guardrail_tolerance": guardrail_tolerance,
        },
        "selected": selected,
        "stop_triggered_at_step": stop_step,
        "evaluations_consumed": len(history),
        "history": history,
        "guardrails_vs_m00": guardrails,
        "eligible_for_locked_test": bool(
            guardrails is not None and all(item["passed"] for item in guardrails.values())
        ),
    }


def atomic_write(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.rename(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="append", required=True, dest="evals", metavar="STEP=PATH")
    parser.add_argument("--baseline-eval", type=Path)
    parser.add_argument("--expected-rows", type=int, default=1500)
    parser.add_argument("--min-steps", type=int, default=2000)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--guardrail-tolerance", type=float, default=0.005)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = select(
        [parse_eval_spec(spec) for spec in args.evals],
        expected_rows=args.expected_rows,
        min_steps=args.min_steps,
        min_delta=args.min_delta,
        patience=args.patience,
        baseline_eval=args.baseline_eval,
        guardrail_tolerance=args.guardrail_tolerance,
    )
    atomic_write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
