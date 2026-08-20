#!/usr/bin/env python3
"""Select a trajectory-quality arm only after both paired gates are known."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


def read_gate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    comparisons = value.get("comparisons")
    if not isinstance(comparisons, list) or {
        item.get("label") for item in comparisons
    } != {"checkpoint-500", "final-1000"}:
        raise ValueError(f"{path}: invalid paired comparison labels")
    gate = value.get("gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("passed"), bool):
        raise ValueError(f"{path}: missing gate result")
    return value


def final_metrics(value: dict[str, Any]) -> tuple[float, float]:
    final = next(
        item for item in value["comparisons"] if item["label"] == "final-1000"
    )
    return float(final["mrr"]["delta"]), float(final["recall_at_1"]["delta"])


def select(arm_b: dict[str, Any], arm_c: dict[str, Any]) -> dict[str, Any]:
    for value in (arm_b, arm_c):
        if value.get("bootstrap_samples", 0) < 1000:
            raise ValueError("gate must use at least 1000 bootstrap samples")
    raw_b = {
        item["label"]: item["raw_eval"] for item in arm_b["comparisons"]
    }
    raw_c = {
        item["label"]: item["raw_eval"] for item in arm_c["comparisons"]
    }
    if raw_b != raw_c:
        raise ValueError("B and C gates do not share identical Arm A evaluations")

    values = {"B": arm_b, "C": arm_c}
    passed = [arm for arm, value in values.items() if value["gate"]["passed"]]
    selected = None
    if passed:
        # Higher final MRR, then R@1.  Exact ties prefer C because it retains
        # every official row and changes only the documented soft weight.
        selected = max(
            passed,
            key=lambda arm: (*final_metrics(values[arm]), arm == "C"),
        )
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "predeclared_selection": [
            "arm gate passed",
            "higher final-1000 MRR delta",
            "higher final-1000 R@1 delta",
            "exact tie prefers C because no rows are deleted",
        ],
        "arm_b_passed": arm_b["gate"]["passed"],
        "arm_c_passed": arm_c["gate"]["passed"],
        "arm_b_final_delta": {
            "mrr": final_metrics(arm_b)[0],
            "recall_at_1": final_metrics(arm_b)[1],
        },
        "arm_c_final_delta": {
            "mrr": final_metrics(arm_c)[0],
            "recall_at_1": final_metrics(arm_c)[1],
        },
        "full_epoch_authorized": selected is not None,
        "selected_arm": selected,
        "locked_test_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-b-gate", required=True, type=Path)
    parser.add_argument("--arm-c-gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = select(read_gate(args.arm_b_gate), read_gate(args.arm_c_gate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.tmp.", dir=args.output.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, args.output)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
