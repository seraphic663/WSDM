#!/usr/bin/env python3
"""Print the newest complete Hugging Face Trainer checkpoint, if any."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
REQUIRED_FILES = (
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
)
MODEL_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def validation_error(path: Path) -> str | None:
    step = checkpoint_step(path)
    if step is None or not path.is_dir():
        return "not a checkpoint-N directory"

    missing = [
        name
        for name in REQUIRED_FILES
        if not (path / name).is_file() or (path / name).stat().st_size == 0
    ]
    if missing:
        return f"missing or empty: {', '.join(missing)}"

    if not any(
        (path / name).is_file() and (path / name).stat().st_size > 0
        for name in MODEL_FILES
    ):
        return "missing model weights"

    try:
        state = json.loads((path / "trainer_state.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"invalid trainer_state.json: {exc}"

    if state.get("global_step") != step:
        return (
            "trainer_state global_step does not match directory: "
            f"{state.get('global_step')} != {step}"
        )
    return None


def valid_checkpoints(output_dir: Path) -> tuple[list[Path], list[str]]:
    valid: list[tuple[int, Path]] = []
    rejected: list[str] = []
    if not output_dir.is_dir():
        return [], rejected

    for path in output_dir.iterdir():
        step = checkpoint_step(path)
        if step is None:
            continue
        error = validation_error(path)
        if error is None:
            valid.append((step, path.resolve()))
        else:
            rejected.append(f"{path}: {error}")
    valid.sort(key=lambda item: item[0])
    return [path for _, path in valid], rejected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--show-rejected",
        action="store_true",
        help="print rejected checkpoint reasons to stderr",
    )
    args = parser.parse_args()

    valid, rejected = valid_checkpoints(args.output_dir)
    if args.show_rejected:
        import sys

        for message in rejected:
            print(message, file=sys.stderr)
    if valid:
        print(valid[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
