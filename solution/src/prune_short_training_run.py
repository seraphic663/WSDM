#!/usr/bin/env python3
"""Prune redundant optimizer state after a completed short diagnostic run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


INFERENCE_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
}
TRAINING_STATE_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "scaler.pt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_eval(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("rows") != 1500:
        raise ValueError(f"{path}: expected 1500 evaluation rows")
    if "details" not in value or len(value["details"]) != 1500:
        raise ValueError(f"{path}: missing per-query evaluation evidence")
    if "test" in path.name.casefold() or "test" in str(value.get("input", "")).casefold():
        raise ValueError(f"{path}: test-like evaluation evidence is forbidden")
    return value


def checkpoint_inventory(path: Path) -> dict[str, int]:
    result = {}
    for item in sorted(path.iterdir()):
        if item.is_symlink() or not item.is_file():
            raise ValueError(f"unexpected checkpoint entry: {item}")
        result[item.name] = item.stat().st_size
    return result


def prune(
    output_dir: Path,
    eval_500: Path,
    eval_1000: Path,
    *,
    allowed_root: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if allowed_root is not None:
        allowed_root = allowed_root.resolve()
        if output_dir == allowed_root or not output_dir.is_relative_to(allowed_root):
            raise ValueError(f"output directory is outside allowed root: {output_dir}")
    manifest_path = output_dir / "SHORT_RUN_PRUNE_MANIFEST.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError(f"invalid output directory: {output_dir}")
    if not (output_dir / "COMPLETED").is_file():
        raise ValueError("short run is not completed")
    if (output_dir / "RUNNING").exists() or (output_dir / "FAILED").exists():
        raise ValueError("short run has active/failure markers")

    checkpoint_500 = output_dir / "checkpoint-500"
    checkpoint_1000 = output_dir / "checkpoint-1000"
    for checkpoint in (checkpoint_500, checkpoint_1000):
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise ValueError(f"missing non-symlink checkpoint: {checkpoint}")
    eval_500_value = validate_eval(eval_500)
    eval_1000_value = validate_eval(eval_1000)

    root_weight = output_dir / "model.safetensors"
    weight_500 = checkpoint_500 / "model.safetensors"
    weight_1000 = checkpoint_1000 / "model.safetensors"
    root_sha = sha256_file(root_weight)
    sha_500 = sha256_file(weight_500)
    sha_1000 = sha256_file(weight_1000)
    if root_sha != sha_1000:
        raise ValueError("final root weight differs from checkpoint-1000")

    inventory_500 = checkpoint_inventory(checkpoint_500)
    inventory_1000 = checkpoint_inventory(checkpoint_1000)
    allowed = INFERENCE_FILES | TRAINING_STATE_FILES
    unknown_500 = {
        name for name in inventory_500
        if name not in allowed and not (name.startswith("rng_state_") and name.endswith(".pth"))
    }
    unknown_1000 = {
        name for name in inventory_1000
        if name not in allowed and not (name.startswith("rng_state_") and name.endswith(".pth"))
    }
    if unknown_500 or unknown_1000:
        raise ValueError(f"unknown checkpoint files: {sorted(unknown_500 | unknown_1000)}")
    missing_inference = INFERENCE_FILES - set(inventory_500)
    if missing_inference:
        raise ValueError(f"checkpoint-500 lacks inference files: {sorted(missing_inference)}")

    removed_500 = {}
    for name, size in inventory_500.items():
        if name in TRAINING_STATE_FILES or (name.startswith("rng_state_") and name.endswith(".pth")):
            path = checkpoint_500 / name
            path.unlink()
            removed_500[name] = size
    shutil.rmtree(checkpoint_1000)

    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "contract": {
            "reason": "preserve evaluated short-run inference evidence while removing redundant recovery state",
            "checkpoint_500_inference_preserved": True,
            "checkpoint_1000_replaced_by_identical_final_root": True,
            "locked_test_used": False,
        },
        "output_dir": str(output_dir),
        "weights": {
            "checkpoint_500_sha256": sha_500,
            "checkpoint_1000_sha256": sha_1000,
            "final_root_sha256": root_sha,
        },
        "eval": {
            "step_500": {
                "path": str(eval_500.resolve()),
                "sha256": sha256_file(eval_500),
                "rows": eval_500_value["rows"],
            },
            "step_1000": {
                "path": str(eval_1000.resolve()),
                "sha256": sha256_file(eval_1000),
                "rows": eval_1000_value["rows"],
            },
        },
        "before_bytes": {
            "checkpoint_500": sum(inventory_500.values()),
            "checkpoint_1000": sum(inventory_1000.values()),
        },
        "removed": {
            "checkpoint_500_training_state": removed_500,
            "checkpoint_1000_all_files": inventory_1000,
        },
        "after": {
            "checkpoint_500_files": sorted(item.name for item in checkpoint_500.iterdir()),
            "checkpoint_1000_exists": checkpoint_1000.exists(),
        },
    }
    fd, temporary = tempfile.mkstemp(prefix=f".{manifest_path.name}.", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, manifest_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--eval-500", required=True, type=Path)
    parser.add_argument("--eval-1000", required=True, type=Path)
    parser.add_argument("--allowed-root", required=True, type=Path)
    args = parser.parse_args()
    result = prune(
        args.output_dir,
        args.eval_500,
        args.eval_1000,
        allowed_root=args.allowed_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
