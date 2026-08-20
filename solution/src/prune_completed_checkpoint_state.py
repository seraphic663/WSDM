#!/usr/bin/env python3
"""Remove rebuildable recovery state from completed training checkpoints.

The final model and every checkpoint's inference files and trainer_state.json
are preserved. Only optimizer/scheduler/RNG/scaler/training-arguments files are
removed after lifecycle, path, and checkpoint-shape validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


REMOVABLE_STATE_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "training_args.bin",
}
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_removable_state(name: str) -> bool:
    return name in REMOVABLE_STATE_FILES or (
        name.startswith("rng_state_") and name.endswith(".pth")
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _checkpoint_dirs(checkpoint_root: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for item in checkpoint_root.iterdir():
        match = CHECKPOINT_PATTERN.fullmatch(item.name)
        if match:
            if item.is_symlink() or not item.is_dir():
                raise ValueError(f"invalid checkpoint directory: {item}")
            checkpoints.append((int(match.group(1)), item))
    checkpoints.sort()
    if not checkpoints:
        raise ValueError(f"no checkpoint directories found in {checkpoint_root}")
    return checkpoints


def _inside(path: Path, root: Path) -> bool:
    return path != root and path.is_relative_to(root)


def prune_completed_run(
    run_dir: Path,
    *,
    allowed_root: Path,
    extra_checkpoint_roots: list[Path] | None = None,
    allow_superseded_failure: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    original_run_dir = run_dir
    run_dir = run_dir.resolve()
    allowed_root = allowed_root.resolve()
    if not _inside(run_dir, allowed_root):
        raise ValueError(f"run directory is outside allowed root: {run_dir}")
    if original_run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError(f"invalid run directory: {run_dir}")
    completed_marker = run_dir / "COMPLETED"
    failed_marker = run_dir / "FAILED"
    if not completed_marker.is_file():
        raise ValueError(f"run lacks COMPLETED marker: {run_dir}")
    if (run_dir / "RUNNING").exists():
        raise ValueError(f"run has active marker: {run_dir}")
    superseded_failure = False
    if failed_marker.exists():
        if not allow_superseded_failure:
            raise ValueError(f"run has failure marker: {run_dir}")
        if not failed_marker.is_file():
            raise ValueError(f"invalid failure marker: {failed_marker}")
        if failed_marker.stat().st_mtime_ns >= completed_marker.stat().st_mtime_ns:
            raise ValueError(
                f"failure marker is not older than COMPLETED marker: {run_dir}"
            )
        superseded_failure = True

    manifest_path = run_dir / "CHECKPOINT_STATE_PRUNE_MANIFEST.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("completed") is True:
            return existing
        raise ValueError(f"incomplete existing manifest: {manifest_path}")

    root_weight = run_dir / "model.safetensors"
    if not root_weight.is_file() or root_weight.is_symlink():
        raise ValueError(f"missing final root model: {root_weight}")

    checkpoint_roots = [run_dir]
    for original_root in extra_checkpoint_roots or []:
        resolved_root = original_root.resolve()
        if not _inside(resolved_root, allowed_root):
            raise ValueError(
                f"extra checkpoint root is outside allowed root: {resolved_root}"
            )
        if original_root.is_symlink() or not resolved_root.is_dir():
            raise ValueError(f"invalid extra checkpoint root: {resolved_root}")
        checkpoint_roots.append(resolved_root)
    if len(set(checkpoint_roots)) != len(checkpoint_roots):
        raise ValueError("duplicate checkpoint root")

    checkpoint_values = []
    total_removable = 0
    for checkpoint_root in checkpoint_roots:
        for step, checkpoint in _checkpoint_dirs(checkpoint_root):
            weight = checkpoint / "model.safetensors"
            trainer_state = checkpoint / "trainer_state.json"
            if not weight.is_file() or weight.is_symlink():
                raise ValueError(f"missing checkpoint model: {weight}")
            if not trainer_state.is_file() or trainer_state.is_symlink():
                raise ValueError(f"missing trainer state: {trainer_state}")

            removable = {}
            for item in sorted(checkpoint.iterdir(), key=lambda path: path.name):
                if item.is_symlink():
                    raise ValueError(f"checkpoint symlink is forbidden: {item}")
                if item.is_file() and is_removable_state(item.name):
                    removable[item.name] = item.stat().st_size
            total_removable += sum(removable.values())
            checkpoint_values.append(
                {
                    "step": step,
                    "checkpoint_root": str(checkpoint_root),
                    "path": str(checkpoint),
                    "model_bytes": weight.stat().st_size,
                    "model_sha256": sha256_file(weight),
                    "trainer_state_bytes": trainer_state.stat().st_size,
                    "removable_state": removable,
                }
            )

    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir),
        "contract": {
            "reason": "completed run; preserve inference checkpoints and trainer metadata while removing rebuildable recovery state",
            "final_root_preserved": True,
            "all_checkpoint_models_preserved": True,
            "all_trainer_state_preserved": True,
            "checkpoint_directories_removed": False,
            "superseded_failure_accepted": superseded_failure,
        },
        "lifecycle": {
            "completed_marker": str(completed_marker),
            "completed_marker_mtime_ns": completed_marker.stat().st_mtime_ns,
            "failed_marker": str(failed_marker) if failed_marker.exists() else None,
            "failed_marker_mtime_ns": (
                failed_marker.stat().st_mtime_ns if failed_marker.exists() else None
            ),
        },
        "checkpoint_roots": [str(root) for root in checkpoint_roots],
        "final_root": {
            "bytes": root_weight.stat().st_size,
            "sha256": sha256_file(root_weight),
        },
        "checkpoints": checkpoint_values,
        "planned_removed_bytes": total_removable,
        "dry_run": dry_run,
        "completed": False,
    }
    if dry_run:
        return result

    _atomic_json(manifest_path, result)
    removed_bytes = 0
    for checkpoint in checkpoint_values:
        checkpoint_path = Path(checkpoint["path"])
        for name, expected_size in checkpoint["removable_state"].items():
            path = checkpoint_path / name
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"state file changed before deletion: {path}")
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"state file size changed before deletion: {path}: "
                    f"{actual_size} != {expected_size}"
                )
            path.unlink()
            removed_bytes += actual_size

    for checkpoint in checkpoint_values:
        checkpoint_path = Path(checkpoint["path"])
        if not (checkpoint_path / "model.safetensors").is_file():
            raise RuntimeError(f"checkpoint model missing after cleanup: {checkpoint_path}")
        if not (checkpoint_path / "trainer_state.json").is_file():
            raise RuntimeError(f"trainer state missing after cleanup: {checkpoint_path}")
        residual = [
            item.name
            for item in checkpoint_path.iterdir()
            if item.is_file() and is_removable_state(item.name)
        ]
        if residual:
            raise RuntimeError(f"removable state remains in {checkpoint_path}: {residual}")

    result["completed"] = True
    result["completed_at"] = datetime.now().astimezone().isoformat()
    result["removed_bytes"] = removed_bytes
    result["dry_run"] = False
    _atomic_json(manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--allowed-root", required=True, type=Path)
    parser.add_argument(
        "--extra-checkpoint-root",
        action="append",
        default=[],
        type=Path,
        help="additional directory containing checkpoint-N subdirectories",
    )
    parser.add_argument(
        "--allow-superseded-failure",
        action="store_true",
        help="accept FAILED only when its mtime is older than COMPLETED",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = prune_completed_run(
        args.run_dir,
        allowed_root=args.allowed_root,
        extra_checkpoint_roots=args.extra_checkpoint_root,
        allow_superseded_failure=args.allow_superseded_failure,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
