#!/usr/bin/env python3
"""Delete an explicit, hash-locked set of rebuildable large artifacts.

This is intentionally narrower than a generic cleanup command: every target
must be declared in a JSON plan with its exact relative path, size, SHA-256,
reason, and at least one surviving rebuild-evidence path. The audit manifest is
written before deletion and finalized only after all targets are confirmed
absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _resolve_relative(repo_root: Path, relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must be a safe relative path: {relative}")
    resolved = (repo_root / candidate).resolve()
    if resolved == repo_root or not resolved.is_relative_to(repo_root):
        raise ValueError(f"{label} escapes repository root: {relative}")
    return resolved


def remove_rebuildable_artifacts(
    *,
    repo_root: Path,
    allowed_root: Path,
    plan_path: Path,
    manifest_path: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    allowed_root = allowed_root.resolve()
    plan_path = plan_path.resolve()
    manifest_path = manifest_path.resolve()
    if not repo_root.is_dir():
        raise ValueError(f"invalid repository root: {repo_root}")
    if allowed_root == repo_root or not allowed_root.is_relative_to(repo_root):
        raise ValueError(f"allowed root must be a repository subdirectory: {allowed_root}")
    if not plan_path.is_file() or plan_path.is_symlink():
        raise ValueError(f"invalid cleanup plan: {plan_path}")
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("completed") is True:
            return existing
        raise ValueError(f"incomplete existing cleanup manifest: {manifest_path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    targets = plan.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("cleanup plan must contain a non-empty targets list")

    checked: list[dict[str, Any]] = []
    seen: set[Path] = set()
    total_bytes = 0
    for index, item in enumerate(targets):
        if not isinstance(item, dict):
            raise ValueError(f"target {index} is not an object")
        relative_path = item.get("path")
        expected_bytes = item.get("bytes")
        expected_sha256 = item.get("sha256")
        reason = item.get("reason")
        evidence = item.get("rebuild_evidence")
        if not isinstance(relative_path, str):
            raise ValueError(f"target {index} lacks path")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ValueError(f"target {index} has invalid bytes")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_sha256)
        ):
            raise ValueError(f"target {index} has invalid sha256")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"target {index} lacks reason")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"target {index} lacks rebuild evidence")

        target = _resolve_relative(repo_root, relative_path, label="target")
        if not target.is_relative_to(allowed_root):
            raise ValueError(f"target is outside allowed root: {target}")
        if target in seen:
            raise ValueError(f"duplicate target: {target}")
        seen.add(target)
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"target is not a regular file: {target}")
        actual_bytes = target.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"target size mismatch: {target}: {actual_bytes} != {expected_bytes}"
            )
        actual_sha256 = sha256_file(target)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"target SHA-256 mismatch: {target}: "
                f"{actual_sha256} != {expected_sha256}"
            )

        checked_evidence = []
        for evidence_relative in evidence:
            if not isinstance(evidence_relative, str):
                raise ValueError(f"non-string rebuild evidence for {target}")
            evidence_path = _resolve_relative(
                repo_root, evidence_relative, label="rebuild evidence"
            )
            if not evidence_path.exists() or evidence_path.is_symlink():
                raise ValueError(f"missing rebuild evidence: {evidence_path}")
            checked_evidence.append(evidence_relative)

        total_bytes += actual_bytes
        checked.append(
            {
                "path": relative_path,
                "absolute_path": str(target),
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "reason": reason,
                "rebuild_evidence": checked_evidence,
            }
        )

    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "allowed_root": str(allowed_root),
        "contract": {
            "only_exact_hash_locked_files_removed": True,
            "directories_removed": False,
            "rebuild_evidence_preserved": True,
        },
        "targets": checked,
        "planned_removed_bytes": total_bytes,
        "dry_run": dry_run,
        "completed": False,
    }
    if dry_run:
        return result

    _atomic_json(manifest_path, result)
    removed_bytes = 0
    for item in checked:
        target = Path(item["absolute_path"])
        if target.is_symlink() or not target.is_file():
            raise RuntimeError(f"target changed before deletion: {target}")
        if target.stat().st_size != item["bytes"]:
            raise RuntimeError(f"target size changed before deletion: {target}")
        target.unlink()
        removed_bytes += item["bytes"]
    remaining = [item["absolute_path"] for item in checked if Path(item["absolute_path"]).exists()]
    if remaining:
        raise RuntimeError(f"targets remain after deletion: {remaining}")

    result["completed"] = True
    result["completed_at"] = datetime.now().astimezone().isoformat()
    result["removed_bytes"] = removed_bytes
    result["dry_run"] = False
    _atomic_json(manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--allowed-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = remove_rebuildable_artifacts(
        repo_root=args.repo_root,
        allowed_root=args.allowed_root,
        plan_path=args.plan,
        manifest_path=args.manifest,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
