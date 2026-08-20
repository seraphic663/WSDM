#!/usr/bin/env python3
"""Atomically freeze a completed training output into a read-only inference copy."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable


WHITELIST = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "generation_config.json",
)
REQUIRED = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rename_noreplace(source: Path, target: Path) -> None:
    """Linux atomic rename that fails if target already exists."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is required for atomic no-replace freeze")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(target)
    if error not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
        raise OSError(error, os.strerror(error), target)

    # Some distributed filesystems reject renameat2 flags with EINVAL. Reserve
    # the destination name atomically, then replace only our own empty
    # reservation with the staging directory using same-filesystem rename.
    target.mkdir(mode=0o700, exist_ok=False)
    try:
        os.rename(source, target)
    except BaseException:
        target.rmdir()
        raise


def subprocess_validator(path: Path) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--validate-only", str(path)],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    return json.loads(result.stdout)


def freeze_model(source: Path, target: Path, validator: Callable[[Path], dict] = subprocess_validator) -> dict:
    source = source.resolve()
    target = target.absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite freeze target: {target}")
    if not (source / "COMPLETED").is_file() or (source / "RUNNING").exists():
        raise RuntimeError("source is not a completed, inactive training output")
    for name in REQUIRED:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging.", dir=target.parent))
    started = datetime.now().astimezone().isoformat()
    try:
        copied = []
        for name in WHITELIST:
            source_file = source / name
            if not source_file.is_file():
                continue
            target_file = staging / name
            shutil.copy2(source_file, target_file)
            source_hash = sha256(source_file)
            target_hash = sha256(target_file)
            if source_hash != target_hash or source_file.stat().st_size != target_file.stat().st_size:
                raise RuntimeError(f"copy verification failed: {name}")
            copied.append({"name": name, "bytes": target_file.stat().st_size, "sha256": target_hash})

        load_result = validator(staging)
        manifest = {
            "source": str(source),
            "target": str(target),
            "simulation_only": False,
            "started_at": started,
            "completed_at": datetime.now().astimezone().isoformat(),
            "whitelist": list(WHITELIST),
            "files": copied,
            "model_sha256": next(item["sha256"] for item in copied if item["name"] == "model.safetensors"),
            "new_cpu_process_load": load_result,
        }
        manifest_path = staging / "FREEZE_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for path in staging.iterdir():
            if path.is_file():
                path.chmod(0o444)
        staging.chmod(0o555)
        rename_noreplace(staging, target)
        if target.stat().st_mode & 0o777 != 0o555:
            raise RuntimeError("final directory mode verification failed")
        for path in target.iterdir():
            if path.is_file() and path.stat().st_mode & 0o777 != 0o444:
                raise RuntimeError(f"final file mode verification failed: {path.name}")
        return manifest
    except BaseException:
        if staging.exists():
            staging.chmod(0o755)
            for path in staging.iterdir():
                if path.is_file():
                    path.chmod(0o644)
            shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_only(path: Path) -> None:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(path, local_files_only=True)
    result = {
        "pid": os.getpid(),
        "device": "cpu",
        "class": type(model).__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dtype": str(next(model.parameters()).dtype),
    }
    print(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--validate-only", type=Path)
    args = parser.parse_args()
    if args.validate_only:
        validate_only(args.validate_only)
        return
    if not args.source or not args.target:
        parser.error("--source and --target are required")
    print(json.dumps(freeze_model(args.source, args.target), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
