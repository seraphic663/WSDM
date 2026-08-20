#!/usr/bin/env python3
"""Pure helpers for safe pause/resume checkpoints during sequential training."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path


def should_pause(global_step: int, max_steps: int, target_step: int) -> bool:
    if target_step < 1 or max_steps < 1 or global_step < 0:
        raise ValueError("steps must be positive (global_step may be zero)")
    return global_step >= target_step and global_step < max_steps


def write_pause_marker(path: Path, *, global_step: int, max_steps: int, target_step: int) -> None:
    if global_step < target_step or global_step >= max_steps:
        raise ValueError("pause marker requires target_step <= global_step < max_steps")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "paused_at": datetime.now().astimezone().isoformat(),
        "global_step": global_step,
        "max_steps": max_steps,
        "target_step": target_step,
    }
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
