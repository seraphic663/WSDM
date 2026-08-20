#!/usr/bin/env python3
"""Self-contained tests for checkpoint completeness selection."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from find_valid_checkpoint import valid_checkpoints, validation_error


def write_checkpoint(root: Path, step: int, *, complete: bool) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"model")
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step})
    )
    if complete:
        (path / "optimizer.pt").write_bytes(b"optimizer")
        (path / "scheduler.pt").write_bytes(b"scheduler")
    return path


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        first = write_checkpoint(root, 10, complete=True)
        partial = write_checkpoint(root, 20, complete=False)
        latest = write_checkpoint(root, 30, complete=True)

        valid, rejected = valid_checkpoints(root)
        assert valid == [first.resolve(), latest.resolve()]
        assert any(str(partial) in message for message in rejected)
        assert validation_error(latest) is None
        assert validation_error(partial) is not None
    print("checkpoint selection: ok")


if __name__ == "__main__":
    main()
