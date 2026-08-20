#!/usr/bin/env python3
"""Average compatible safetensors models into a new atomic directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


FILES = ("config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def average(models: list[Path], output: Path) -> dict:
    if len(models) not in (2, 3):
        raise ValueError("exactly two or three models are required")
    if output.exists():
        raise FileExistsError(output)
    weights = [model / "model.safetensors" for model in models]
    for path in weights:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent))
    handles = [safe_open(path, framework="pt", device="cpu") for path in weights]
    try:
        keys = list(handles[0].keys())
        if any(list(handle.keys()) != keys for handle in handles[1:]):
            raise ValueError("tensor keys differ")
        tensors = {}
        for key in keys:
            values = [handle.get_tensor(key) for handle in handles]
            if any(value.shape != values[0].shape or value.dtype != values[0].dtype for value in values[1:]):
                raise ValueError(f"tensor incompatibility: {key}")
            if values[0].is_floating_point():
                tensors[key] = (sum(value.float() for value in values) / len(values)).to(values[0].dtype)
            else:
                if any(not torch.equal(values[0], value) for value in values[1:]):
                    raise ValueError(f"non-floating tensor differs: {key}")
                tensors[key] = values[0]
        save_file(tensors, temp / "model.safetensors", metadata=handles[-1].metadata())
        del tensors
        for name in FILES:
            shutil.copy2(models[-1] / name, temp / name)
        report = {
            "formula": f"equal arithmetic mean of {len(models)} checkpoints",
            "models": [{"path": str(path.resolve()), "sha256": digest(weight)} for path, weight in zip(models, weights)],
            "tensor_keys": len(keys),
            "output_sha256": digest(temp / "model.safetensors"),
        }
        (temp / "AVERAGE.json").write_text(json.dumps(report, indent=2) + "\n")
        os.replace(temp, output)
        return report
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    finally:
        del handles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(average(args.model, args.output), indent=2))


if __name__ == "__main__":
    main()
