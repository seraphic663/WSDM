#!/usr/bin/env python3
"""Create an atomic weight-space interpolation of two compatible HF models."""

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


MODEL_FILE = "model.safetensors"
INFERENCE_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interpolate_models(model_a: Path, model_b: Path, output: Path, alpha: float) -> dict:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    weight_a = model_a / MODEL_FILE
    weight_b = model_b / MODEL_FILE
    for path in (weight_a, weight_b):
        if not path.is_file():
            raise FileNotFoundError(path)
    for name in INFERENCE_FILES:
        if not (model_b / name).is_file():
            raise FileNotFoundError(model_b / name)

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent))
    try:
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(weight_a, framework="pt", device="cpu") as first, safe_open(
            weight_b, framework="pt", device="cpu"
        ) as second:
            keys_a = list(first.keys())
            keys_b = list(second.keys())
            if keys_a != keys_b:
                raise ValueError("model tensor keys do not match")
            metadata = second.metadata()
            for key in keys_a:
                tensor_a = first.get_tensor(key)
                tensor_b = second.get_tensor(key)
                if tensor_a.shape != tensor_b.shape or tensor_a.dtype != tensor_b.dtype:
                    raise ValueError(f"incompatible tensor {key}")
                if tensor_a.is_floating_point():
                    tensors[key] = torch.lerp(
                        tensor_a.float(), tensor_b.float(), alpha
                    ).to(tensor_a.dtype)
                else:
                    if not torch.equal(tensor_a, tensor_b):
                        raise ValueError(f"non-floating tensor differs: {key}")
                    tensors[key] = tensor_b

        save_file(tensors, temp / MODEL_FILE, metadata=metadata)
        del tensors
        for name in INFERENCE_FILES:
            shutil.copy2(model_b / name, temp / name)

        manifest = {
            "formula": "output = (1 - alpha) * model_a + alpha * model_b",
            "alpha": alpha,
            "model_a": str(model_a.resolve()),
            "model_b": str(model_b.resolve()),
            "model_a_sha256": sha256(weight_a),
            "model_b_sha256": sha256(weight_b),
            "output_sha256": sha256(temp / MODEL_FILE),
        }
        (temp / "INTERPOLATION.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, output)
        return manifest
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True, type=Path)
    parser.add_argument("--model-b", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", required=True, type=float)
    args = parser.parse_args()
    print(
        json.dumps(
            interpolate_models(args.model_a, args.model_b, args.output, args.alpha),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
