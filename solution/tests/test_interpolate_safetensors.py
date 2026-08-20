#!/usr/bin/env python3
"""Checks for atomic Hugging Face model interpolation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from interpolate_safetensors import INFERENCE_FILES, interpolate_models


def write_model(path: Path, value: float) -> None:
    path.mkdir()
    save_file(
        {
            "float_weight": torch.tensor([value, value + 2], dtype=torch.float32),
            "integer_state": torch.tensor([7], dtype=torch.int64),
        },
        path / "model.safetensors",
        metadata={"format": "pt"},
    )
    for name in INFERENCE_FILES:
        (path / name).write_text(f"{name}\n", encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        first, second, output = root / "a", root / "b", root / "out"
        write_model(first, 0.0)
        write_model(second, 4.0)

        manifest = interpolate_models(first, second, output, 0.25)
        result = load_file(output / "model.safetensors")
        torch.testing.assert_close(result["float_weight"], torch.tensor([1.0, 3.0]))
        torch.testing.assert_close(result["integer_state"], torch.tensor([7]))
        assert manifest["alpha"] == 0.25
        assert json.loads((output / "INTERPOLATION.json").read_text())["output_sha256"]

        try:
            interpolate_models(first, second, output, 0.5)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing output was overwritten")
    print("safetensors interpolation: ok")


if __name__ == "__main__":
    main()
