#!/usr/bin/env python3
"""Run CPU-only loading, sampling, and loss checks for the strategy suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(path: Path) -> dict:
    model = AutoModel.from_pretrained(path, local_files_only=True)
    result = {
        "path": str(path.resolve()),
        "sha256": sha256(path / "model.safetensors"),
        "class": type(model).__name__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "dtype": str(next(model.parameters()).dtype),
    }
    del model
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root / "cpu_validation.json"
    if output.exists():
        raise FileExistsError(output)

    h_path = args.data_root / "H/proven_multi_positive32.jsonl"
    sampling = []
    for row_number, line in enumerate(h_path.open(encoding="utf-8"), 1):
        row = json.loads(line)
        selected = set()
        for seed in range(256):
            selected.add(random.Random(seed + row_number * 1000).randrange(len(row["pos"])))
        sampling.append({"row": row_number, "positive_count": len(row["pos"]), "distinct_indices_selected": sorted(selected), "all_reachable": len(selected) == len(row["pos"])})

    scores = torch.tensor([[2.0, 0.0], [0.2, 0.0]], dtype=torch.float64)
    targets = torch.tensor([0, 0])
    per_row = F.cross_entropy(scores, targets, reduction="none")
    official_weights = torch.tensor([0.5, 1.5], dtype=torch.float64)
    unit_loss = per_row.mean()
    weighted_loss = (per_row * official_weights).sum() / official_weights.sum()

    models = []
    for name in ("tail_avg_2", "tail_avg_3"):
        models.append(load_model(args.output_root / f"G/{name}"))

    report = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "simulation_only": True,
        "C_weighted_loss_unit": {"per_row": per_row.tolist(), "weights": official_weights.tolist(), "unit_loss": unit_loss.item(), "weighted_loss": weighted_loss.item(), "different": not torch.isclose(unit_loss, weighted_loss).item()},
        "G_new_process_load": models,
        "H_sampling": {"rows": len(sampling), "all_rows_expose_every_proven_positive": all(item["all_reachable"] for item in sampling), "details": sampling, "limitation": "current training dataset chooses exactly one positive via random.choice; this validates sampling, not a joint multi-positive objective"},
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
