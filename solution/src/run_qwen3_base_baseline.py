#!/usr/bin/env python3
"""Plan or launch the untrained Qwen3-Embedding-0.6B A-board baseline.

The default action is a non-destructive preflight. Execution is deliberately
gated because the upstream pipeline may download datasets/models and create or
update Python environments.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


RETRIEVER_REPO = "Qwen/Qwen3-Embedding-0.6B"
AGENT_REPO = "Qwen/Qwen3.5-4B"


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    total_mib: int
    used_mib: int
    utilization_percent: int

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.used_mib

    @property
    def idle(self) -> bool:
        return self.used_mib <= 2048 and self.utilization_percent <= 20


def detect_gpus() -> list[Gpu]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    gpus: list[Gpu] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        index, name, total, used, utilization = [value.strip() for value in line.split(",", 4)]
        gpus.append(
            Gpu(
                index=int(index),
                name=name,
                total_mib=int(total),
                used_mib=int(used),
                utilization_percent=int(utilization),
            )
        )
    return gpus


def model_is_complete(path: Path) -> bool:
    has_weights = (path / "model.safetensors").is_file() or (
        path / "model.safetensors.index.json"
    ).is_file()
    return (path / "config.json").is_file() and has_weights and (
        path / "tokenizer_config.json"
    ).is_file()


def prepared_assets(root: Path) -> dict[str, bool]:
    agent_dir = root / "models" / "_llm" / "Qwen__Qwen3.5-4B"
    return {
        "browsecomp_source": (root / "src" / "BrowseComp-Plus" / "pyproject.toml").is_file(),
        "main_environment": (root / "envs" / "browsecomp-plus" / "bin" / "python").is_file(),
        "qwen35_environment": (root / "envs" / "browsecomp-plus-llm-qwen35" / "bin" / "vllm").is_file(),
        "agent_judge_model": model_is_complete(agent_dir),
        "benchmark_dataset": (root / "src" / "BrowseComp-Plus" / "data" / "browsecomp_plus_decrypted.jsonl").is_file(),
        "benchmark_queries": (root / "src" / "BrowseComp-Plus" / "topics-qrels" / "queries.tsv").is_file(),
        "full_corpus_snapshot": (root / "data" / "hf_snapshots" / "browsecomp-plus-corpus" / ".bcp_snapshot_complete").is_file(),
    }


def build_command(args: argparse.Namespace, idle_gpus: list[Gpu]) -> list[str]:
    selected = [gpu.index for gpu in idle_gpus[:2]]
    index_devices = ",".join(str(index) for index in selected)
    agent_device = str(selected[0])
    search_device = str(selected[1])
    command = [
        "bash",
        str(args.official_script),
        "--root",
        str(args.root),
        "--agent-model",
        AGENT_REPO,
        "--judge-model",
        AGENT_REPO,
        "--index-gpu-devices",
        index_devices,
        "--index-shards",
        str(len(selected)),
        "--agent-gpu-devices",
        agent_device,
        "--search-gpu-devices",
        search_device,
        "--judge-gpu-devices",
        agent_device,
        "--tensor-parallel-size",
        "1",
        "--retrieval-batch-size",
        "16",
        "--attn-implementation",
        "sdpa",
        "--agent-workers",
        "2",
        "--vllm-max-num-seqs",
        "2",
        "--vllm-gpu-memory-util",
        "0.75",
        "--keep-models",
        "--keep-indexes",
    ]
    if not args.allow_downloads_and_setup:
        command.append("--skip-env")
    if args.mode == "smoke":
        command.extend(["--corpus-limit", str(args.corpus_limit)])
        command.extend(["--eval-limit", str(args.eval_limit)])
    command.append(RETRIEVER_REPO)
    return command


def stage_retriever(source: Path, root: Path) -> Path:
    destination = root / "models" / "Qwen__Qwen3-Embedding-0.6B"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"refusing to replace existing retriever path: {destination}")
    else:
        destination.symlink_to(source.resolve(), target_is_directory=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "smoke", "full"), default="preflight")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-downloads-and-setup", action="store_true")
    parser.add_argument("--corpus-limit", type=int, default=1000)
    parser.add_argument("--eval-limit", type=int, default=5)
    parser.add_argument(
        "--retriever-model",
        type=Path,
        default=Path("/root/data/LRAT/ccir/models/Qwen3-Embedding-0.6B"),
    )
    parser.add_argument(
        "--official-script",
        type=Path,
        default=Path("/root/data/LRAT/xir_a_leaderboard_eval/run_browsecomp_plus_eval.sh"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/root/data/LRAT/ccir/outputs/a_leaderboard_base"),
    )
    parser.add_argument(
        "--output-plan",
        type=Path,
        default=Path("/root/data/LRAT/ccir/outputs/eval/qwen3_base_preflight.json"),
    )
    args = parser.parse_args()

    if args.corpus_limit <= 0 or args.eval_limit <= 0:
        parser.error("--corpus-limit and --eval-limit must be positive")
    if args.execute and args.mode == "preflight":
        parser.error("--execute requires --mode smoke or --mode full")

    gpus = detect_gpus()
    idle_gpus = sorted((gpu for gpu in gpus if gpu.idle), key=lambda gpu: gpu.free_mib, reverse=True)
    retriever_complete = model_is_complete(args.retriever_model)
    assets = prepared_assets(args.root)
    errors: list[str] = []
    warnings: list[str] = []
    if not args.official_script.is_file():
        errors.append(f"official evaluation script is missing: {args.official_script}")
    if not retriever_complete:
        errors.append(f"original Qwen3 retriever is incomplete: {args.retriever_model}")
    if len(idle_gpus) < 2:
        errors.append("at least two idle GPUs are required by this 2xA40 adaptation")
    if not args.allow_downloads_and_setup:
        missing = [name for name, present in assets.items() if not present]
        if missing:
            errors.append(
                "downloads/setup are disabled and prepared assets are missing: " + ", ".join(missing)
            )
    else:
        warnings.append("official pipeline may download models/datasets and create or update environments")
    if args.mode == "full":
        warnings.append("full mode indexes the complete corpus and evaluates all benchmark queries")

    command = build_command(args, idle_gpus) if len(idle_gpus) >= 2 else []
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "untrained original Qwen3-Embedding-0.6B in the official A-board pipeline",
        "training_required": False,
        "mode": args.mode,
        "execute_requested": args.execute,
        "allow_downloads_and_setup": args.allow_downloads_and_setup,
        "gpus": [asdict(gpu) | {"free_mib": gpu.free_mib, "idle": gpu.idle} for gpu in gpus],
        "selected_idle_gpus": [gpu.index for gpu in idle_gpus[:2]],
        "retriever_model": str(args.retriever_model),
        "retriever_complete": retriever_complete,
        "prepared_assets": assets,
        "errors": errors,
        "warnings": warnings,
        "command": command,
        "command_shell": shlex.join(command) if command else "",
    }
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not args.execute:
        return
    if errors:
        print("execution blocked by preflight errors", file=sys.stderr)
        raise SystemExit(2)
    stage_retriever(args.retriever_model, args.root)
    # Re-check immediately before launch so a stale preflight cannot reserve busy GPUs.
    refreshed_idle = [gpu for gpu in detect_gpus() if gpu.idle]
    selected = {gpu.index for gpu in idle_gpus[:2]}
    if not selected.issubset({gpu.index for gpu in refreshed_idle}):
        raise SystemExit("selected GPU state changed after preflight; refusing to launch")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
