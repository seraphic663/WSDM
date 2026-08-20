#!/usr/bin/env python3
"""Build LRAT paper-style training pairs from freshly collected trajectories.

This intentionally differs from the repository's more permissive data builder:
only judge-approved browsed documents remain positives, and every negative is
drawn from the candidate set of the Search that produced that Browse action.
Weights are then computed globally over the retained positives using Eq. (3).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

BROWSE_TOOLS = {"get_document", "visit"}


@dataclass(frozen=True)
class BrowseCandidate:
    trajectory_path: str
    query: str
    positive_id: str
    candidate_ids: tuple[str, ...]
    reasoning_text: str


def _get_docid_from_browse_step(step: dict[str, Any]) -> str | None:
    arguments = json.loads(step.get("arguments", "{}") or "{}")
    doc_id = arguments.get("docid")
    if isinstance(doc_id, list):
        doc_id = doc_id[0] if doc_id else None
    return str(doc_id).split(":")[-1] if doc_id is not None else None


def _extract_reasoning_text_from_next_step(
    steps: list[dict[str, Any]], index: int
) -> str:
    if index + 1 >= len(steps) or steps[index + 1].get("type") != "reasoning":
        return ""
    output = steps[index + 1].get("output", "")
    if isinstance(output, list):
        return " ".join(map(str, output))
    return str(output)


def _parse_search(step: dict[str, Any]) -> tuple[str, list[str]]:
    arguments = json.loads(step.get("arguments", "{}") or "{}")
    query = arguments.get("query", [""])
    query = query[0] if isinstance(query, list) and query else str(query)
    docs = []
    for line in str(step.get("output", "") or "").splitlines():
        if line.startswith("DocID:"):
            docs.append(line.split(":", 1)[1].strip())
    return query, docs


def _token_len(tokenizer: Any, text: str) -> int | None:
    try:
        output = tokenizer(text, add_special_tokens=False)
        if isinstance(output, dict) and "input_ids" in output:
            return len(output["input_ids"])
    except Exception:
        pass
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return None


def iter_trajectory_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("run_*.json")):
        if path.is_file() and not path.is_symlink():
            yield path


def collect_browse_candidates(
    trajectory: dict[str, Any], *, trajectory_path: str
) -> list[BrowseCandidate]:
    steps = trajectory.get("result")
    if not isinstance(steps, list):
        raise ValueError(f"trajectory lacks result list: {trajectory_path}")
    current_query: str | None = None
    current_docs: list[str] | None = None
    candidates: list[BrowseCandidate] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        if step.get("type") == "tool_call" and step.get("tool_name") == "search":
            current_query, current_docs = _parse_search(step)
            current_docs = list(dict.fromkeys(map(str, current_docs)))
            continue
        if (
            step.get("type") == "tool_call"
            and step.get("tool_name") in BROWSE_TOOLS
            and current_query is not None
            and current_docs is not None
        ):
            positive_id = _get_docid_from_browse_step(step)
            if positive_id is None or positive_id not in current_docs:
                continue
            reasoning_text = _extract_reasoning_text_from_next_step(steps, index)
            candidates.append(
                BrowseCandidate(
                    trajectory_path=trajectory_path,
                    query=current_query,
                    positive_id=positive_id,
                    candidate_ids=tuple(current_docs),
                    reasoning_text=reasoning_text,
                )
            )
    return candidates


def build_retained_rows(
    candidates: list[BrowseCandidate],
    *,
    corpus: dict[str, str],
    tokenizer: Any,
    judge: Callable[[str], bool],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {
        "browse_candidates": len(candidates),
        "judge_relevant": 0,
        "judge_irrelevant": 0,
        "missing_reasoning": 0,
        "missing_positive_in_corpus": 0,
        "missing_negative_in_corpus": 0,
        "empty_negative_set": 0,
    }
    for candidate in candidates:
        relevant = bool(judge(candidate.reasoning_text))
        if not candidate.reasoning_text.strip():
            counts["missing_reasoning"] += 1
            counts["judge_irrelevant"] += 1
            continue
        if not relevant:
            counts["judge_irrelevant"] += 1
            continue
        counts["judge_relevant"] += 1
        positive_text = corpus.get(candidate.positive_id)
        if positive_text is None:
            counts["missing_positive_in_corpus"] += 1
            continue
        negative_ids: list[str] = []
        negative_texts: list[str] = []
        for doc_id in candidate.candidate_ids:
            if doc_id == candidate.positive_id:
                continue
            text = corpus.get(doc_id)
            if text is None:
                counts["missing_negative_in_corpus"] += 1
                continue
            negative_ids.append(doc_id)
            negative_texts.append(text)
        if not negative_ids:
            counts["empty_negative_set"] += 1
            continue
        reasoning_len = _token_len(tokenizer, candidate.reasoning_text)
        if not isinstance(reasoning_len, int) or reasoning_len <= 0:
            counts["missing_reasoning"] += 1
            continue
        rows.append(
            {
                "query": candidate.query,
                "pos": [positive_text],
                "neg": negative_texts,
                "pos_id": [candidate.positive_id],
                "neg_id": negative_ids,
                "reasoning_len": reasoning_len,
                "satisfied": True,
                "source_trajectory": candidate.trajectory_path,
            }
        )
    return rows, counts


def add_paper_weights(rows: list[dict[str, Any]]) -> tuple[float, float]:
    if not rows:
        raise ValueError("no judge-approved paper pairs")
    lengths = [float(row["reasoning_len"]) for row in rows]
    beta = statistics.median(lengths)
    raw = [
        1.0 - math.exp(-math.log(2.0) * length / beta)
        for length in lengths
    ]
    mean_raw = statistics.fmean(raw)
    for row, score in zip(rows, raw):
        row["reweight_rate"] = score / mean_raw
    return beta, mean_raw


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_candidates(directory: Path) -> list[BrowseCandidate]:
    candidates: list[BrowseCandidate] = []
    for path in iter_trajectory_files(directory):
        value = json.loads(path.read_text(encoding="utf-8"))
        candidates.extend(
            collect_browse_candidates(value, trajectory_path=path.name)
        )
    return candidates


def main() -> None:
    from transformers import AutoTokenizer

    from src.data_builder import judge_relevance, load_corpus_jsonl

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path", required=True, type=Path)
    parser.add_argument("--traj-dir", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--judge-api-url", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--future-timeout", type=int, default=600)
    args = parser.parse_args()
    if args.output_path.exists() or args.summary_path.exists():
        raise FileExistsError("output and summary must not already exist")

    corpus = load_corpus_jsonl(str(args.corpus_path))
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    candidates = _load_candidates(args.traj_dir)

    judgments: dict[int, bool] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                judge_relevance,
                candidate.reasoning_text,
                judge_api_url=args.judge_api_url,
                judge_model=args.judge_model,
                headers={"Content-Type": "application/json"},
            ): index
            for index, candidate in enumerate(candidates)
            if candidate.reasoning_text.strip()
        }
        for index, candidate in enumerate(candidates):
            if not candidate.reasoning_text.strip():
                judgments[index] = False
        for future in as_completed(futures):
            index = futures[future]
            judgments[index] = bool(future.result(timeout=args.future_timeout))

    ordered_judgments = iter(judgments[index] for index in range(len(candidates)))
    rows, counts = build_retained_rows(
        candidates,
        corpus=corpus,
        tokenizer=tokenizer,
        judge=lambda _text: next(ordered_judgments),
    )
    beta, mean_raw = add_paper_weights(rows)
    _atomic_jsonl(args.output_path, rows)
    summary = {
        "trajectory_files": sum(1 for _ in iter_trajectory_files(args.traj_dir)),
        **counts,
        "retained_pairs": len(rows),
        "global_median_reasoning_len": beta,
        "global_mean_raw_weight": mean_raw,
        "mean_normalized_weight": statistics.fmean(
            float(row["reweight_rate"]) for row in rows
        ),
        "paper_contract": {
            "irrelevant_browses_removed_from_positives": True,
            "negatives_only_from_corresponding_search_candidate_set": True,
            "immediate_post_browse_reasoning_used": True,
            "global_eq3_weights": True,
        },
        "competition_submission_eligible": False,
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
