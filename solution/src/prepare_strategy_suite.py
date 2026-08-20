#!/usr/bin/env python3
"""Build isolated, provenance-rich samples for the CCIR strategy suite."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def aligned(row: dict, line: int) -> None:
    for text_key, id_key in (("pos", "pos_id"), ("neg", "neg_id")):
        if not isinstance(row.get(text_key), list) or not isinstance(row.get(id_key), list):
            raise ValueError(f"line {line}: invalid {text_key}/{id_key}")
        if len(row[text_key]) != len(row[id_key]):
            raise ValueError(f"line {line}: misaligned {text_key}/{id_key}")
    if not isinstance(row.get("query"), str):
        raise ValueError(f"line {line}: invalid query")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--cleaned", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--git-head", required=True)
    args = parser.parse_args()

    started = datetime.now(timezone.utc).astimezone().isoformat()
    for root in (args.data_root, args.output_root):
        if root.exists():
            raise FileExistsError(f"refusing existing suite root: {root}")
        root.mkdir(parents=True)

    schema = collections.Counter()
    satisfied = collections.Counter()
    reasoning: list[float] = []
    weights: list[float] = []
    positives: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    query_lines: dict[str, list[int]] = collections.defaultdict(list)
    changed: list[tuple[int, dict, dict]] = []
    group_candidates: list[tuple[int, dict]] = []

    with args.raw.open(encoding="utf-8") as raw_handle, args.cleaned.open(encoding="utf-8") as clean_handle:
        for line_number, (raw_line, clean_line) in enumerate(zip(raw_handle, clean_handle), 1):
            raw = json.loads(raw_line)
            clean = json.loads(clean_line)
            aligned(raw, line_number)
            aligned(clean, line_number)
            if raw["query"] != clean["query"] or raw["pos_id"] != clean["pos_id"]:
                raise ValueError(f"line {line_number}: raw/cleaned alignment changed")
            schema[tuple(sorted(raw))] += 1
            satisfied[str(raw.get("satisfied"))] += 1
            if isinstance(raw.get("reasoning_len"), (int, float)):
                reasoning.append(float(raw["reasoning_len"]))
            if isinstance(raw.get("reweight_rate"), (int, float)) and math.isfinite(raw["reweight_rate"]):
                weights.append(float(raw["reweight_rate"]))
            query = raw["query"]
            query_lines[query].append(line_number)
            for pid, text in zip(raw["pos_id"], raw["pos"]):
                positives[query].setdefault(str(pid), {"text": text, "lines": []})["lines"].append(line_number)
            if raw != clean and len(clean["neg"]) >= 5 and len(changed) < 64:
                changed.append((line_number, raw, clean))
            if len(raw["neg"]) >= 9 and len(group_candidates) < 256:
                group_candidates.append((line_number, raw))
        if raw_handle.readline() or clean_handle.readline():
            raise ValueError("raw and cleaned row counts differ")

    # A: deterministic query-disjoint mini split. Selection depends only on query hash.
    train: list[dict] = []
    diag: list[dict] = []
    train_prov: list[dict] = []
    diag_prov: list[dict] = []
    seen_train: set[str] = set()
    seen_diag: set[str] = set()
    with args.raw.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            query = row["query"]
            bucket = hashlib.sha256(query.encode()).digest()[0]
            if bucket < 32 and query not in seen_diag and len(diag) < 32:
                diag.append(row); seen_diag.add(query)
                diag_prov.append({"source_line": line_number, "query_sha256": hashlib.sha256(query.encode()).hexdigest(), "source_query_rows": query_lines[query]})
            elif bucket >= 32 and query not in seen_train and len(train) < 64:
                train.append(row); seen_train.add(query)
                train_prov.append({"source_line": line_number, "query_sha256": hashlib.sha256(query.encode()).hexdigest()})
            if len(train) == 64 and len(diag) == 32:
                break
    if seen_train & seen_diag:
        raise AssertionError("query split overlap")

    # B: line-aligned raw versus existing cleaned rows.
    if len(changed) < 64:
        raise ValueError("fewer than 64 trainable changed rows")
    b_raw = [item[1] for item in changed]
    b_clean = [item[2] for item in changed]
    b_prov = [{"source_line": line, "raw_negatives": len(raw["neg"]), "cleaned_negatives": len(clean["neg"]), "removed": len(raw["neg"]) - len(clean["neg"])} for line, raw, clean in changed]

    # C/E/F: one fixed 64-row sample with >=9 negatives and unique queries.
    fixed: list[dict] = []
    fixed_prov: list[dict] = []
    used: set[str] = set()
    for line_number, row in group_candidates:
        if row["query"] in used:
            continue
        fixed.append(row); used.add(row["query"])
        fixed_prov.append({"source_line": line_number, "official_reweight_rate": row.get("reweight_rate"), "reasoning_len": row.get("reasoning_len"), "satisfied": row.get("satisfied")})
        if len(fixed) == 64:
            break
    if len(fixed) < 64:
        raise ValueError("fewer than 64 group-size-compatible rows")
    c_unit = [{**row, "reweight_rate": 1.0} for row in fixed]
    c_weighted = [dict(row) for row in fixed]

    # D: existing-official-negative ranking pool with global same-query positive shielding.
    d_rows: list[dict] = []
    d_prov: list[dict] = []
    with args.raw.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            positive_union = set(positives[row["query"]])
            keep = [i for i, nid in enumerate(row["neg_id"]) if str(nid) not in positive_union]
            if len(keep) < 9:
                continue
            candidate = dict(row)
            candidate["neg"] = [row["neg"][i] for i in keep[:16]]
            candidate["neg_id"] = [row["neg_id"][i] for i in keep[:16]]
            d_rows.append(candidate)
            d_prov.append({"source_line": line_number, "global_positive_ids": sorted(positive_union), "removed_false_negatives": len(row["neg"]) - len(keep), "candidate_origin": "official_existing_neg_only"})
            if len(d_rows) == 16:
                break

    # H: only provable multi-positive groups from identical official query text.
    multi_queries = [q for q, p in positives.items() if len(p) > 1]
    h_rows: list[dict] = []
    h_prov: list[dict] = []
    wanted = set(multi_queries[:32])
    with args.raw.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            query = row["query"]
            if query not in wanted:
                continue
            positive_map = positives[query]
            positive_ids = list(positive_map)
            neg_pairs = [(nid, text) for nid, text in zip(row["neg_id"], row["neg"]) if str(nid) not in positive_map]
            if not neg_pairs:
                continue
            merged = dict(row)
            merged["pos_id"] = positive_ids
            merged["pos"] = [positive_map[pid]["text"] for pid in positive_ids]
            merged["neg_id"] = [pair[0] for pair in neg_pairs]
            merged["neg"] = [pair[1] for pair in neg_pairs]
            h_rows.append(merged)
            h_prov.append({"source_lines": query_lines[query], "positive_ids": positive_ids, "positive_source_lines": {pid: positive_map[pid]["lines"] for pid in positive_ids}})
            wanted.remove(query)
            if not wanted:
                break

    files = {
        "a_train64": ("A/query_isolated_train64.jsonl", train),
        "a_diag32": ("A/query_isolated_diag32.jsonl", diag),
        "b_raw64": ("B/original_changed64.jsonl", b_raw),
        "b_clean64": ("B/cleaned_changed64.jsonl", b_clean),
        "c_unit64": ("C/unit_weight64.jsonl", c_unit),
        "c_weighted64": ("C/trajectory_weight64.jsonl", c_weighted),
        "d_candidates16": ("D/shielded_existing_neg16.jsonl", d_rows),
        "fixed64": ("shared/fixed_group9_64.jsonl", fixed),
        "h_multi32": ("H/proven_multi_positive32.jsonl", h_rows),
    }
    output_hashes = {}
    for key, (relative, rows) in files.items():
        path = args.data_root / relative
        write_jsonl(path, rows)
        output_hashes[key] = {"path": str(path), "rows": len(rows), "bytes": path.stat().st_size, "sha256": sha256(path)}

    provenance = {
        "A": {"train": train_prov, "diag": diag_prov, "exact_query_overlap": 0, "task_mapping_available": False},
        "B": b_prov,
        "C_E_F": fixed_prov,
        "D": d_prov,
        "H": h_prov,
    }
    write_json(args.output_root / "provenance.json", provenance)
    distribution = lambda v: {"count": len(v), "min": min(v), "p25": percentile(v, .25), "median": statistics.median(v), "p75": percentile(v, .75), "max": max(v), "mean": statistics.fmean(v)}
    report = {
        "status": {"A": "PASS", "B": "PARTIAL", "C": "PARTIAL", "D": "PARTIAL", "E": "PARTIAL", "F": "PARTIAL", "G": "PARTIAL", "H": "PASS", "I": "PARTIAL"},
        "raw_schema": [{"keys": list(keys), "rows": count} for keys, count in schema.items()],
        "task_or_trajectory_identity_fields_found": sorted(set().union(*(set(keys) for keys in schema)) & {"task_id", "trajectory_id", "turn_id", "session_id"}),
        "task_mapping_evidence": "pairs contain no task/trajectory identity field; no separate official trajectory file was present in the server inventory",
        "satisfied": dict(satisfied),
        "reasoning_len": distribution(reasoning),
        "reweight_rate": distribution(weights),
        "unique_queries": len(positives),
        "provable_multi_positive_query_groups": len(multi_queries),
        "official_corpus_present": False,
        "corpus_evidence": "no LRAT offline corpus file found; BrowseComp-Plus TSV is evaluation task data, not the official offline corpus",
        "outputs": output_hashes,
    }
    write_json(args.output_root / "cpu_report.json", report)
    manifest = {
        "suite_id": "strategy_suite_20260718",
        "simulation_only": True,
        "git_head": args.git_head,
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "command": " ".join(__import__("sys").argv),
        "inputs": {
            "raw": {"path": str(args.raw.resolve()), "sha256": sha256(args.raw)},
            "cleaned": {"path": str(args.cleaned.resolve()), "sha256": sha256(args.cleaned)},
        },
        "outputs": output_hashes,
        "notes": ["all products are isolated simulations", "dev500 is training-overlap diagnostics only"],
    }
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
