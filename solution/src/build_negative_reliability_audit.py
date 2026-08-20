#!/usr/bin/env python3
"""Build the candidate-level table for the WSDM negative-reliability audit.

The input is the official BrowseComp-Plus search-agent trajectory release.  A
row in the output is one document occurrence in one search observation.  The
script deliberately does not infer ``utilized`` from visits or final answers:
that concept is not directly observable in the released trajectories.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SEARCH_HIT_RE = re.compile(
    r"(?m)^(?P<rank>\d+)\. Document ID: (?P<docid>[^\r\n]+)\r?\n"
    r"Score: (?P<score>[^\r\n]+)\r?\nSnippet:",
)
VISIT_CONTENT_RE = re.compile(r"\bdocid\s+([^\s.]+)", re.I)
WS_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalized_text(value: Any) -> str:
    return WS_RE.sub(" ", unicodedata.normalize("NFKC", str(value or "")).strip()).casefold()


def text_sha256(value: Any) -> str:
    return hashlib.sha256(normalized_text(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_answer_match(answer: Any, prediction: Any) -> bool | None:
    """Return a conservative text-containment proxy, not an official judge label."""
    gold = WORD_RE.findall(normalized_text(answer))
    pred = WORD_RE.findall(normalized_text(prediction))
    if not gold or not pred:
        return None
    return " ".join(gold) in " ".join(pred)


def load_queries(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    by_text: dict[str, str] = {}
    by_id: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                query_id, question = line.split("\t", 1)
            except ValueError as exc:
                raise ValueError(f"malformed query row {line_no}: {line!r}") from exc
            key = normalized_text(question)
            if key in by_text and by_text[key] != query_id:
                raise ValueError(f"normalized question collision: {query_id} and {by_text[key]}")
            by_text[key] = query_id
            by_id[query_id] = question
    return by_text, by_id


def load_qrels(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = collections.defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 4:
                raise ValueError(f"malformed qrel row {line_no}: {line!r}")
            query_id, _, doc_id, relevance = parts
            if int(relevance) > 0:
                result[query_id].add(doc_id)
    return dict(result)


def parse_search_observation(content: Any) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for match in SEARCH_HIT_RE.finditer(str(content or "")):
        score_text = match.group("score").strip()
        try:
            score = float(score_text)
        except ValueError:
            score = None
        hits.append(
            {
                "rank": int(match.group("rank")),
                "doc_id": match.group("docid").strip(),
                "score": score,
                "score_raw": score_text,
            }
        )
    return hits


def call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    value = call.get("arguments", {})
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def visit_doc_id(message: dict[str, Any], call: dict[str, Any]) -> str:
    value = call_arguments(call).get("docid")
    if isinstance(value, list):
        value = value[0] if value else None
    if value is not None:
        return str(value).strip()
    match = VISIT_CONTENT_RE.search(str(message.get("content") or ""))
    return match.group(1).strip() if match else ""


@dataclass
class ParsedTrajectory:
    rows: list[dict[str, Any]]
    stats: dict[str, int]


def parse_trajectory(
    record: dict[str, Any],
    *,
    agent: str,
    trajectory_id: str,
    query_id: str,
    evidence_docs: set[str],
    gold_docs: set[str],
) -> ParsedTrajectory:
    messages = record.get("messages")
    if not isinstance(messages, list):
        messages = []

    calls_by_id: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    stats = collections.Counter()
    for message_index, raw_message in enumerate(messages):
        message = raw_message if isinstance(raw_message, dict) else {}
        if message.get("role") == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for call_index, raw_call in enumerate(tool_calls):
                    call = raw_call if isinstance(raw_call, dict) else {}
                    call_id = str(call.get("id") or f"message-{message_index}-call-{call_index}")
                    calls_by_id[call_id] = call
            continue
        if message.get("role") != "tool":
            continue
        call_id = str(message.get("tool_call_id") or "")
        call = calls_by_id.get(call_id, {})
        name = str(message.get("name") or call.get("name") or "")
        if name == "search":
            stats["search_observations"] += 1
            content = str(message.get("content") or "")
            hits = parse_search_observation(content)
            if hits:
                stats["parsed_search_observations"] += 1
            elif "search request failed" in content.casefold():
                stats["failed_search_observations"] += 1
            elif "invalid request format" in content.casefold():
                stats["invalid_search_request_observations"] += 1
            else:
                stats["unrecognized_search_observations"] += 1
            if len(hits) == 5:
                stats["five_hit_search_observations"] += 1
            query = str(call_arguments(call).get("query") or "")
            events.append(
                {
                    "kind": "search",
                    "message_index": message_index,
                    "query": query,
                    "hits": hits,
                }
            )
        elif name == "visit":
            stats["visit_observations"] += 1
            doc_id = visit_doc_id(message, call)
            if doc_id:
                stats["parsed_visit_observations"] += 1
            content = str(message.get("content") or "")
            successful = content.startswith("Successfully visited")
            stats["successful_visit_observations" if successful else "failed_visit_observations"] += 1
            events.append(
                {"kind": "visit", "message_index": message_index, "doc_id": doc_id, "successful": successful}
            )

    search_event_positions = [index for index, event in enumerate(events) if event["kind"] == "search"]
    total_searches = len(search_event_positions)
    all_visits = [
        (index, event["doc_id"])
        for index, event in enumerate(events)
        if event["kind"] == "visit" and event["doc_id"] and event["successful"]
    ]
    surfaced_before: collections.Counter[str] = collections.Counter()
    surfaced_doc_ids: set[str] = set()
    evidence_seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    answer_match = strict_answer_match(record.get("answer"), record.get("prediction"))

    for search_step, event_pos in enumerate(search_event_positions):
        event = events[event_pos]
        next_search_pos = (
            search_event_positions[search_step + 1]
            if search_step + 1 < total_searches
            else len(events)
        )
        immediate_visits = {
            visit_doc
            for visit_pos, visit_doc in all_visits
            if event_pos < visit_pos < next_search_pos
        }
        later_visits = {
            visit_doc
            for visit_pos, visit_doc in all_visits
            if visit_pos >= next_search_pos
        }
        prior_visits = {
            visit_doc for visit_pos, visit_doc in all_visits if visit_pos < event_pos
        }
        evidence_before = len(evidence_seen)
        for hit in event["hits"]:
            doc_id = hit["doc_id"]
            immediate = doc_id in immediate_visits
            later = (not immediate) and doc_id in later_visits
            evidence_positive = doc_id in evidence_docs
            gold_positive = doc_id in gold_docs
            prior_exposures = surfaced_before[doc_id]
            human_relevance = (
                "gold_qrel_positive"
                if gold_positive
                else "evidence_qrel_positive"
                if evidence_positive
                else "not_in_known_qrels"
            )
            rows.append(
                {
                    "query_id": query_id,
                    "question_sha256": text_sha256(record.get("question")),
                    "trajectory_id": trajectory_id,
                    "agent": agent,
                    "termination": record.get("termination"),
                    "strict_answer_match_proxy": answer_match,
                    "search_step": search_step,
                    "search_step_1based": search_step + 1,
                    "total_searches": total_searches,
                    "search_progress": (search_step + 1) / total_searches if total_searches else None,
                    "search_query": event["query"],
                    "doc_id": doc_id,
                    "rank": hit["rank"],
                    "score": hit["score"],
                    "surfaced": True,
                    "visited": immediate,
                    "visited_before_next_search": immediate,
                    "later_visited": later,
                    "visited_ever_after_exposure": immediate or later,
                    "visited_before_exposure": doc_id in prior_visits,
                    "utilized": None,
                    "human_relevance": human_relevance,
                    "evidence_qrel_positive": evidence_positive,
                    "gold_qrel_positive": gold_positive,
                    "evidence_qrel_count": len(evidence_docs),
                    "gold_qrel_count": len(gold_docs),
                    "evidence_seen_before": evidence_before,
                    "evidence_state": (
                        "none"
                        if evidence_before == 0
                        else "complete"
                        if evidence_docs and evidence_before >= len(evidence_docs)
                        else "partial"
                    ),
                    "prior_exposures_in_trajectory": prior_exposures,
                    "repeated_retrieval": prior_exposures > 0,
                    "source_dataset": "liuqi6777/bcp_search_agent_trajectory",
                    "qrel_source": "BrowseComp-Plus evidence/gold qrels",
                }
            )
            surfaced_before[doc_id] += 1
            surfaced_doc_ids.add(doc_id)
        evidence_seen.update(hit["doc_id"] for hit in event["hits"] if hit["doc_id"] in evidence_docs)

    stats["candidate_rows"] = len(rows)
    stats["successful_visit_observations_mapped_to_any_surface"] = sum(
        doc_id in surfaced_doc_ids for _, doc_id in all_visits
    )
    stats["successful_visit_observations_unmapped_to_any_surface"] = sum(
        doc_id not in surfaced_doc_ids for _, doc_id in all_visits
    )
    return ParsedTrajectory(rows=rows, stats=dict(stats))


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not an object")
            yield line_no, value


def build(args: argparse.Namespace) -> dict[str, Any]:
    by_text, queries = load_queries(args.queries)
    evidence = load_qrels(args.evidence_qrels)
    gold = load_qrels(args.gold_qrels)
    if any(not docs <= evidence.get(query_id, set()) for query_id, docs in gold.items()):
        raise ValueError("gold qrels are not a subset of evidence qrels")

    input_files = sorted(args.trajectories_root.glob("*/predictions.jsonl"))
    if args.agents:
        selected_agents = set(args.agents)
        input_files = [path for path in input_files if path.parent.name in selected_agents]
    if not input_files:
        raise FileNotFoundError(f"no predictions.jsonl below {args.trajectories_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    totals = collections.Counter()
    per_agent: dict[str, collections.Counter[str]] = {}
    seen_query_ids: dict[str, set[str]] = collections.defaultdict(set)
    unmatched_questions: list[dict[str, Any]] = []

    with gzip.open(args.output, "wt", encoding="utf-8") as output:
        for path in input_files:
            agent = path.parent.name
            agent_counts: collections.Counter[str] = collections.Counter()
            per_agent[agent] = agent_counts
            for line_no, record in iter_jsonl(path):
                totals["trajectory_rows"] += 1
                agent_counts["trajectory_rows"] += 1
                termination = str(record.get("termination") or "unknown")
                totals[f"termination_{termination}"] += 1
                agent_counts[f"termination_{termination}"] += 1
                question_key = normalized_text(record.get("question"))
                query_id = by_text.get(question_key)
                if query_id is None:
                    unmatched_questions.append(
                        {"agent": agent, "line": line_no, "question_sha256": text_sha256(record.get("question"))}
                    )
                    continue
                if query_id in seen_query_ids[agent]:
                    raise ValueError(f"duplicate query {query_id} in {agent}")
                seen_query_ids[agent].add(query_id)
                parsed = parse_trajectory(
                    record,
                    agent=agent,
                    trajectory_id=f"{agent}:{query_id}",
                    query_id=query_id,
                    evidence_docs=evidence.get(query_id, set()),
                    gold_docs=gold.get(query_id, set()),
                )
                for row in parsed.rows:
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                totals.update(parsed.stats)
                agent_counts.update(parsed.stats)

    for agent, ids in seen_query_ids.items():
        per_agent[agent]["matched_queries"] = len(ids)
        per_agent[agent]["missing_queries"] = len(queries) - len(ids)
    totals["matched_trajectory_rows"] = sum(len(ids) for ids in seen_query_ids.values())
    totals["unmatched_trajectory_rows"] = len(unmatched_questions)
    trajectory_inputs = [
        {"path": str(path), "run_id": path.parent.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in input_files
    ]
    manifest_validation: dict[str, Any] | None = None
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        expected = {str(run["run_id"]): run for run in manifest.get("runs", [])}
        observed = {item["run_id"]: item for item in trajectory_inputs}
        checks = []
        for run_id in sorted(set(expected) | set(observed)):
            exp = expected.get(run_id)
            obs = observed.get(run_id)
            counts = per_agent.get(run_id, {})
            checks.append(
                {
                    "run_id": run_id,
                    "present": exp is not None and obs is not None,
                    "bytes_match": bool(exp and obs and int(exp["bytes"]) == int(obs["bytes"])),
                    "sha256_match": bool(exp and obs and exp["sha256"] == obs["sha256"]),
                    "rows_match": bool(exp and int(exp["rows_out"]) == counts.get("trajectory_rows", 0)),
                    "terminations_match": bool(
                        exp
                        and all(
                            counts.get(f"termination_{name}", 0) == int(count)
                            for name, count in exp.get("terminations", {}).items()
                        )
                    ),
                }
            )
        manifest_validation = {
            "manifest": str(args.manifest),
            "checks": checks,
            "all_checks_pass": all(all(value for key, value in check.items() if key != "run_id") for check in checks),
        }
        if not manifest_validation["all_checks_pass"]:
            raise ValueError("trajectory files do not match the official manifest")

    summary = {
        "schema_version": "candidate_audit.v1",
        "grain": "one document occurrence in one search observation",
        "definitions": {
            "visited": "visit observation after this search and before the next search observation",
            "later_visited": "not visited before the next search, but visited after a later search observation",
            "utilized": "null because use in reasoning is not directly observable",
            "human_relevance": "gold/evidence qrel membership; otherwise not_in_known_qrels",
            "strict_answer_match_proxy": "normalized gold-answer text occurs in prediction; not the official judge",
        },
        "inputs": {
            "queries": str(args.queries),
            "evidence_qrels": str(args.evidence_qrels),
            "gold_qrels": str(args.gold_qrels),
            "trajectory_files": trajectory_inputs,
        },
        "manifest_validation": manifest_validation,
        "query_count": len(queries),
        "totals": dict(totals),
        "per_agent": {agent: dict(counts) for agent, counts in sorted(per_agent.items())},
        "unmatched_questions": unmatched_questions[:100],
        "output": str(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--evidence-qrels", type=Path, required=True)
    parser.add_argument("--gold-qrels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--agents", nargs="+", help="optional run ids for a scoped audit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(args)
    print(json.dumps(summary["totals"], sort_keys=True))


if __name__ == "__main__":
    main()
