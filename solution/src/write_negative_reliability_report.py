#!/usr/bin/env python3
"""Render the Stage 0/1 machine-readable audits as a concise Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(value):
    return "NA" if value is None else f"{100 * value:.2f}%"


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def result_row(label, value):
    low, high = value["ci95"]
    return (
        f"| {label} | {value['candidate_occurrences']:,} | {value['queries_with_candidates']:,} | "
        f"{pct(value['estimate'])} | [{pct(low)}, {pct(high)}] |"
    )


def render(stage0: dict, stage1: dict) -> str:
    totals = stage0["totals"]
    visit_parse = rate(totals.get("parsed_visit_observations", 0), totals.get("visit_observations", 0))
    recoverable_searches = totals.get("parsed_search_observations", 0) + totals.get("unrecognized_search_observations", 0)
    search_rate = rate(totals.get("parsed_search_observations", 0), recoverable_searches)
    visit_map = rate(totals.get("successful_visit_observations_mapped_to_any_surface", 0), totals.get("successful_visit_observations", 0))
    lines = [
        "# Negative Reliability Audit — Stage 0 and Stage 1",
        "",
        "Date: 2026-08-20",
        "",
        "## Outcome",
        "",
        f"The pre-registered diagnostic gate returns **{stage1['gate']['decision']}**. The primary outcome is evidence-qrel membership among candidate occurrences not visited before the next search observation. All 95% intervals use {stage1['uncertainty']['samples']:,} nonparametric bootstrap samples with `query_id` as the resampling cluster.",
        "",
        "This audit does not call every unlisted document truly irrelevant. `not_in_known_qrels` means only that the document is absent from the benchmark's known evidence and gold sets. Visits, later visits, repeated retrieval and answer-text matching remain behavior variables or proxies, never replacements for qrels.",
        "",
        "## Stage 0 coverage",
        "",
        f"The official six-run release contributes {totals.get('trajectory_rows', 0):,} trajectory rows and {totals.get('candidate_rows', 0):,} candidate occurrences. Exact normalized question text maps {totals.get('matched_trajectory_rows', 0):,}/{totals.get('trajectory_rows', 0):,} trajectories to the 830 BrowseComp-Plus qids; numeric qids from the unrelated InfoSeekQA-based LRAT archive are never used for this join.",
        "",
        "| Check | Recovered | Total | Rate |",
        "|---|---:|---:|---:|",
        f"| Question mapping | {totals.get('matched_trajectory_rows', 0):,} | {totals.get('trajectory_rows', 0):,} | {pct(rate(totals.get('matched_trajectory_rows', 0), totals.get('trajectory_rows', 0)))} |",
        f"| Recoverable successful search observations | {totals.get('parsed_search_observations', 0):,} | {recoverable_searches:,} | {pct(search_rate)} |",
        f"| Visit arguments parsed | {totals.get('parsed_visit_observations', 0):,} | {totals.get('visit_observations', 0):,} | {pct(visit_parse)} |",
        f"| Successful visits mapped to a surfaced document in the same trajectory | {totals.get('successful_visit_observations_mapped_to_any_surface', 0):,} | {totals.get('successful_visit_observations', 0):,} | {pct(visit_map)} |",
        "",
        f"There are {totals.get('failed_search_observations', 0):,} local search failures, {totals.get('invalid_search_request_observations', 0):,} invalid search calls, and {totals.get('unrecognized_search_observations', 0):,} otherwise unrecognized search observations. There are {totals.get('failed_visit_observations', 0):,} failed visits and {totals.get('successful_visit_observations_unmapped_to_any_surface', 0):,} successful visits whose document cannot be recovered from a successful search observation in the same trajectory. Failed visits are not treated as document visits; all such events remain in the coverage report and generate no fabricated candidates.",
        "",
        "### Per-agent reconstruction",
        "",
        "| Agent | Queries | Candidates | Search recovery | Visit mapping |",
        "|---|---:|---:|---:|---:|",
    ]
    for agent, value in stage0["per_agent"].items():
        lines.append(
            f"| {agent} | {value.get('matched_queries', 0):,} | {value.get('candidate_rows', 0):,} | "
            f"{pct(rate(value.get('parsed_search_observations', 0), value.get('parsed_search_observations', 0) + value.get('unrecognized_search_observations', 0)))} | "
            f"{pct(rate(value.get('successful_visit_observations_mapped_to_any_surface', 0), value.get('successful_visit_observations', 0)))} |"
        )
    lines.extend(
        [
            "",
            "## Stage 1 conditional relevance",
            "",
            "| Condition | Unvisited candidate occurrences | Queries | Evidence-qrel rate | Query-bootstrap 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    selected = [
        ("Overall", "overall", "all"),
        ("Later visited: no", "later_visited", "false"),
        ("Later visited: yes", "later_visited", "true"),
        ("Later visited on first exposure: no", "later_visit_first_exposure", "false"),
        ("Later visited on first exposure: yes", "later_visit_first_exposure", "true"),
        ("Repeated retrieval: no", "repeated_retrieval", "false"),
        ("Repeated retrieval: yes", "repeated_retrieval", "true"),
        ("Repeated, excluding all later visits: no", "repeated_without_later_visit", "false"),
        ("Repeated, excluding all later visits: yes", "repeated_without_later_visit", "true"),
        ("Rank 1", "rank", "1"),
        ("Rank 2", "rank", "2"),
        ("Rank 3-5", "rank", "3-5"),
        ("Early search", "search_phase", "early"),
        ("Middle search", "search_phase", "middle"),
        ("Late search", "search_phase", "late"),
        ("No evidence seen before", "evidence_state", "none"),
        ("Partial evidence seen before", "evidence_state", "partial"),
        ("Complete evidence seen before", "evidence_state", "complete"),
        ("First exposure only", "robustness", "first exposure only"),
        ("Hard query (gold surfaced by 0-1 agents)", "query_difficulty", "hard (0-1 agents)"),
        ("Medium query (gold surfaced by 2-4 agents)", "query_difficulty", "medium (2-4 agents)"),
        ("Easy query (gold surfaced by 5-6 agents)", "query_difficulty", "easy (5-6 agents)"),
    ]
    for label, dimension, category in selected:
        value = stage1["stratified_results"].get(dimension, {}).get(category)
        if value:
            lines.append(result_row(label, value))
    overall = stage1["stratified_results"]["overall"]["all"]
    lines.extend(["", f"The query-balanced mean of within-query evidence-qrel rates is {pct(overall['query_balanced_estimate'])}, with query-bootstrap 95% CI [{pct(overall['query_balanced_ci95'][0])}, {pct(overall['query_balanced_ci95'][1])}]. This differs from the pooled candidate rate because long trajectories contribute more candidate occurrences to the pooled estimand."])
    lines.extend(["", "### Agent variation", "", "| Agent | Candidates | Queries | Evidence-qrel rate | Query-bootstrap 95% CI |", "|---|---:|---:|---:|---:|"])
    for agent, value in stage1["stratified_results"]["agent"].items():
        lines.append(result_row(agent, value))
    lines.extend(["", "### Pre-specified contrasts", "", "| Contrast | Difference | 95% CI | Excludes zero |", "|---|---:|---:|:---:|"])
    for name, value in stage1["pre_specified_contrasts"].items():
        low, high = value["ci95"]
        lines.append(f"| {name} | {pct(value['difference'])} | [{pct(low)}, {pct(high)}] | {'yes' if value['excludes_zero'] else 'no'} |")
    gold = stage1["gold_qrel_sensitivity"]
    lines.extend(["", "### Gold-qrel sensitivity", "", "The stricter gold-qrel membership analysis uses the same populations and query-cluster bootstrap. It is a sensitivity analysis because evidence qrels are the primary retrieval-side relevance definition.", "", "| Condition | Candidates | Queries | Gold-qrel rate | Query-bootstrap 95% CI |", "|---|---:|---:|---:|---:|"])
    for label, dimension, category in [
        ("Overall", "overall", "all"),
        ("Later visited: no", "later_visited", "false"),
        ("Later visited: yes", "later_visited", "true"),
        ("Repeated retrieval: no", "repeated_retrieval", "false"),
        ("Repeated retrieval: yes", "repeated_retrieval", "true"),
    ]:
        value = gold["stratified_results"].get(dimension, {}).get(category)
        if value:
            lines.append(result_row(label, value))
    lines.extend(
        [
            "",
            "## Gate interpretation",
            "",
            f"Coverage gate: **{'pass' if stage1['gate']['coverage_gate'] else 'fail'}**. Behavioral-signal gate: **{'pass' if stage1['gate']['signal_gate'] else 'fail'}**. Rule: {stage1['gate']['rule']}",
            "",
            "A GO decision authorizes only the controlled intervention pilot. It does not establish that unvisited candidates are universally false negatives, that later visit causes relevance, or that filtering improves retriever training. Those claims require the Stage 3 random-removal and exposure controls.",
            "",
            "## Reproducibility artifacts",
            "",
            "- Candidate table: `analysis/negative_reliability/candidates.jsonl.gz`.",
            "- Stage 0 machine-readable summary: `analysis/negative_reliability/stage0_summary.json`.",
            "- Stage 1 machine-readable summary: `analysis/negative_reliability/stage1_summary.json`.",
            "- Vector figures: `analysis/negative_reliability/coverage_reliability.svg` and `coverage_reliability.pdf`.",
            "- Builders and tests: `solution/src/build_negative_reliability_audit.py`, `solution/src/analyze_negative_reliability.py`, and corresponding `solution/tests/` modules.",
            "",
            "## Validity boundaries",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in stage1["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0", required=True, type=Path)
    parser.add_argument("--stage1", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    stage0 = json.loads(args.stage0.read_text(encoding="utf-8"))
    stage1 = json.loads(args.stage1.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(stage0, stage1), encoding="utf-8")


if __name__ == "__main__":
    main()
