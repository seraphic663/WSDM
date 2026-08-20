#!/usr/bin/env python3
"""Analyze the candidate table with query-cluster bootstrap uncertainty.

This script treats evidence-qrel membership as the primary relevance outcome.
Behavioral fields (visit, later visit, repeated exposure, cross-agent agreement)
remain separate conditioning variables and are never relabeled as ground truth.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import html
import json
from pathlib import Path
from typing import Any

import numpy as np


def register_audit_fonts() -> tuple[str, str]:
    """Register an explicitly embedded TrueType font for vector PDF figures."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ]
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            pdfmetrics.registerFont(TTFont("AuditSans", str(regular)))
            pdfmetrics.registerFont(TTFont("AuditSansBold", str(bold)))
            return "AuditSans", "AuditSansBold"
    raise FileNotFoundError("no embeddable TrueType font found for PDF figures")


def iter_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"row {line_no} is not an object")
                yield value


def rank_bucket(rank: int) -> str:
    return "1" if rank == 1 else "2" if rank == 2 else "3-5"


def search_phase(progress: float) -> str:
    if progress <= 1 / 3:
        return "early"
    if progress <= 2 / 3:
        return "middle"
    return "late"


def exposure_bucket(count: int) -> str:
    return "1" if count <= 1 else "2" if count == 2 else "3-4" if count <= 4 else "5-6"


def cluster_ratio_ci(
    query_counts: dict[str, list[int]],
    query_ids: list[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    numerator = np.array([query_counts.get(qid, [0, 0])[0] for qid in query_ids], dtype=np.int64)
    denominator = np.array([query_counts.get(qid, [0, 0])[1] for qid in query_ids], dtype=np.int64)
    total_numerator = int(numerator.sum())
    total_denominator = int(denominator.sum())
    estimate = total_numerator / total_denominator if total_denominator else None
    if not total_denominator:
        return {
            "estimate": None,
            "ci95": [None, None],
            "query_balanced_estimate": None,
            "query_balanced_ci95": [None, None],
            "relevant_candidates": 0,
            "candidate_occurrences": 0,
            "queries_with_candidates": 0,
        }
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    query_draws: list[np.ndarray] = []
    query_rates = np.divide(
        numerator.astype(float), denominator,
        out=np.full(len(query_ids), np.nan, dtype=float), where=denominator > 0,
    )
    batch_size = min(500, samples)
    for offset in range(0, samples, batch_size):
        current = min(batch_size, samples - offset)
        indices = rng.integers(0, len(query_ids), size=(current, len(query_ids)))
        draw_denominator = denominator[indices].sum(axis=1)
        draw_numerator = numerator[indices].sum(axis=1)
        valid = draw_denominator > 0
        draws.append(draw_numerator[valid] / draw_denominator[valid])
        query_draws.append(np.nanmean(query_rates[indices], axis=1))
    distribution = np.concatenate(draws)
    query_distribution = np.concatenate(query_draws)
    low, high = np.quantile(distribution, [0.025, 0.975]).tolist()
    query_low, query_high = np.quantile(query_distribution, [0.025, 0.975]).tolist()
    return {
        "estimate": estimate,
        "ci95": [low, high],
        "query_balanced_estimate": float(np.nanmean(query_rates)),
        "query_balanced_ci95": [query_low, query_high],
        "relevant_candidates": total_numerator,
        "candidate_occurrences": total_denominator,
        "queries_with_candidates": int(np.count_nonzero(denominator)),
    }


def cluster_difference_ci(
    left: dict[str, list[int]],
    right: dict[str, list[int]],
    query_ids: list[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    arrays = []
    for source in (left, right):
        arrays.append(
            (
                np.array([source.get(qid, [0, 0])[0] for qid in query_ids], dtype=np.int64),
                np.array([source.get(qid, [0, 0])[1] for qid in query_ids], dtype=np.int64),
            )
        )
    (left_num, left_den), (right_num, right_den) = arrays
    left_est = left_num.sum() / left_den.sum() if left_den.sum() else np.nan
    right_est = right_num.sum() / right_den.sum() if right_den.sum() else np.nan
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    batch_size = min(500, samples)
    for offset in range(0, samples, batch_size):
        current = min(batch_size, samples - offset)
        indices = rng.integers(0, len(query_ids), size=(current, len(query_ids)))
        ln = left_num[indices].sum(axis=1)
        ld = left_den[indices].sum(axis=1)
        rn = right_num[indices].sum(axis=1)
        rd = right_den[indices].sum(axis=1)
        valid = (ld > 0) & (rd > 0)
        draws.append(ln[valid] / ld[valid] - rn[valid] / rd[valid])
    distribution = np.concatenate(draws)
    low, high = np.quantile(distribution, [0.025, 0.975]).tolist()
    return {
        "left_estimate": float(left_est),
        "right_estimate": float(right_est),
        "difference": float(left_est - right_est),
        "ci95": [low, high],
        "excludes_zero": bool(low > 0 or high < 0),
    }


def add_count(container, dimension: str, category: str, query_id: str, relevant: bool) -> None:
    cell = container[dimension][category][query_id]
    cell[0] += int(relevant)
    cell[1] += 1


def first_pass(path: Path):
    query_agent = collections.defaultdict(lambda: {"docs": set(), "evidence": set(), "gold": set()})
    query_doc = collections.defaultdict(lambda: {"surfaced_agents": set(), "visited_agents": set()})
    query_ids: set[str] = set()
    row_count = 0
    for row in iter_rows(path):
        row_count += 1
        query_id = str(row["query_id"])
        agent = str(row["agent"])
        doc_id = str(row["doc_id"])
        query_ids.add(query_id)
        qa = query_agent[(query_id, agent)]
        qa["docs"].add(doc_id)
        if row["evidence_qrel_positive"]:
            qa["evidence"].add(doc_id)
        if row["gold_qrel_positive"]:
            qa["gold"].add(doc_id)
        qd = query_doc[(query_id, doc_id)]
        qd["surfaced_agents"].add(agent)
        if row["visited_ever_after_exposure"] or row["visited_before_exposure"]:
            qd["visited_agents"].add(agent)
    agents = sorted({agent for _, agent in query_agent})
    gold_success_agents = collections.Counter()
    for (query_id, _), values in query_agent.items():
        gold_success_agents[query_id] += bool(values["gold"])
    difficulty = {}
    for query_id in query_ids:
        successes = gold_success_agents[query_id]
        difficulty[query_id] = "hard (0-1 agents)" if successes <= 1 else "medium (2-4 agents)" if successes <= 4 else "easy (5-6 agents)"
    return sorted(query_ids), agents, query_agent, query_doc, difficulty, row_count


def write_svg(path: Path, coverage: list[tuple[str, float]], reliability: list[tuple[str, float, float, float]]) -> None:
    width, height = 1120, 520
    left_x, right_x = 60, 590
    panel_w = 470
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:20px;font-weight:700}.label{font-size:13px}.tick{font-size:11px;fill:#555}</style>',
        f'<text class="title" x="{left_x}" y="34">Stage 0: parsed search coverage</text>',
        f'<text class="title" x="{right_x}" y="34">Stage 1: evidence-qrel rate among unvisited</text>',
    ]
    bar_h = 42
    for idx, (label, value) in enumerate(coverage):
        y = 66 + idx * 68
        bar_w = panel_w * max(0.0, min(1.0, value))
        lines.extend(
            [
                f'<text class="label" x="{left_x}" y="{y}">{html.escape(label)}</text>',
                f'<rect x="{left_x}" y="{y + 9}" width="{panel_w}" height="{bar_h}" fill="#e8edf2" rx="3"/>',
                f'<rect x="{left_x}" y="{y + 9}" width="{bar_w:.1f}" height="{bar_h}" fill="#2878b5" rx="3"/>',
                f'<text class="label" x="{left_x + 8}" y="{y + 36}" fill="white">{value * 100:.2f}%</text>',
            ]
        )
    max_value = max((high for _, _, _, high in reliability), default=0.01) * 1.15
    chart_top, chart_bottom = 70, 460
    chart_h = chart_bottom - chart_top
    for tick in range(6):
        value = max_value * tick / 5
        y = chart_bottom - chart_h * tick / 5
        lines.append(f'<line x1="{right_x}" y1="{y:.1f}" x2="{right_x + panel_w}" y2="{y:.1f}" stroke="#ddd"/>')
        lines.append(f'<text class="tick" x="{right_x - 8}" y="{y + 4:.1f}" text-anchor="end">{value * 100:.1f}%</text>')
    slot = panel_w / max(1, len(reliability))
    for idx, (label, estimate, low, high) in enumerate(reliability):
        cx = right_x + slot * (idx + 0.5)
        y = chart_bottom - chart_h * estimate / max_value
        low_y = chart_bottom - chart_h * low / max_value
        high_y = chart_bottom - chart_h * high / max_value
        lines.extend(
            [
                f'<line x1="{cx:.1f}" y1="{low_y:.1f}" x2="{cx:.1f}" y2="{high_y:.1f}" stroke="#c43c39" stroke-width="3"/>',
                f'<line x1="{cx - 6:.1f}" y1="{low_y:.1f}" x2="{cx + 6:.1f}" y2="{low_y:.1f}" stroke="#c43c39" stroke-width="2"/>',
                f'<line x1="{cx - 6:.1f}" y1="{high_y:.1f}" x2="{cx + 6:.1f}" y2="{high_y:.1f}" stroke="#c43c39" stroke-width="2"/>',
                f'<circle cx="{cx:.1f}" cy="{y:.1f}" r="6" fill="#c43c39"/>',
                f'<text class="label" x="{cx:.1f}" y="{chart_bottom + 24}" text-anchor="middle">{html.escape(label)}</text>',
                f'<text class="tick" x="{cx:.1f}" y="{chart_bottom + 42}" text-anchor="middle">{estimate * 100:.2f}%</text>',
            ]
        )
    lines.append('</svg>')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pdf(path: Path, coverage: list[tuple[str, float]], reliability: list[tuple[str, float, float, float]]) -> None:
    """Write the same compact result panel as a LaTeX-ready vector PDF."""
    from reportlab.lib.colors import HexColor, black
    from reportlab.pdfgen import canvas

    width, height = 720, 330
    regular_font, bold_font = register_audit_fonts()
    pdf = canvas.Canvas(str(path), pagesize=(width, height))
    pdf.setTitle("Coverage and conditional reliability")
    pdf.setFont(bold_font, 12)
    pdf.drawString(35, 305, "Stage 0: parsed coverage")
    pdf.drawString(385, 305, "Stage 1: evidence-qrel rate among unvisited")
    panel_w = 290
    for idx, (label, value) in enumerate(coverage):
        y = 258 - idx * 58
        pdf.setFont(regular_font, 9)
        pdf.setFillColor(black)
        pdf.drawString(35, y + 24, label)
        pdf.setFillColor(HexColor("#e8edf2"))
        pdf.roundRect(35, y, panel_w, 18, 2, fill=1, stroke=0)
        pdf.setFillColor(HexColor("#2878b5"))
        pdf.roundRect(35, y, panel_w * max(0.0, min(1.0, value)), 18, 2, fill=1, stroke=0)
        pdf.setFillColor(black)
        pdf.drawRightString(325, y + 5, f"{value * 100:.2f}%")

    chart_left, chart_bottom, chart_w, chart_h = 385, 55, 300, 220
    max_value = max((high for _, _, _, high in reliability), default=0.01) * 1.15
    pdf.setStrokeColor(HexColor("#dddddd"))
    for tick in range(6):
        value = max_value * tick / 5
        y = chart_bottom + chart_h * tick / 5
        pdf.line(chart_left, y, chart_left + chart_w, y)
        pdf.setFillColor(black)
        pdf.setFont(regular_font, 7)
        pdf.drawRightString(chart_left - 4, y - 2, f"{value * 100:.1f}%")
    slot = chart_w / max(1, len(reliability))
    for idx, (label, estimate, low, high) in enumerate(reliability):
        x = chart_left + slot * (idx + 0.5)
        y = chart_bottom + chart_h * estimate / max_value
        low_y = chart_bottom + chart_h * low / max_value
        high_y = chart_bottom + chart_h * high / max_value
        pdf.setStrokeColor(HexColor("#c43c39"))
        pdf.setLineWidth(1.5)
        pdf.line(x, low_y, x, high_y)
        pdf.line(x - 3, low_y, x + 3, low_y)
        pdf.line(x - 3, high_y, x + 3, high_y)
        pdf.setFillColor(HexColor("#c43c39"))
        pdf.circle(x, y, 3.5, fill=1, stroke=0)
        pdf.setFillColor(black)
        pdf.setFont(regular_font, 6.5)
        pdf.drawCentredString(x, 38, label)
        pdf.drawCentredString(x, 28, f"{estimate * 100:.2f}%")
    pdf.setFont(regular_font, 7)
    pdf.drawString(385, 12, "Points: pooled candidate rate; bars: 95% query-cluster bootstrap CI.")
    pdf.showPage()
    pdf.save()


def write_column_pdf(path: Path, coverage: list[tuple[str, float]], reliability: list[tuple[str, float, float, float]]) -> None:
    """Write a compact single-column vector figure for the four-page paper."""
    from reportlab.lib.colors import HexColor, black
    from reportlab.pdfgen import canvas

    width, height = 360, 300
    regular_font, bold_font = register_audit_fonts()
    pdf = canvas.Canvas(str(path), pagesize=(width, height))
    pdf.setTitle("Coverage and conditional reliability")
    pdf.setFont(bold_font, 8)
    pdf.drawString(28, 286, "Coverage")
    bar_w = 185
    for idx, (label, value) in enumerate(coverage):
        y = 262 - idx * 24
        pdf.setFont(regular_font, 6)
        pdf.setFillColor(black)
        pdf.drawString(28, y + 8, label)
        pdf.setFillColor(HexColor("#e8edf2"))
        pdf.roundRect(120, y + 5, bar_w, 7, 1, fill=1, stroke=0)
        pdf.setFillColor(HexColor("#2878b5"))
        pdf.roundRect(120, y + 5, bar_w * value, 7, 1, fill=1, stroke=0)
        pdf.setFillColor(black)
        pdf.drawRightString(333, y + 6, f"{value * 100:.2f}%")

    pdf.setFont(bold_font, 8)
    pdf.drawString(28, 164, "Evidence-qrel rate among unvisited candidates")
    left, bottom, chart_w, chart_h = 45, 38, 285, 112
    max_value = max(high for _, _, _, high in reliability) * 1.12
    pdf.setStrokeColor(HexColor("#dddddd"))
    for tick in range(5):
        value = max_value * tick / 4
        y = bottom + chart_h * tick / 4
        pdf.line(left, y, left + chart_w, y)
        pdf.setFillColor(black)
        pdf.setFont(regular_font, 5.5)
        pdf.drawRightString(left - 3, y - 2, f"{value * 100:.0f}%")
    slot = chart_w / len(reliability)
    for idx, (label, estimate, low, high) in enumerate(reliability):
        x = left + slot * (idx + 0.5)
        y = bottom + chart_h * estimate / max_value
        low_y = bottom + chart_h * low / max_value
        high_y = bottom + chart_h * high / max_value
        pdf.setStrokeColor(HexColor("#c43c39"))
        pdf.setLineWidth(1.2)
        pdf.line(x, low_y, x, high_y)
        pdf.line(x - 2.5, low_y, x + 2.5, low_y)
        pdf.line(x - 2.5, high_y, x + 2.5, high_y)
        pdf.setFillColor(HexColor("#c43c39"))
        pdf.circle(x, y, 2.7, fill=1, stroke=0)
        pdf.setFillColor(black)
        pdf.setFont(regular_font, 5)
        pdf.drawCentredString(x, 25, label)
        pdf.drawCentredString(x, 17, f"{estimate * 100:.1f}%")
    pdf.setFont(regular_font, 5.5)
    pdf.drawString(28, 5, "Points: pooled rate; bars: 95% question-cluster bootstrap CI.")
    pdf.showPage()
    pdf.save()


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    query_ids, agents, query_agent, query_doc, difficulty, row_count = first_pass(args.candidates)
    grouped = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0])))
    grouped_gold = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0])))
    quality = collections.Counter()
    for row in iter_rows(args.candidates):
        quality["candidate_rows"] += 1
        if row["visited"]:
            quality["visited_before_next_search"] += 1
            continue
        quality["unvisited_before_next_search"] += 1
        relevant = bool(row["evidence_qrel_positive"])
        gold_relevant = bool(row["gold_qrel_positive"])
        query_id = str(row["query_id"])
        doc_key = (query_id, str(row["doc_id"]))
        cross = query_doc[doc_key]
        surfaced_count = len(cross["surfaced_agents"])
        visited_count = len(cross["visited_agents"])
        visit_agreement = "never visited" if visited_count == 0 else "all surfaced agents visited" if visited_count == surfaced_count else "mixed visit decisions"
        categories = {
            "overall": "all",
            "agent": str(row["agent"]),
            "rank": rank_bucket(int(row["rank"])),
            "search_phase": search_phase(float(row["search_progress"])),
            "evidence_state": str(row["evidence_state"]),
            "repeated_retrieval": str(bool(row["repeated_retrieval"])).lower(),
            "later_visited": str(bool(row["later_visited"])).lower(),
            "query_difficulty": difficulty[query_id],
            "termination": str(row["termination"]),
            "strict_answer_match_proxy": str(row["strict_answer_match_proxy"]).lower(),
            "cross_agent_exposure": exposure_bucket(surfaced_count),
            "cross_agent_visit_agreement": visit_agreement,
        }
        for dimension, category in categories.items():
            add_count(grouped, dimension, category, query_id, relevant)
            add_count(grouped_gold, dimension, category, query_id, gold_relevant)
        if not bool(row["repeated_retrieval"]):
            category = str(bool(row["later_visited"])).lower()
            add_count(grouped, "later_visit_first_exposure", category, query_id, relevant)
            add_count(grouped_gold, "later_visit_first_exposure", category, query_id, gold_relevant)
        if not bool(row["later_visited"]):
            category = str(bool(row["repeated_retrieval"])).lower()
            add_count(grouped, "repeated_without_later_visit", category, query_id, relevant)
            add_count(grouped_gold, "repeated_without_later_visit", category, query_id, gold_relevant)
        if int(row["prior_exposures_in_trajectory"]) == 0:
            add_count(grouped, "robustness", "first exposure only", query_id, relevant)
            add_count(grouped_gold, "robustness", "first exposure only", query_id, gold_relevant)

    results: dict[str, dict[str, Any]] = {}
    gold_results: dict[str, dict[str, Any]] = {}
    seed_offset = 0
    for dimension in sorted(grouped):
        results[dimension] = {}
        for category in sorted(grouped[dimension]):
            results[dimension][category] = cluster_ratio_ci(
                grouped[dimension][category], query_ids, samples=args.bootstrap_samples, seed=args.seed + seed_offset
            )
            seed_offset += 1
    for dimension in sorted(grouped_gold):
        gold_results[dimension] = {}
        for category in sorted(grouped_gold[dimension]):
            gold_results[dimension][category] = cluster_ratio_ci(
                grouped_gold[dimension][category], query_ids,
                samples=args.bootstrap_samples, seed=args.seed + 20_000 + seed_offset,
            )
            seed_offset += 1

    contrast_specs = [
        ("later_visited", "true", "false"),
        ("repeated_retrieval", "true", "false"),
        ("rank", "1", "3-5"),
        ("evidence_state", "partial", "none"),
        ("cross_agent_exposure", "5-6", "1"),
        ("cross_agent_visit_agreement", "mixed visit decisions", "never visited"),
        ("later_visit_first_exposure", "true", "false"),
        ("repeated_without_later_visit", "true", "false"),
    ]
    contrasts = {}
    gold_contrasts = {}
    for idx, (dimension, left, right) in enumerate(contrast_specs):
        if left in grouped[dimension] and right in grouped[dimension]:
            contrasts[f"{dimension}:{left}_minus_{right}"] = cluster_difference_ci(
                grouped[dimension][left], grouped[dimension][right], query_ids,
                samples=args.bootstrap_samples, seed=args.seed + 10_000 + idx,
            )
        if left in grouped_gold[dimension] and right in grouped_gold[dimension]:
            gold_contrasts[f"{dimension}:{left}_minus_{right}"] = cluster_difference_ci(
                grouped_gold[dimension][left], grouped_gold[dimension][right], query_ids,
                samples=args.bootstrap_samples, seed=args.seed + 30_000 + idx,
            )

    agent_coverage = {}
    for agent in agents:
        values = [query_agent[(qid, agent)] for qid in query_ids if (qid, agent) in query_agent]
        agent_coverage[agent] = {
            "queries": len(values),
            "queries_with_any_evidence_qrel": sum(bool(value["evidence"]) for value in values),
            "queries_with_any_gold_qrel": sum(bool(value["gold"]) for value in values),
            "unique_docs_mean": sum(len(value["docs"]) for value in values) / len(values) if values else None,
        }

    stage0 = json.loads(args.stage0_summary.read_text(encoding="utf-8"))
    totals = stage0["totals"]
    recoverable_searches = totals.get("parsed_search_observations", 0) + totals.get("unrecognized_search_observations", 0)
    search_parse_rate = totals.get("parsed_search_observations", 0) / max(1, recoverable_searches)
    visit_parse_rate = totals.get("parsed_visit_observations", 0) / max(1, totals.get("visit_observations", 0))
    visit_map_rate = totals.get("successful_visit_observations_mapped_to_any_surface", 0) / max(1, totals.get("successful_visit_observations", 0))
    complete_agents = sum(v.get("matched_queries", 0) == stage0["query_count"] for v in stage0["per_agent"].values())
    coverage_gate = (
        complete_agents == 6
        and stage0["totals"].get("unmatched_trajectory_rows", 1) == 0
        and search_parse_rate >= 0.99
        and visit_parse_rate >= 0.99
        and visit_map_rate >= 0.99
    )
    stable_behavioral = {
        name: value
        for name, value in contrasts.items()
        if name.startswith(("later_visited:", "repeated_retrieval:", "cross_agent_"))
        and value["excludes_zero"]
        and abs(value["difference"]) >= 0.005
    }
    signal_gate = bool(stable_behavioral)
    decision = "GO" if coverage_gate and signal_gate else "PIVOT"

    summary = {
        "schema_version": "negative_reliability_audit.v1",
        "outcome": "evidence_qrel_positive",
        "conditioning_population": "candidate occurrences not visited before the next search observation",
        "uncertainty": {
            "method": "nonparametric cluster bootstrap",
            "cluster": "query_id",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "candidate_rows": row_count,
        "quality": dict(quality),
        "agent_retrieval_coverage": agent_coverage,
        "stratified_results": results,
        "gold_qrel_sensitivity": {
            "stratified_results": gold_results,
            "pre_specified_contrasts": gold_contrasts,
        },
        "pre_specified_contrasts": contrasts,
        "gate": {
            "decision": decision,
            "coverage_gate": coverage_gate,
            "signal_gate": signal_gate,
            "stable_behavioral_contrasts": stable_behavioral,
            "rule": "GO requires complete six-agent coverage, >=99% search/visit parsing and visit mapping, plus a behavioral contrast with query-bootstrap CI excluding zero and absolute difference >=0.5 percentage points.",
        },
        "limitations": [
            "Unlisted documents are benchmark negatives under the qrel evaluation protocol, but not proven universally irrelevant.",
            "A visit is behavioral evidence, not a relevance judgment.",
            "The strict answer-match field is a text proxy, not the official GPT-4o judge label.",
            "Utilization in reasoning is not directly observed and remains null in the candidate table.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    coverage = [
        ("question mapping", totals.get("matched_trajectory_rows", 0) / max(1, totals.get("trajectory_rows", 0))),
        ("search observations", search_parse_rate),
        ("visit arguments", visit_parse_rate),
        ("visits mapped", visit_map_rate),
    ]
    plot_keys = [
        ("overall", "all", "Overall"),
        ("later_visited", "false", "Never/later no"),
        ("later_visited", "true", "Later visited"),
        ("repeated_retrieval", "true", "Repeated"),
        ("rank", "1", "Rank 1"),
    ]
    reliability = []
    for dimension, category, label in plot_keys:
        value = results.get(dimension, {}).get(category)
        if value and value["estimate"] is not None:
            reliability.append((label, value["estimate"], value["ci95"][0], value["ci95"][1]))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    write_svg(args.figure, coverage, reliability)
    args.figure_pdf.parent.mkdir(parents=True, exist_ok=True)
    write_pdf(args.figure_pdf, coverage, reliability)
    args.figure_column_pdf.parent.mkdir(parents=True, exist_ok=True)
    write_column_pdf(args.figure_column_pdf, coverage, reliability)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--stage0-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--figure-pdf", type=Path, required=True)
    parser.add_argument("--figure-column-pdf", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(args)
    print(json.dumps(summary["gate"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
