# Negative Reliability Audit

This directory is the durable evidence bundle for the 2026-08-20 WSDM Short audit. It uses only public BrowseComp-Plus qrels and the six-run trajectory release associated with *Diagnosing Search Behavior and Failure Modes in Long-Horizon Search Agents*. No teacher/advisor private data source was assumed or imported.

## Decision

The pre-registered joint gate is **PIVOT**. The behavioral signal gate passes, but successful-visit mapping is 8,587/8,684 = 98.88%, below the 99% coverage threshold. The deficit is concentrated in Tongyi-DR (96.25%); the other Agents range from 98.95% to 100%. No retriever intervention was trained, and there is no method-gain claim.

## Main results

- 4,980/4,980 trajectories map to the 830 qids by normalized question text.
- 127,591/127,591 recoverable successful Search observations yield ranked lists.
- 638,215 candidate occurrences are reconstructed; 633,083 are unvisited before the next Search.
- Evidence-qrel rate among unvisited occurrences: 13.77%, query-cluster 95% CI [12.81%, 14.74%].
- Query-balanced mean within-query rate: 19.69% [18.49%, 20.89%].
- Later visited versus not later visited: 66.69% versus 12.14%; difference 54.54 points [51.66, 57.43].
- On first exposure: 73.79% versus 5.79%; difference 68.00 points [65.87, 70.11].
- Gold-qrel sensitivity preserves direction: 40.19% versus 5.72% overall later-visit comparison; 44.55% versus 2.56% on first exposure.

## Artifacts

| File | Role |
|---|---|
| `candidates.jsonl.gz` | One row per document occurrence in one successful Search result list. |
| `stage0_summary.json` | Manifest identity, reconstruction counts, per-Agent coverage, and explicit tool failures. |
| `stage1_summary.json` | Evidence/gold results, denominators, query-balanced estimates, 10,000-sample query-cluster CIs, contrasts, and gate. |
| `stage0_stage1_report.md` | Human-readable report generated from the two summaries. |
| `coverage_reliability.svg` | Wide analysis figure. |
| `coverage_reliability.pdf` | Wide vector figure. |
| `coverage_reliability_column.pdf` | Single-column vector figure used by the four-page paper. |
| `stage2_split_audit.json` | Fresh source/split audit; confirms zero normalized-query overlap for `early_stop_v1`. |
| `stage2_alignment.md` | Boundary between BrowseComp diagnosis and existing LRAT/P0 experiments. |
| `stage3_rebuilt_split_manifest.json` | Local reconstruction of the exact 94,113-row LRAT train split. |
| `stage3_provenance_manifest.json` | Dry-run provenance: 93,977 stable, 136 ambiguous, 0 mismatch. |
| `stage3_arm_manifest.json` | Fair offline arms; random, exposure-prioritized, and later-visit each change 28,880 rows and remove 43,575 negatives. They are untrained. |

The candidate table deliberately stores `utilized = null`; utilization is not directly observable in the released data. `human_relevance = not_in_known_qrels` means absence from the benchmark qrels, not proof of universal irrelevance. Failed visits are not counted as successful visits.

## Reproduction

Run from `/home/seraphic/WSDM`. The Stage 1 analysis requires NumPy and ReportLab; the Codex bundled Python runtime was used for the current artifacts.

```bash
python3 solution/src/build_negative_reliability_audit.py \
  --trajectories-root data/raw/bcp_search_agent_trajectory \
  --queries LRAT/topics-qrels/queries.tsv \
  --evidence-qrels LRAT/topics-qrels/qrel_evidence.txt \
  --gold-qrels LRAT/topics-qrels/qrel_golds.txt \
  --manifest data/raw/bcp_search_agent_trajectory/manifest.json \
  --output analysis/negative_reliability/candidates.jsonl.gz \
  --summary analysis/negative_reliability/stage0_summary.json
```

```text
python solution/src/analyze_negative_reliability.py \
  --candidates analysis/negative_reliability/candidates.jsonl.gz \
  --stage0-summary analysis/negative_reliability/stage0_summary.json \
  --output analysis/negative_reliability/stage1_summary.json \
  --figure analysis/negative_reliability/coverage_reliability.svg \
  --figure-pdf analysis/negative_reliability/coverage_reliability.pdf \
  --figure-column-pdf analysis/negative_reliability/coverage_reliability_column.pdf \
  --bootstrap-samples 10000 --seed 20260820
```

```bash
python3 solution/src/write_negative_reliability_report.py \
  --stage0 analysis/negative_reliability/stage0_summary.json \
  --stage1 analysis/negative_reliability/stage1_summary.json \
  --output analysis/negative_reliability/stage0_stage1_report.md
```

Relevant tests are `solution/tests/test_build_negative_reliability_audit.py`, `test_analyze_negative_reliability.py`, `test_build_lrat_reliability_arms.py`, and `test_analyze_reliability_arm_grid.py`.
