# Negative Reliability Audit — Stage 0 and Stage 1

Date: 2026-08-20

## Outcome

The pre-registered diagnostic gate returns **PIVOT**. The primary outcome is evidence-qrel membership among candidate occurrences not visited before the next search observation. All 95% intervals use 10,000 nonparametric bootstrap samples with `query_id` as the resampling cluster.

This audit does not call every unlisted document truly irrelevant. `not_in_known_qrels` means only that the document is absent from the benchmark's known evidence and gold sets. Visits, later visits, repeated retrieval and answer-text matching remain behavior variables or proxies, never replacements for qrels.

## Stage 0 coverage

The official six-run release contributes 4,980 trajectory rows and 638,215 candidate occurrences. Exact normalized question text maps 4,980/4,980 trajectories to the 830 BrowseComp-Plus qids; numeric qids from the unrelated InfoSeekQA-based LRAT archive are never used for this join.

| Check | Recovered | Total | Rate |
|---|---:|---:|---:|
| Question mapping | 4,980 | 4,980 | 100.00% |
| Recoverable successful search observations | 127,591 | 127,591 | 100.00% |
| Visit arguments parsed | 9,796 | 9,807 | 99.89% |
| Successful visits mapped to a surfaced document in the same trajectory | 8,587 | 8,684 | 98.88% |

There are 9 local search failures, 2,661 invalid search calls, and 0 otherwise unrecognized search observations. There are 1,123 failed visits and 97 successful visits whose document cannot be recovered from a successful search observation in the same trajectory. Failed visits are not treated as document visits; all such events remain in the coverage report and generate no fabricated candidates.

### Per-agent reconstruction

| Agent | Queries | Candidates | Search recovery | Visit mapping |
|---|---:|---:|---:|---:|
| ds4pro | 830 | 119,240 | 100.00% | 100.00% |
| glm51 | 830 | 82,455 | 100.00% | 100.00% |
| gpt-oss-120b-high | 830 | 132,025 | 100.00% | 98.95% |
| kimi26 | 830 | 113,140 | 100.00% | 99.90% |
| qwen35-35b-a3b | 830 | 89,435 | 100.00% | 99.85% |
| tongyi-dr | 830 | 101,920 | 100.00% | 96.25% |

## Stage 1 conditional relevance

| Condition | Unvisited candidate occurrences | Queries | Evidence-qrel rate | Query-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Overall | 633,083 | 830 | 13.77% | [12.83%, 14.75%] |
| Later visited: no | 614,217 | 830 | 12.14% | [11.27%, 13.04%] |
| Later visited: yes | 18,866 | 820 | 66.69% | [63.44%, 69.85%] |
| Later visited on first exposure: no | 239,229 | 830 | 5.79% | [5.41%, 6.20%] |
| Later visited on first exposure: yes | 5,093 | 820 | 73.79% | [71.51%, 76.05%] |
| Repeated retrieval: no | 244,322 | 830 | 7.21% | [6.76%, 7.68%] |
| Repeated retrieval: yes | 388,761 | 830 | 17.89% | [16.59%, 19.25%] |
| Repeated, excluding all later visits: no | 239,229 | 830 | 5.79% | [5.41%, 6.21%] |
| Repeated, excluding all later visits: yes | 374,988 | 830 | 16.20% | [14.98%, 17.47%] |
| Rank 1 | 125,279 | 830 | 23.77% | [22.20%, 25.45%] |
| Rank 2 | 126,564 | 830 | 16.38% | [15.19%, 17.61%] |
| Rank 3-5 | 381,240 | 830 | 9.62% | [8.90%, 10.40%] |
| Early search | 196,484 | 830 | 9.24% | [8.43%, 10.07%] |
| Middle search | 214,301 | 830 | 13.21% | [12.22%, 14.26%] |
| Late search | 222,298 | 830 | 18.31% | [17.17%, 19.55%] |
| No evidence seen before | 263,886 | 830 | 2.03% | [1.82%, 2.26%] |
| Partial evidence seen before | 314,113 | 805 | 19.76% | [18.51%, 21.04%] |
| Complete evidence seen before | 55,084 | 451 | 35.86% | [33.03%, 38.81%] |
| First exposure only | 244,322 | 830 | 7.21% | [6.77%, 7.69%] |
| Hard query (gold surfaced by 0-1 agents) | 117,161 | 91 | 2.96% | [1.95%, 4.15%] |
| Medium query (gold surfaced by 2-4 agents) | 209,139 | 202 | 7.50% | [6.55%, 8.49%] |
| Easy query (gold surfaced by 5-6 agents) | 306,783 | 537 | 22.17% | [20.83%, 23.56%] |

The query-balanced mean of within-query evidence-qrel rates is 19.69%, with query-bootstrap 95% CI [18.50%, 20.92%]. This differs from the pooled candidate rate because long trajectories contribute more candidate occurrences to the pooled estimand.

### Agent variation

| Agent | Candidates | Queries | Evidence-qrel rate | Query-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| ds4pro | 118,298 | 830 | 12.53% | [11.45%, 13.66%] |
| glm51 | 81,646 | 830 | 16.21% | [14.91%, 17.58%] |
| gpt-oss-120b-high | 131,245 | 830 | 13.00% | [11.76%, 14.30%] |
| kimi26 | 112,511 | 829 | 15.80% | [14.64%, 17.03%] |
| qwen35-35b-a3b | 88,684 | 830 | 11.97% | [10.92%, 13.12%] |
| tongyi-dr | 100,699 | 829 | 13.57% | [12.36%, 14.81%] |

### Pre-specified contrasts

| Contrast | Difference | 95% CI | Excludes zero |
|---|---:|---:|:---:|
| cross_agent_exposure:5-6_minus_1 | 22.76% | [20.88%, 24.69%] | yes |
| cross_agent_visit_agreement:mixed visit decisions_minus_never visited | 54.56% | [51.22%, 57.85%] | yes |
| evidence_state:partial_minus_none | 17.74% | [16.52%, 18.99%] | yes |
| later_visit_first_exposure:true_minus_false | 68.00% | [65.87%, 70.11%] | yes |
| later_visited:true_minus_false | 54.54% | [51.66%, 57.43%] | yes |
| rank:1_minus_3-5 | 14.15% | [12.99%, 15.43%] | yes |
| repeated_retrieval:true_minus_false | 10.68% | [9.66%, 11.75%] | yes |
| repeated_without_later_visit:true_minus_false | 10.40% | [9.41%, 11.45%] | yes |

### Gold-qrel sensitivity

The stricter gold-qrel membership analysis uses the same populations and query-cluster bootstrap. It is a sensitivity analysis because evidence qrels are the primary retrieval-side relevance definition.

| Condition | Candidates | Queries | Gold-qrel rate | Query-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Overall | 633,083 | 830 | 6.75% | [6.17%, 7.37%] |
| Later visited: no | 614,217 | 830 | 5.72% | [5.21%, 6.26%] |
| Later visited: yes | 18,866 | 820 | 40.19% | [37.19%, 43.22%] |
| Repeated retrieval: no | 244,322 | 830 | 3.43% | [3.16%, 3.71%] |
| Repeated retrieval: yes | 388,761 | 830 | 8.84% | [8.01%, 9.69%] |

## Gate interpretation

Coverage gate: **fail**. Behavioral-signal gate: **pass**. Rule: GO requires complete six-agent coverage, >=99% search/visit parsing and visit mapping, plus a behavioral contrast with query-bootstrap CI excluding zero and absolute difference >=0.5 percentage points.

A GO decision authorizes only the controlled intervention pilot. It does not establish that unvisited candidates are universally false negatives, that later visit causes relevance, or that filtering improves retriever training. Those claims require the Stage 3 random-removal and exposure controls.

## Reproducibility artifacts

- Candidate table: `analysis/negative_reliability/candidates.jsonl.gz`.
- Stage 0 machine-readable summary: `analysis/negative_reliability/stage0_summary.json`.
- Stage 1 machine-readable summary: `analysis/negative_reliability/stage1_summary.json`.
- Vector figures: `analysis/negative_reliability/coverage_reliability.svg` and `coverage_reliability.pdf`.
- Builders and tests: `solution/src/build_negative_reliability_audit.py`, `solution/src/analyze_negative_reliability.py`, and corresponding `solution/tests/` modules.

## Validity boundaries

- Unlisted documents are benchmark negatives under the qrel evaluation protocol, but not proven universally irrelevant.
- A visit is behavioral evidence, not a relevance judgment.
- The strict answer-match field is a text proxy, not the official GPT-4o judge label.
- Utilization in reasoning is not directly observed and remains null in the candidate table.
