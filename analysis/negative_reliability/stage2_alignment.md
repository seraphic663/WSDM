# Stage 2 — Existing Experiment Alignment

Date: 2026-08-20

## Decision summary

The existing P0 sweep is suitable as configuration evidence and as a source of controlled training defaults, but it cannot be joined row-by-row to the BrowseComp-Plus reliability audit. The two evidence sources use different questions: the released LRAT training trajectories use 10K InfoSeekQA seed queries, whereas the six-agent diagnostic release uses 830 BrowseComp-Plus questions. Any numeric-query-id join between them is invalid.

The defensible bridge is therefore an intervention transfer test: calibrate a behavior-only warning signal on BrowseComp-Plus qrels, apply the same observable rule to LRAT trajectories without importing BrowseComp labels, and compare the resulting LRAT training arm against vanilla and removal-count-matched controls on the fixed query-disjoint Dev1500 split.

## Locally re-verified split contract

The fresh audit in `stage2_split_audit.json` read the 3,883,089,616-byte released pair file directly. It found 96,504 rows and reproduced the source identity recorded by `early_stop_v1`. The proposed split contains 94,113 training rows, 1,500 development query groups and 500 locked-test query groups, with zero normalized-query overlap between training and the two held-out sets. The locked test was not read for method selection.

The older `data/processed/dev.jsonl` is not a valid independent holdout: its 500 records are exact records from the full released pair file, and their normalized queries collectively correspond to 605 source rows. It remains historical material only and must not be used for the WSDM intervention gate.

## P0 evidence retained

The 2026-08-15 experiment snapshot records 65 completed unique learning-rate × group-size × reweight conditions, plus eight usable new-seed runs. The comparable P0 subset contains four historical seed-20260716 runs and eight new runs for seeds 20260815 and 20260816. All use 500 optimizer steps, Qwen3-Embedding-0.6B, the same query-disjoint Dev1500 artifact, cross-device negatives and no locked-test metrics.

Across the 56-condition complete grid through learning rate 7e-6, reweight-on won 15 of 28 matched MRR cells and reweight-off won 13; the mean on-minus-off MRR difference was -0.000394. For R@1, on won 12 cells and off won 16, with a mean difference of -0.000476. The direction changes with learning rate, group size, seed and metric.

The two added seeds do not resolve this interaction. At learning rate 6e-6 and group size 5, the off-minus-on MRR difference was -0.000181 with query-bootstrap 95% CI [-0.006612, 0.006474] for seed 20260815, and +0.003730 with CI [-0.003332, 0.010881] for seed 20260816. The intervals cross zero. These are offline Dev1500 comparisons, not new leaderboard gains and not mechanism evidence.

## What can and cannot be aligned

| Link | Status | Interpretation |
|---|---|---|
| LRAT pair pool → training sample | Verified | A row contains a stored negative pool; group size 6 draws five negatives and cross-device/in-batch negatives add further contrasts. |
| P0 configuration → Dev1500 outcome | Verified | Same split and controlled training settings are available; seed and query-level uncertainty are recorded. |
| BrowseComp candidate → human qrel | Verified after Stage 0 | Exact question-text mapping and official document IDs provide the diagnostic label. |
| BrowseComp qrel → LRAT pair label | Prohibited | The datasets contain different questions; no row-level label transfer is valid. |
| Behavior rule → LRAT intervention | Testable | Later visit or repeated exposure can be computed separately in each trajectory collection without copying qrel labels. |
| Intervention → mechanism | Not yet established | A controlled random-removal arm and multiple seeds are required; even then the claim is about the tested observable rule. |

## Stage 3 defaults if the diagnostic gate passes

The preferred first method is later-visit filtering because it has an exact trajectory definition and does not require a new judge. For a mapped LRAT pair, remove a stored negative if that document is visited after the pair's positive browse event, while retaining at least five negatives. The exposure heuristic prioritizes documents surfaced at least twice before the event and deterministically fills any shortfall, so it removes exactly the same number per row as the later-visit arm. A deterministic random arm is matched in the same way.

The first pilot should use the P0-centered training setting rather than claiming it is globally optimal: Qwen3-Embedding-0.6B, group size 5 or 6 as explicitly recorded, 500 optimizer steps, the fixed Dev1500 split and a single non-selection seed. Expansion to multiple seeds is permitted only after the code/data audit passes and the pilot is directionally coherent. E5 is a robustness extension, not a prerequisite for interpreting a failed Qwen pilot.

## Evidence sources

- Fresh local split audit: `analysis/negative_reliability/stage2_split_audit.json`.
- Current split manifest: `data/processed/early_stop_v1/manifest.json`.
- P0 snapshot: `/home/seraphic/xir/report/CCIR_report/ai_analysis/result.md` (read-only source; not copied or modified).
- Official LRAT construction: `LRAT/src/data_builder.py`.
- Controlled arm builder: `solution/src/build_lrat_reliability_arms.py`.

The remote canonical P0 JSON artifacts are listed in the snapshot under `/root/data/LRAT/ccir/outputs/`. At the time of this local audit the documented SSH endpoint refused the TCP connection before authentication, so the numerical conclusions above are re-verified against the local durable snapshot and split files, not falsely presented as a fresh remote readback.
