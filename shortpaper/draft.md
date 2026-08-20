# Paper Idea and Working Draft

> Status: Stage 0/1 empirical draft complete, 2026-08-20; the pre-registered decision is PIVOT.  
> Evidence rule: qrels-grounded heterogeneity is supported; causal mechanisms and intervention gains remain unsupported because the coverage gate stopped training.  
> Current output: the author-visible four-page ACM `sigconf` paper is `shortpaper/latex/demo.pdf`; this document retains fuller design and reviewer-defense material.

## Working Title

**When Is an Unvisited Result a Negative? Auditing Behavioral Supervision for Agentic Retrieval**

Alternative method-oriented title, to be used only if the intervention becomes a genuine contribution:

**Calibrating Behavioral Negative Supervision for Agentic Retrieval**

## Empirical Abstract

Agent trajectories provide scalable retriever supervision, but a surfaced result that is not visited is not necessarily irrelevant. We reconstruct 638,215 ranked candidate occurrences from 4,980 trajectories produced by six search Agents on 830 BrowseComp-Plus questions and align them with evidence and gold qrels. Among candidates unvisited before the next search, 13.77% are evidence-qrel positives; the query-balanced mean is 19.69%. Reliability is strongly conditional: later-visited candidates are evidence-relevant 66.69% of the time versus 12.14% otherwise, and the contrast grows to 67.99 percentage points on first exposures. Repetition, rank, search stage, and accumulated evidence show additional heterogeneity under question-cluster bootstrap uncertainty. A pre-registered method gate nevertheless returns PIVOT because successful-visit mapping reaches 98.88%, below its 99% threshold. We therefore contribute a qrels-grounded diagnosis and a reproducible stopping decision rather than an unsupported training-gain claim.

## Current Claim Boundary

- Supported: unvisited candidate occurrences contain nontrivial qrel positives and their rates vary strongly across later visit, repetition, rank, search phase, evidence state, Agent, and cross-Agent exposure.
- Supported: the main directions persist under first-exposure and gold-qrel sensitivity checks, with 10,000 query-cluster bootstrap samples.
- Supported: missing visit mappings are Agent-dependent and concentrated in Tongyi-DR.
- Unsupported: later visit causes relevance; these cases are universally false negatives; filtering improves retriever training; BrowseComp qrels can label LRAT/InfoSeek rows.
- Decision: retain a diagnostic Short; do not run or report the prepared intervention arms unless a future data release repairs the coverage gate under a prospectively declared rerun.

## 1. Introduction

Retriever training has traditionally relied on human relevance judgments or interaction logs. Agentic search introduces a different source of behavioral data: a search Agent repeatedly issues queries, receives ranked results, visits a subset of them, incorporates some evidence into its reasoning, and eventually produces an answer. These trajectories make it possible to construct supervision at scale, but they also entangle relevance with exposure, search state, Agent policy, and stopping decisions.

LRAT demonstrates that Agent trajectories can support effective retriever training. In its data construction, visited documents that pass a relevance judgment become positives, while unvisited or behaviorally rejected candidates contribute negative supervision. The overall effectiveness of this construction does not imply that every candidate-level negative label has the same reliability.

The central question is therefore not whether occasional false negatives exist. That phenomenon is already familiar in dense retrieval. We instead ask whether the reliability of retrieved-but-unvisited feedback changes systematically with the search process, and whether measuring that structure provides information beyond simply using fewer negatives.

The intended contribution chain is:

1. define a relevance-grounded diagnostic view of Agent search behavior;
2. measure the conditional reliability of trajectory-derived negative labels;
3. test whether the measured structure explains retriever-training behavior;
4. if warranted, validate a simple reliability-aware intervention.

The fourth contribution is conditional. The paper remains viable as an empirical characterization only if the diagnosis is sufficiently stable and general; a heuristic intervention without a strong diagnosis is not treated as a contribution.

## 2. Problem Formulation

For a query or sub-query (q), document (d), and trajectory context (x), define:

- (S(q,d)=1) if (d) is surfaced in a retrieved result list;
- (V(q,d)=1) if the Agent visits or browses (d);
- (U(q,d)=1) if information from (d) is utilized in later reasoning or the answer;
- (R(q,d)=1) if a human document-level relevance judgment marks (d) as relevant.

The primary diagnostic population is:

$$
\{(q,d,x): S(q,d)=1, V(q,d)=0\}.
$$

For this population, negative-label reliability is:

$$
\operatorname{RelNeg}(x)
=
P(R=0\mid S=1,V=0,x).
$$

The first scientific question is whether:

$$
\operatorname{RelNeg}(x_1)
\neq
\operatorname{RelNeg}(x_2)
$$

for interpretable contexts such as rank, search stage, currently retrieved evidence, repeated retrieval, query difficulty, or Agent policy.

This formulation deliberately separates human relevance from behavior. A document can be relevant but unvisited because it was ranked low, redundant with evidence already obtained, overlooked by the Agent, or surfaced after the Agent had effectively finished the task. Conversely, a high-ranked unvisited result may supply stronger negative evidence if the Agent had sufficient exposure and continued searching.

## 3. Research Questions and Hypotheses

### RQ1: How often are retrieved-but-unvisited candidates human-relevant?

The existence of such cases is necessary but not sufficient for the paper. A non-zero average false-negative rate alone would be a limited observation.

### RQ2: Is negative-label reliability conditionally heterogeneous?

The main hypothesis is that relevance among retrieved-but-unvisited candidates varies systematically with observable trajectory context. Candidate variables include rank, search stage, evidence saturation, repeated retrieval, later visit, Agent identity, and query difficulty.

### RQ3: Does label reliability help explain training behavior?

The planned analysis tests whether negative-set reliability, rather than negative count alone, is associated with the previously observed negative-size anomaly. This is an empirical question, not an assumed causal chain.

### RQ4: Can a simple reliability-aware intervention outperform trivial controls?

Any proposed intervention must be compared with randomly using fewer negatives. If it does not outperform this control consistently, the paper should not claim that reliability modeling improves training.

## 4. Data and Measurement

### 4.1 Training-supervision data

LRAT training pairs and trajectories provide the complete path from Agent behavior to retriever training samples. They support candidate reconstruction, reproduction of vanilla training, negative-set analysis, and controlled evaluation of any final intervention.

The available competition experiments provide preliminary anomalies and robust baselines. They do not provide relevance labels for every trajectory-derived negative and therefore cannot by themselves establish the proposed mechanism.

### 4.2 Ground-truth diagnostic data

BrowseComp-Plus document-level human relevance judgments and associated multi-Agent trajectories can separate what was surfaced, what was visited, what was utilized, and what was relevant. Their primary role is diagnosis rather than direct retriever training.

Before analysis, the project must verify identifier compatibility, qrels coverage, result-list recovery, rank semantics, visit mapping, trajectory ordering, and the availability of evidence-state variables. Data from non-public collaborator work may be used only after explicit permission.

### 4.3 Measurement hierarchy

Human qrels are the primary relevance judgment. Later visits, repeated retrieval, and cross-Agent disagreement are behavioral consistency signals. LLM judgments and answer-evidence matching may provide auxiliary checks but are not treated as ground truth.

## 5. Diagnostic Design

### 5.1 Stage 0: feasibility and coverage

Construct a candidate-level table with query ID, Agent, trajectory ID, search step, document ID, rank, surfaced state, visited state, later-visited state, utilized state where available, human relevance, and current evidence state. Report join coverage, missingness, duplicated document mappings, and the number of usable queries before reporting any reliability curve.

### 5.2 Stage 1: conditional reliability

Estimate relevance among retrieved-but-unvisited candidates by rank bucket, search stage, evidence saturation, repeated retrieval, Agent, and query difficulty. Report denominators and uncertainty intervals. Avoid interpreting small strata or a single Agent as general evidence.

### 5.3 Stage 2: relation to training anomalies

Reanalyse negative-set configurations using both retrieval metrics and estimated negative reliability. The goal is to distinguish a mere hyperparameter optimum from evidence that different configurations expose the model to negative sets of different quality.

## 6. Conditional Intervention

The intervention is intentionally unspecified until the diagnostic audit is complete. Candidate implementations include filtering candidates in low-reliability strata, sampling in proportion to estimated reliability, or weighting negative terms in the contrastive loss.

For a candidate reliability weight (c_i\in[0,1]), a possible weighted objective is:

$$
\mathcal{L}_{\mathrm{cal}}
=
-\log
\frac{\exp s(q,d^+)}
{\exp s(q,d^+)+\sum_i c_i\exp s(q,d_i)}.
$$

This equation defines an interface, not the selected method. A valid (c_i) must have an interpretable behavioral meaning, correlate with human-qrels reliability, and improve over both vanilla LRAT and random downsampling.

## 7. Evaluation Plan

The minimum comparison set is:

1. pretrained retriever without trajectory fine-tuning;
2. vanilla LRAT-style training;
3. LRAT with randomly fewer negatives;
4. a simple rank- or exposure-based heuristic;
5. the selected reliability-aware intervention;
6. where feasible, a representative false-negative-aware dense-retrieval baseline.

Primary retrieval metrics may include MRR, Recall@1, Recall@k, and evidence recall. Diagnostic results should report human-relevant rate among unvisited candidates, reliability by condition, denominators, uncertainty intervals, and sensitivity to the operational definition of each state.

Robustness priorities are multiple Agents or diagnostic settings, Qwen3-Embedding-0.6B, a different retriever family such as E5, multiple negative-set configurations, and two or more seeds where training cost permits. End-to-end Agent evaluation is useful but not mandatory for the first Short-paper closure.

## 8. Relationship to Prior Work

LRAT establishes the usefulness of trajectory-derived supervision. The present work asks whether the candidate-level reliability of one component of that supervision is homogeneous across trajectory contexts. The intended relationship is extension and boundary characterization, not rebuttal.

DND and other competition systems show that negative construction and behavior-aware hard-negative selection matter in practice. Their existence raises the novelty bar: the paper must lead with human-qrels measurement and conditional structure, not with another filtering heuristic.

Prior false-negative and noise-robust retrieval methods motivate suitable baselines. They also prevent the paper from claiming that false negatives or confidence weighting are new in themselves.

The diagnostic framework in *Diagnosing Search Behavior and Failure Modes in Long-Horizon Search Agents* motivates the explicit separation of surfaced, visited, utilized, and relevant states. That paper studies search effort and failure attribution; the proposed paper studies the reliability of those states when converted into retriever-training labels.

## 9. Expected Paper Structure

The author-visible arXiv-style version can follow the rhetorical rhythm of the diagnostic reference paper:

1. Introduction: problem, capability gap, and contribution boundary;
2. Related Work;
3. Problem Formulation and Diagnostic Framework;
4. Reliability Diagnosis;
5. Training Consequences and Conditional Intervention;
6. Discussion and Limitations;
7. Conclusion;
8. Ethical Considerations;
9. References.

The four-page WSDM version should compress this into approximately one page of motivation and setup, one page of diagnosis and method, and two pages of experiments, limitations, and conclusion. Related work should be integrated into the introduction and method discussion rather than occupying a large standalone section.

## 10. Claim Ledger

### Currently supportable

- Agent trajectories are an effective source of retrieval supervision according to LRAT.
- Existing local experiments do not establish a stable independent benefit from reasoning-length reweighting.
- Negative construction is an important empirical design choice and is already explored by multiple systems.
- The proposed reliability question is testable if candidate exposure, Agent behavior, and human relevance can be aligned.

### Not yet supportable

- Retrieved-but-unvisited candidates contain a substantial false-negative rate.
- Reliability varies with rank, stage, evidence state, or Agent.
- More negatives reduce average reliability.
- Reliability explains the observed performance anomaly.
- Filtering, sampling, or weighting improves retriever performance.
- Any observed pattern generalizes across Agents, datasets, or retrievers.

## 11. Decision Rule

Proceed with the full paper story only if relevance-grounded diagnosis yields stable conditional structure. If diagnosis is strong but training gains are limited, retain an empirical-characterization framing. If diagnosis is weak, inconsistent, or not recoverable from the available data, stop rather than adding post-hoc heuristics.

## 12. Falsifiable Hypotheses

The research questions above are operationalized as five hypotheses. None is treated as established before the corresponding audit.

### H1: Some surfaced-but-unvisited candidates are human-relevant

$$
P(R=1\mid S=1,V=0)>0.
$$

H1 establishes that behavioral negatives are not perfectly clean. It is necessary but insufficient: occasional counterexamples are expected in dense retrieval and do not by themselves constitute a contribution.

### H2: Reliability is conditionally heterogeneous

For interpretable contexts (x_1,x_2),

$$
P(R=0\mid S=1,V=0,x_1)
\neq
P(R=0\mid S=1,V=0,x_2).
$$

Candidate conditions include rank, search stage, evidence saturation, repeated retrieval, later visit, Agent policy, and query difficulty. H2 is the main step from generic label noise to an Agent-trajectory supervision problem.

### H3: Larger sampled negative sets may contain more low-reliability candidates

For a sampled negative set (N), the testable claim is whether increasing (|N|) changes the distribution of estimated reliability. Low estimated reliability must not be equated with a confirmed false negative unless human relevance supports that interpretation.

### H4: Negative-set reliability is associated with the observed training anomaly

The target relationship is:

$$
\text{negative-set reliability}
\leftrightarrow
\text{retrieval performance},
$$

not merely (|N|\leftrightarrow\text{performance}). Existing group-size results motivate H4 but do not prove it. Any causal language requires a controlled intervention that changes reliability while matching other training properties.

### H5: A targeted intervention outperforms random downsampling

At matched expected negative count or mass, the selected filtering, sampling, or weighting rule should outperform randomly using fewer negatives. Failure on this comparison means that the intervention has not demonstrated value beyond reducing supervision volume.

## 13. Reliability Signals and Intervention Candidates

### 13.1 Later visit and repeated retrieval

If a document is surfaced but unvisited at an early step, then retrieved again or visited later, the early behavioral label is unstable. This is a trajectory-internal signal and requires no external judge. It is not itself a human relevance label; it tests behavioral consistency.

### 13.2 Exposure-aware signal

Rank, snippet visibility, and interface constraints affect the opportunity to evaluate a result. A high-ranked, clearly exposed candidate that remains unvisited may provide stronger negative evidence than a low-ranked candidate, but the direction cannot be assumed because Agent policies may also exhibit position or context bias. Rank-based rules require a qrels-grounded curve before use.

### 13.3 Cross-trajectory and cross-Agent disagreement

When the same or a closely aligned query-document pair is surfaced under multiple trajectories or Agents and receives different visit decisions, the behavior is policy-dependent. This can support H2, but entity/query matching error and different trajectory states must be reported.

### 13.4 Independent relevance or utility judge

An LLM or stronger reranker can increase coverage for candidate scoring, but it risks replacing one noisy supervision source with another. Judge output is an auxiliary validation signal unless calibrated against human judgments, and its model, prompt, cost, variance, and possible leakage must be documented.

### 13.5 Answer-evidence matching

Candidate text can be checked against final answer evidence. This provides a downstream-utility proxy, but answer presence does not imply relevance at a particular search step, lexical overlap can create false positives, and multiple documents may jointly support an answer. It is therefore secondary evidence.

### 13.6 Intervention interface

The candidate weight (c_i\in[0,1]) denotes confidence that document (d_i) is a genuine negative under the current measurement model. It is not automatically a calibrated probability and does not create a new ground-truth label. Possible implementations are:

1. **Filtering:** exclude only a preregistered low-reliability stratum.
2. **Sampling:** sample negatives in proportion to estimated reliability while controlling count and hardness.
3. **Weighting:** scale negative contributions in the contrastive denominator.
4. **Hybrid:** use a small number of high-confidence behavioral negatives and hardness-matched candidates.

The simplest implementation that passes the diagnostic and control gates is preferred. A learned neural confidence model is outside the initial Short scope.

## 14. Reviewer Attacks and Required Defenses

### Attack 1: False negatives are already well studied

The novelty is not the existence of false negatives. It is the selection mechanism produced by a multi-step Agent policy and the measurement of conditional label reliability. Classic ambiguous-negative, noise-robust retrieval, and confidence-regularization work must be acknowledged and used as baselines where feasible.

### Attack 2: This is only click or position bias under a new name

Agent browsing is related to implicit feedback, but depends on reasoning state, acquired evidence, search budget, stopping, redundancy, tool policy, and Agent-specific behavior. The paper should frame this as a new behavioral-feedback regime, not claim that selection bias is new.

### Attack 3: The proxy is not relevance

The primary analysis must use document-level human qrels when available. Later visit, repeated retrieval, cross-Agent disagreement, judges, and answer matching are reported separately. Agreement across signals strengthens interpretation but does not convert proxies into ground truth.

### Attack 4: The method merely uses fewer negatives

Compare against random downsampling at matched expected negative count or mass. Also match negative similarity/hardness where possible; otherwise a gain may be ordinary hard-negative mining.

### Attack 5: One dataset or competition setting is insufficient

Short scope can be narrow, but the design should seek multiple Agents or diagnostic settings, query subsets, negative sizes, and at least a second retriever family for the final intervention. If unavailable, the limitation must be explicit and the claim narrowed to the observed setting.

### Attack 6: Why not clean labels with a stronger reranker or LLM judge?

The preferred contribution uses trajectory-internal and human-qrels-grounded evidence without requiring expensive external annotation at training time. If a judge becomes part of the method, its cost, bias, and calibration must be compared with simpler alternatives.

### Attack 7: The novelty is too close to LRAT

The first page must state the relation precisely: LRAT shows trajectory supervision can work; this paper measures whether one behavioral negative signal is equally reliable across trajectory contexts and tests a targeted correction only if the measurement supports it.

## 15. Detailed Experimental Matrix

### 15.1 Stage 0: data quality and recoverability

Required outputs are a candidate-level table, schema, row/column profile, candidate key definition, duplicate rates, join coverage, missingness by Agent/rank/stage, repeated-document statistics, temporal ordering checks, and documented exclusions. Single-Search candidates, saved pair-level pools, query-level aggregates, and training-time samples remain separate tables or explicitly typed grains.

### 15.2 Stage 1: diagnosis

Primary estimates include overall human-relevant rate among (S=1,V=0), rank buckets, search stage, evidence saturation, repeated retrieval, later visit, Agent, and query difficulty. Report query count, candidate count, rate, query-level bootstrap interval, and sensitivity to first-exposure versus all-exposure weighting.

### 15.3 Stage 2: relation to existing training results

For each comparable negative-size/training configuration, estimate the reliability distribution of the candidates actually eligible for sampling. Align this with seed-level MRR/Recall results without treating Dev1500 as new leaderboard evidence. Test association first; causal claims require a later controlled sample construction.

### 15.4 Stage 3: intervention pilot

Run a deterministic smoke test that verifies row counts, selected negative count, no positive/negative collisions, no future-trajectory leakage, reproducible selection, and compatibility with the existing trainer. Then run one seed for vanilla, random-fewer, simple heuristic, and the selected reliability intervention.

### 15.5 Stage 4: robustness

After the pilot gate, prioritize Qwen3-Embedding-0.6B, E5 as a different model family, multiple seeds, multiple negative sizes, and query-level paired intervals. An end-to-end Agent metric is additive evidence, not a prerequisite for the initial diagnosis.

### 15.6 Metrics and uncertainty

Retrieval metrics may include MRR, Recall@1, Recall@5, Recall@10, NDCG, gold/evidence recall where defined, and seed-level summaries. Diagnostic metrics include:

$$
\operatorname{RelevantAmongUnvisited}
=P(R=1\mid S=1,V=0),
$$

$$
\operatorname{RelNeg}(x)
=P(R=0\mid S=1,V=0,x).
$$

If multiple proxies are used, do not collapse them into a single ground-truth FNR. Report each definition and its coverage.

## 16. Four-Page WSDM Layout

### Page 1: problem and motivation

Use approximately 0.8–1.0 page for the Agentic Retrieval setting, LRAT foundation, the behavioral-label ambiguity, the (S,V,U,R) definition, the research question, and two or three conditional contributions. A small trajectory diagram should make the distinction between surfaced, visited, utilized, and relevant immediately visible.

### Page 2 upper half: reliability diagnosis

Define the qrels-grounded estimand, coverage checks, and the main rank/stage/evidence-state figure. Denominators and uncertainty must be readable in the paper rather than deferred to supplementary material.

### Page 2 lower half: intervention

Present only the selected lowest-complexity intervention and the necessary random-fewer/hardness-matched control. Do not introduce several alternative neural models.

### Pages 3–4: experiments and limitations

The main table compares base, vanilla trajectory fine-tuning, random fewer negatives, a simple heuristic, and the selected intervention across the strongest available settings. Add one compact diagnostic-to-training table, essential ablations, limitations, and a short conclusion. Related work is integrated into the introduction/method discussion; Ethical Considerations and references follow under the venue rule.

## 17. Title, Abstract, and Claim Variants

### Diagnostic-first title

**When Are Unvisited Results Reliable Negatives in Agentic Search?** This is preferred before a method gain is established.

### Method-oriented title

**Calibrating Behavioral Negative Supervision for Agentic Retrieval.** Use only if the intervention passes random-downsampling and robustness gates.

### Broader title

**Learning from Selective Behavioral Feedback in Agentic Retrieval.** This risks overclaiming beyond unvisited negatives and should be avoided unless the scope genuinely expands.

### Safe pre-result claim

Agent trajectories provide scalable retrieval supervision, but the candidate-level reliability of retrieved-but-unvisited feedback has not been established across trajectory contexts; we introduce a relevance-grounded audit and a conditional intervention protocol.

### Target post-result claim

Only if supported: retrieved-but-unvisited feedback is informative but heterogeneously reliable across specified contexts, and a targeted intervention improves over matched random downsampling.

### Claims to avoid

- LRAT assumes all unvisited documents are irrelevant.
- LRAT introduces severe label noise.
- More negatives necessarily cause more false negatives.
- A single Dev1500 delta proves a mechanism.
- Behavior-based calibration universally improves Agentic Retrieval.
