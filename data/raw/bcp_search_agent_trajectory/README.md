---
license: apache-2.0
viewer: false
task_categories:
  - question-answering
language:
  - en
tags:
  - search-agent
  - agent-trajectories
  - browsecomp-plus
  - deep-research
size_categories:
  - 1K<n<10K
---

# BrowseComp-Plus search-agent trajectories

Full ReAct trajectories for six search agents on
[BrowseComp-Plus](https://arxiv.org/abs/2508.06600), released with
*Diagnosing Search Behavior and Failure Modes in Long-Horizon Search Agents*.

Every reasoning trace, tool call, tool return and final answer, for all 830
questions × 6 agents. ~2.1 GB.

Code, harness and documentation: https://github.com/liuqi6777/search_agent

## Setup the trajectories were produced under

All six agents ran through the same harness against the same corpus and
retriever, so behavioural differences are not confounded with setup:

- **Corpus / retriever** — the official BrowseComp-Plus corpus (~100K docs),
  indexed with `Qwen/Qwen3-Embedding-8B`. The index is published separately at
  [`liuqi6777/Browsecomp-Plus-Indexes`](https://huggingface.co/datasets/liuqi6777/Browsecomp-Plus-Indexes).
- **Tools** — two. `search` returns the top *K*=5 documents, each with a docid,
  retrieval score, and a snippet truncated to 512 tokens. `visit` returns the
  full text of a document given its id.
- **Budget** — 128 turns and 150 minutes per question. Hitting either cap
  without an answer is recorded as `incomplete`.
- **Sampling** — each model's recommended parameters; harness defaults are
  temperature 1.0, top-p 0.95.

## Runs

| Directory | Agent | Acc (%) | Gold Rec. (%) | Incomplete (%) |
|---|---|---|---|---|
| `gpt-oss-120b-high` | gpt-oss-120b | 38.0 | 52.0 | 12.8 |
| `tongyi-dr` | Tongyi-DR | 52.2 | 64.2 | 38.0 |
| `qwen35-35b-a3b` | Qwen3.5-35B-A3B | 55.2 | 63.0 | 1.3 |
| `ds4pro` | Deepseek V4 Pro | 68.6 | 75.4 | 0.7 |
| `kimi26` | Kimi K2.6 | 69.2 | 78.2 | 16.5 |
| `glm51` | GLM 5.1 | 74.1 | 78.7 | 9.8 |

`gpt-oss-120b-high` is the run at **high** reasoning effort, which is the one
the paper reports. A default-effort run exists and is not part of this release.

Accuracy comes from `gpt-4o-2024-08-06` following the official BrowseComp judge
protocol; every retrieval-side metric in the paper is computed from qrels and is
independent of the judge.

## Format

One `predictions.jsonl` per run, one JSON object per line:

```json
{
  "question": "...",
  "answer": "...",
  "prediction": "...",
  "termination": "answer",
  "messages": [ ... the full history, including every tool observation ... ],
  "metadata": {
    "model_name": "...",
    "model_calls": [
      {"turn": 1, "response": {...}, "tool_calls": [{"name": "search", "arguments": {"query": "..."}}]}
    ]
  }
}
```

`termination` is `answer` or `incomplete`.

`messages` holds every search result and every full document an agent read, and
is most of the size. If you only need behaviour — what was called, in what
order, with what arguments — `metadata.model_calls` carries the same actions
without the observations:

```python
import json

with open("glm51/predictions.jsonl") as handle:
    for line in handle:
        record = json.loads(line)
        actions = [
            call["name"]
            for turn in record["metadata"]["model_calls"]
            for call in turn["tool_calls"]
        ]
```

Recovering *retrieved* docids (rather than visited ones) means parsing the
`search` observations in `messages`. Local search results always render a hit
as:

```
N. Document ID: <docid>
Score: <float>
Snippet: <text>
```

`manifest.json` records per-run row counts, termination breakdowns, byte sizes
and SHA-256 digests.

## Preprocessing

Nothing inside `messages` was rewritten — every observation is exactly as the
agent saw it. Three things were done, by
[`scripts/prepare_trajectories.py`](https://github.com/liuqi6777/search_agent/blob/main/scripts/prepare_trajectories.py):

- **Deduplicated.** A run resumed after a rate-limit storm can hold several rows
  for one question; one survives, preferring the attempt that answered over the
  one that errored. Only `kimi26` was affected (967 rows → 830).
- **Scrubbed.** Error records carried the serving stack's raw failure message,
  including internal hostnames and deployment identifiers. The exception *type*
  is kept, the message dropped.
- **Sorted** into dataset order, so the files are byte-reproducible.

Every run covers all 830 questions with no duplicates.

## License

Apache 2.0. The underlying questions, corpus and qrels are from BrowseComp-Plus
and carry their own terms.
