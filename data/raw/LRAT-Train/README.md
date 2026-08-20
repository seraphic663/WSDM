---
pretty_name: LRAT Training Dataset
task_categories:
- text-retrieval
- sentence-similarity
language:
- en
tags:
- retrieval
- dense-retrieval
- agentic-search
- trajectory
- deep-research
- synthetic-supervision
---

# LRAT Training Dataset

This dataset contains trajectory-derived retrieval supervision used in **LRAT** (Learning to Retrieve from Agent Trajectories).

The dataset is designed for training retrievers for **agentic search**. Instead of relying on human click logs, it is built from deep research agent trajectories that record intermediate search queries, browsing actions, and post-browse reasoning traces.

## Dataset Summary

LRAT formalizes a simple idea: when search is increasingly carried out by agents, retrieval training should be based on **agent interaction data**. This dataset operationalizes that idea by converting search agent trajectories into retrieval training pairs.

In the paper instantiation, the dataset is constructed from:

- 10K seed queries from InfoSeekQA
- Tongyi-DeepResearch-30B-A3B as the trajectory collection agent
- Wiki-25-Dump as the retrieval corpus
- four retrievers used during collection: BM25, Qwen3-Embedding-0.6B, Qwen3-Embedding-4B, and Qwen3-Embedding-8B

The resulting scale reported in the paper is:

- 26,482 valid agent trajectories
- 96,504 training pairs

## How the Dataset Is Built

Each training example is derived from a local search trajectory:

1. The agent issues an intermediate query.
2. The retriever returns a top-k candidate set.
3. Browsed documents become positive candidates.
4. Unbrowsed documents from the retrieved set provide negative signals.
5. An LLM judge checks the post-browse reasoning trace to decide whether the browsed document truly helped the agent.
6. The length of the post-browse reasoning is converted into a soft relevance weight.

This process produces supervision that is better aligned with how search agents use retrieval during multi-step reasoning.

## Data Fields

The LRAT training pipeline writes JSONL records with fields such as:

- `query`: the intermediate query issued by the agent
- `pos`: positive document text list
- `neg`: negative document text list
- `pos_id`: positive document id list
- `neg_id`: negative document id list
- `reasoning_len`: token length of the post-browse reasoning trace
- `satisfied`: whether the reasoning-aware judge treats the browsed document as relevant
- `reweight_rate`: a normalized relevance intensity weight derived from reasoning length

In the current pipeline, `pos` typically contains one retained positive document, while `neg` may contain multiple negatives.

## Example Record

```json
{
  "query": "Who coined the term ...",
  "pos": ["positive document text ..."],
  "neg": ["negative document text A ...", "negative document text B ..."],
  "pos_id": ["12345"],
  "neg_id": ["67890", "54321"],
  "reasoning_len": 504,
  "satisfied": true,
  "reweight_rate": 1.18
}
```

## Why These Signals Matter

The paper's trajectory analysis shows:

- browsing is a necessary signal for successful task completion
- unbrowsed documents are much more reliable negatives for agents than for human click logs
- post-browse reasoning reveals whether a document actually helped
- longer post-browse reasoning is correlated with stronger document utility

The reasoning-aware filter in the paper:

- retains 97.2% of ground-truth evidence documents
- retains 74.8% of browsed non-evidence documents

This means the dataset is neither a simple click-log imitation nor a standard human relevance dataset. It is specifically designed to capture **agent-aligned document utility**.

## Intended Uses

This dataset is intended for:

- training dense retrievers for agentic search
- studying trajectory-derived retrieval supervision
- research on agent-aligned retrieval objectives
- reproducing LRAT-style retriever training

It is not intended as:

- a benchmark for general semantic similarity
- a manually curated relevance dataset
- a universal substitute for all retrieval corpora


## License and Usage Notes

Before public release, please verify compatibility with:

- the source datasets
- the underlying retrieval corpus
- the models used for trajectory collection and judging
- your final redistribution policy

This card can be updated later with the final license statement for the released dataset.

## Citation

If you use this dataset, please cite the LRAT paper.

```bibtex
@inproceedings{zhou2026lrat,
  title={Learning to Retrieve from Agent Trajectories},
  author={Zhou, Yuqi and Dai, Sunhao and Qu, Changle and Pang, Liang and Xu, Jun and Wen, Ji-Rong},
  booktitle={Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2026}
}
```

## Links

- Paper: `https://arxiv.org/abs/2604.04949`
- Project page: `https://yuqi-zhou.github.io/LRAT-homepage/`
- Code: `https://github.com/Yuqi-Zhou/LRAT`
- Dataset: `https://huggingface.co/datasets/Yuqi-Zhou/LRAT-Train`
- Checkpoints:
  - `https://huggingface.co/Yuqi-Zhou/LRAT-Qwen3-Embedding-0.6B`
  - `https://huggingface.co/Yuqi-Zhou/LRAT-multilingual-e5-large`
