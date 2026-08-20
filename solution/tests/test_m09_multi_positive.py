import hashlib
import json
import random
from pathlib import Path

import pytest
import torch

from solution.experiments.m09_multipositive.prepare import (
    build,
    merge_query_group,
)
from solution.experiments.m09_multipositive.train_entry import (
    choose_passage_indexes,
    multi_positive_loss,
)


def test_single_positive_loss_matches_cross_entropy():
    scores = torch.tensor([[2.0, 0.0, -1.0, 0.5, 0.1, -0.2], [0.0, -0.1, 0.2, 1.5, 0.3, -0.4]])
    expected = torch.nn.functional.cross_entropy(scores, torch.tensor([0, 3]))
    actual = multi_positive_loss(scores, [1, 1], group_size=3)
    assert torch.allclose(actual, expected)


def test_multi_positive_loss_uses_all_positive_columns():
    scores = torch.tensor([[2.0, 1.0, -1.0, 0.0, -0.5, -2.0]])
    actual = multi_positive_loss(scores, [2], group_size=6)
    expected = torch.logsumexp(scores, dim=1) - torch.logsumexp(scores[:, :2], dim=1)
    assert torch.allclose(actual, expected.mean())


def test_weighted_multi_positive_loss():
    scores = torch.tensor([[2.0, 0.0, -1.0, 0.5, 0.1, -0.2], [0.0, -0.1, 0.2, 1.5, 1.0, -0.4]])
    row_zero = torch.logsumexp(scores[0], dim=0) - scores[0, 0]
    row_one = torch.logsumexp(scores[1], dim=0) - torch.logsumexp(scores[1, 3:5], dim=0)
    actual = multi_positive_loss(scores, [1, 2], group_size=3, reweight_rates=[1.0, 3.0])
    expected = (row_zero + 3 * row_one) / 4
    assert torch.allclose(actual, expected)


def test_choose_passages_keeps_positive_prefix_and_fixed_group_size():
    positives, negatives = choose_passage_indexes(8, 2, 6, 3, random.Random(7))
    assert len(positives) == 3
    assert len(set(positives)) == 3
    assert len(negatives) == 3
    assert len(positives) + len(negatives) == 6


def test_merge_query_group_unions_positives_and_removes_conflicting_negative():
    rows = [
        {
            "query": "  Same   Query ",
            "pos": ["positive a"],
            "pos_id": ["a"],
            "neg": ["negative b", "positive c"],
            "neg_id": ["b", "c"],
            "reasoning_len": 2,
            "satisfied": True,
            "reweight_rate": 1.0,
        },
        {
            "query": "same query",
            "pos": ["positive c"],
            "pos_id": ["c"],
            "neg": ["negative d"],
            "neg_id": ["d"],
            "reasoning_len": 4,
            "satisfied": True,
            "reweight_rate": 2.0,
        },
    ]
    merged, stats = merge_query_group(rows)
    assert merged["pos_id"] == ["a", "c"]
    assert merged["neg_id"] == ["b", "d"]
    assert merged["reweight_rate"] == 1.5
    assert merged["_source_rows"] == 2
    assert stats["removed_positive_conflicts"] == 1


def test_builder_is_traceable_and_query_unique(tmp_path: Path):
    rows = [
        {"query": "Q", "pos": ["p1"], "pos_id": ["1"], "neg": ["n1"], "neg_id": ["n1"], "reasoning_len": 1, "satisfied": True, "reweight_rate": 1.0},
        {"query": " q ", "pos": ["p2"], "pos_id": ["2"], "neg": ["n2"], "neg_id": ["n2"], "reasoning_len": 2, "satisfied": True, "reweight_rate": 1.0},
        {"query": "other", "pos": ["p3"], "pos_id": ["3"], "neg": ["n3"], "neg_id": ["n3"], "reasoning_len": 3, "satisfied": True, "reweight_rate": 1.0},
    ]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "train.jsonl"
    manifest = tmp_path / "manifest.json"
    result = build(source, output, manifest, expected_sha)
    built = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["source_rows"] == 3
    assert result["output_rows"] == 2
    assert result["multi_positive_rows"] == 1
    assert len({" ".join(row["query"].lower().split()) for row in built}) == 2
    assert manifest.is_file()


def test_builder_preserves_empty_queries_as_separate_source_rows(tmp_path: Path):
    rows = [
        {"query": "", "pos": ["p1"], "pos_id": ["1"], "neg": ["n1"], "neg_id": ["n1"]},
        {"query": "  ", "pos": ["p2"], "pos_id": ["2"], "neg": ["n2"], "neg_id": ["n2"]},
    ]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "train.jsonl"
    manifest = tmp_path / "manifest.json"
    result = build(source, output, manifest, hashlib.sha256(source.read_bytes()).hexdigest())
    assert result["output_rows"] == 2
    assert result["nonempty_unique_normalized_queries"] == 0
    assert result["empty_query_rows_preserved_individually"] == 2


@pytest.mark.parametrize("counts", ([0], [3], [1, 1]))
def test_invalid_positive_counts_are_rejected(counts):
    scores = torch.zeros((1, 3))
    with pytest.raises(ValueError):
        multi_positive_loss(scores, counts, group_size=3)
