import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_early_stop_split", ROOT / "src/prepare_early_stop_split.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_row(query: str, suffix: str, *, shared_positive: bool = False) -> dict:
    positive_id = "shared" if shared_positive else f"p-{suffix}"
    return {
        "query": query,
        "pos_id": [positive_id],
        "pos": [f"positive {suffix}"],
        "neg_id": ["shared", f"n-{suffix}"],
        "neg": ["possible conflict", f"negative {suffix}"],
    }


def write_rows(path: Path) -> None:
    rows = []
    for index in range(12):
        rows.append(make_row(f" Query   {index} ", f"{index}-a", shared_positive=index == 0))
        if index in {0, 3, 7}:
            rows.append(make_row(f"query {index}", f"{index}-b"))
    rows.append(make_row("", "empty"))
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_split_is_exact_disjoint_and_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    write_rows(source)
    first, second = tmp_path / "first", tmp_path / "second"
    report = MODULE.prepare(source, first, dev_queries=4, test_queries=3, salt="fixed", write_train=True)
    MODULE.prepare(source, second, dev_queries=4, test_queries=3, salt="fixed", write_train=True)

    assert report["dev"]["query_groups"] == 4
    assert report["test"]["query_groups"] == 3
    assert report["train"]["normalized_query_overlap_with_dev_or_test"] == 0
    assert report["source"]["empty_query_rows_retained_in_train"] == 1
    assert (first / "dev.jsonl").read_bytes() == (second / "dev.jsonl").read_bytes()
    assert (first / "test.jsonl").read_bytes() == (second / "test.jsonl").read_bytes()
    assert (first / "train.jsonl").read_bytes() == (second / "train.jsonl").read_bytes()

    train_keys = {MODULE.norm_query(row["query"]) for row in read_rows(first / "train.jsonl")}
    dev_keys = {MODULE.norm_query(row["query"]) for row in read_rows(first / "dev.jsonl")}
    test_keys = {MODULE.norm_query(row["query"]) for row in read_rows(first / "test.jsonl")}
    assert not (train_keys & dev_keys or train_keys & test_keys or dev_keys & test_keys)
    assert "" in train_keys


def test_positive_identifier_is_shielded_from_aggregated_negatives(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        make_row("same query", "a", shared_positive=True),
        make_row(" SAME  QUERY ", "b"),
        make_row("other", "c"),
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "output"
    MODULE.prepare(source, output, dev_queries=1, test_queries=1, salt="find-both")
    eval_rows = read_rows(output / "dev.jsonl") + read_rows(output / "test.jsonl")
    aggregate = next(row for row in eval_rows if MODULE.norm_query(row["query"]) == "same query")
    assert "shared" in aggregate["pos_id"]
    assert "shared" not in aggregate["neg_id"]
    assert len(aggregate["pos_id"]) == 2


def test_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    write_rows(source)
    output = tmp_path / "output"
    MODULE.prepare(source, output, dev_queries=2, test_queries=2)
    try:
        MODULE.prepare(source, output, dev_queries=2, test_queries=2)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected overwrite refusal")
