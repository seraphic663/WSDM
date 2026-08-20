import importlib.util
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare_mod = load_module("prepare_cleaned_ab", "src/prepare_cleaned_ab.py")
compare_mod = load_module("compare_paired_evals", "src/compare_paired_evals.py")


def row(query: str, negatives: int = 6) -> dict:
    return {
        "query": query,
        "pos": [f"positive {query}"],
        "pos_id": [f"p-{query}"],
        "neg": [f"negative {query} {index}" for index in range(negatives)],
        "neg_id": [f"n-{query}-{index}" for index in range(negatives)],
        "reweight_rate": 1.0,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8")


def test_prepare_is_paired_and_query_disjoint(tmp_path: Path) -> None:
    raw_rows = [row(f"query {index}") for index in range(40)] + [row("")]
    clean_rows = [dict(value) for value in raw_rows]
    clean_rows[3] = {**clean_rows[3], "neg": clean_rows[3]["neg"][:-2], "neg_id": clean_rows[3]["neg_id"][:-2]}
    raw, cleaned = tmp_path / "raw.jsonl", tmp_path / "clean.jsonl"
    write_jsonl(raw, raw_rows)
    write_jsonl(cleaned, clean_rows)
    output = tmp_path / "paired"
    report = prepare_mod.prepare(raw, cleaned, output, split_threshold=32768)
    assert report["train_rows_per_arm"] > 0
    assert report["heldout_query_groups"] > 0
    assert report["normalized_query_overlap"] == 0
    assert report["short_negative_query_groups_excluded_from_both_arms"] == 1
    assert len((output / "raw_train.jsonl").read_text().splitlines()) == report["train_rows_per_arm"]
    assert len((output / "cleaned_train.jsonl").read_text().splitlines()) == report["train_rows_per_arm"]
    assert '"query": ""' in (output / "raw_train.jsonl").read_text()


def eval_result(path: Path, ranks: list[int], input_name: str = "validation.jsonl") -> None:
    value = {
        "input": input_name,
        "rows": len(ranks),
        "details": [
            {
                "row_index": index,
                "best_positive_rank": rank,
                "query_sha256": hashlib.sha256(f"query-{index}".encode()).hexdigest(),
            }
            for index, rank in enumerate(ranks)
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def test_compare_gate_passes_for_consistent_improvement(tmp_path: Path) -> None:
    pairs = []
    for label in ("checkpoint-500", "final-1000"):
        raw, clean = tmp_path / f"{label}-raw.json", tmp_path / f"{label}-clean.json"
        eval_result(raw, [2] * 50)
        eval_result(clean, [1] * 50)
        pairs.append([label, str(raw), str(clean)])
    report = compare_mod.compare(pairs, tmp_path / "report.json", bootstrap_samples=1000, seed=3)
    assert report["gate"]["passed"] is True


def test_compare_gate_fails_for_mixed_direction(tmp_path: Path) -> None:
    pairs = []
    for label, clean_ranks in (("checkpoint-500", [1] * 50), ("final-1000", [3] * 50)):
        raw, clean = tmp_path / f"{label}-raw.json", tmp_path / f"{label}-clean.json"
        eval_result(raw, [2] * 50)
        eval_result(clean, clean_ranks)
        pairs.append([label, str(raw), str(clean)])
    report = compare_mod.compare(pairs, tmp_path / "report.json", bootstrap_samples=1000, seed=3)
    assert report["gate"]["passed"] is False
