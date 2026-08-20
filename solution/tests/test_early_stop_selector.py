import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("select_early_stop_checkpoint", ROOT / "src/select_early_stop_checkpoint.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_eval(path: Path, mrr: float, *, name: str = "dev.jsonl", r5: float = 0.8, r10: float = 0.9) -> None:
    path.write_text(
        json.dumps(
            {
                "model": str(path.parent),
                "input": name,
                "rows": 1500,
                "metrics": {
                    "recall_at_1": mrr - 0.1,
                    "recall_at_5": r5,
                    "recall_at_10": r10,
                    "mrr": mrr,
                },
            }
        ),
        encoding="utf-8",
    )


def test_selects_best_and_stops_after_patience(tmp_path: Path) -> None:
    specs = []
    for step, mrr in ((1000, 0.60), (2000, 0.62), (3000, 0.6205), (4000, 0.619)):
        path = tmp_path / f"{step}.json"
        write_eval(path, mrr)
        specs.append((step, path))
    report = MODULE.select(specs, min_steps=2000, min_delta=0.001, patience=2)
    assert report["selected"]["step"] == 2000
    assert report["stop_triggered_at_step"] == 4000
    assert report["evaluations_consumed"] == 4


def test_rejects_locked_test(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    write_eval(path, 0.6, name="locked_test.jsonl")
    try:
        MODULE.select([(1000, path)])
    except ValueError as error:
        assert "locked test" in str(error)
    else:
        raise AssertionError("expected locked test rejection")


def test_guardrail_blocks_test_eligibility(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "baseline.json", tmp_path / "candidate.json"
    write_eval(baseline, 0.60, r5=0.90, r10=0.95)
    write_eval(candidate, 0.65, r5=0.89, r10=0.95)
    report = MODULE.select([(1000, candidate)], baseline_eval=baseline, guardrail_tolerance=0.005)
    assert report["eligible_for_locked_test"] is False


def test_without_m00_baseline_is_not_test_eligible(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    write_eval(candidate, 0.65)
    report = MODULE.select([(1000, candidate)])
    assert report["eligible_for_locked_test"] is False
