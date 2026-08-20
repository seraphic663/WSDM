import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("segment_control", ROOT / "src/segment_control.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_pause_only_at_nonfinal_target() -> None:
    assert MODULE.should_pause(999, 5000, 1000) is False
    assert MODULE.should_pause(1000, 5000, 1000) is True
    assert MODULE.should_pause(1001, 5000, 1000) is True
    assert MODULE.should_pause(5000, 5000, 1000) is False


def test_pause_marker_is_atomic_and_valid(tmp_path: Path) -> None:
    marker = tmp_path / "PAUSED_AT_STEP"
    MODULE.write_pause_marker(marker, global_step=1000, max_steps=5000, target_step=1000)
    value = json.loads(marker.read_text())
    assert value["global_step"] == value["target_step"] == 1000
    assert value["max_steps"] == 5000
    assert not list(tmp_path.glob(".PAUSED_AT_STEP.*"))


def test_invalid_marker_is_rejected(tmp_path: Path) -> None:
    try:
        MODULE.write_pause_marker(
            tmp_path / "PAUSED_AT_STEP", global_step=999, max_steps=5000, target_step=1000
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid pause marker rejection")
