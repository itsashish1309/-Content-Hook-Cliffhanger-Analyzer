"""Pure helpers: timestamps, score parsing/aggregation, validators, exports. No dependencies."""
from tests._runner import run

import config as cfg
from core.common import (
    fmt_ts, segments_in_window, speed_score, to_score, transcript_to_srt, validate_alternatives,
    weakest_dimensions, weighted_score,
)
from tests.fixtures import sample_segments


def test_fmt_ts():
    assert fmt_ts(65.3) == "01:05"
    assert fmt_ts(65.3, precise=True) == "01:05.3"
    assert fmt_ts(59.96, precise=True) == "01:00.0"   # rounding carries into minutes
    assert fmt_ts(None) == "--:--"


def test_to_score_accepts_llm_formats():
    assert to_score(7) == 7.0
    assert to_score("8/10") == 8.0
    assert to_score(12) == 10.0          # clamped
    assert to_score("-3") == 0.0
    try:
        to_score("high")
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_weighted_score_renormalises_missing():
    weights = {"a": 0.5, "b": 0.5}
    assert weighted_score({"a": 8, "b": 4}, weights) == 6.0
    assert weighted_score({"a": 8, "b": None}, weights) == 8.0
    assert weighted_score({"a": None, "b": None}, weights) is None


def test_speed_score_table():
    assert speed_score(0.0) == 10.0
    assert speed_score(2.0) == 8.0
    assert speed_score(4.0) == 4.0
    assert speed_score(10.0) == 1.0


def test_weakest_dimensions_tie_break_by_weight():
    sub = {"conflict_tension": 5, "stakes_clarity": 5, "speed": 9}
    assert weakest_dimensions(sub, cfg.HOOK_WEIGHTS, n=1) == ["conflict_tension"]  # 0.25 > 0.15


def test_window_selection_includes_boundary_lines():
    segs = sample_segments()
    window = segments_in_window(segs, 0, 10.0)
    assert [s["id"] for s in window] == [0, 1, 2]   # line 2 starts at 8.2 s and crosses 10 s


def test_validate_alternatives():
    data = {"rewrites": [{"line": "a"}, {"line": "b", "english": "B"}, "c", {"line": ""}]}
    out = validate_alternatives(data, "rewrites", "line")
    assert [r["line"] for r in out["rewrites"]] == ["a", "b", "c"]
    assert out["rewrites"][0]["english"] == "a"          # English defaults to the line itself
    try:
        validate_alternatives({"rewrites": [{"line": "only one"}]}, "rewrites", "line")
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_srt_export():
    srt = transcript_to_srt(sample_segments()[:2])
    assert srt.startswith("1\n00:00:01,200 --> 00:00:04,000\n")


if __name__ == "__main__":
    raise SystemExit(run(globals()))
