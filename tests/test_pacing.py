"""Pacing maths (numpy only — no audio, no LLM)."""
from tests._runner import run

from core.pacing import annotate_slow_sections, compute_pacing, dialogue_gaps
from tests.fixtures import SAMPLE_DURATION, SAMPLE_SILENCES, sample_segments


def test_dialogue_gaps_and_longest_gap():
    gaps = dialogue_gaps(sample_segments(), SAMPLE_DURATION)
    longest = max(gaps, key=lambda g: g["length"])
    assert longest["start"] == 59.0 and longest["end"] == 66.0   # the locker-opening silence
    assert gaps[0]["start"] == 0.0 and gaps[0]["end"] == 1.2     # dead air before the first line


def test_compute_pacing_numbers():
    pacing = compute_pacing(sample_segments(), SAMPLE_DURATION, SAMPLE_SILENCES)
    assert 0 <= pacing["score"] <= 10
    assert pacing["total_words"] > 150
    assert 1.0 < pacing["wps"] < 3.0
    assert 0 < pacing["silence_ratio"] < 0.3
    assert len(pacing["bins"]) == 12                       # 116 s → 11 full windows + 1 partial
    assert pacing["longest_gap"]["length"] == 7.0
    starts = [s["start"] for s in pacing["slow_sections"]]
    assert any(abs(s - 59.0) < 0.01 for s in starts), starts  # the 7 s gap is flagged


def test_slow_sections_are_classified_with_emotion():
    pacing = compute_pacing(sample_segments(), SAMPLE_DURATION, SAMPLE_SILENCES)
    calm = {"points": [{"start": s["start"], "end": s["end"], "intensity": 2.0} for s in sample_segments()]}
    annotated = annotate_slow_sections(pacing, calm)
    assert all(s["kind"] == "drag" for s in annotated["slow_sections"])
    hot = {"points": [{"start": s["start"], "end": s["end"], "intensity": 9.0} for s in sample_segments()]}
    annotated = annotate_slow_sections(compute_pacing(sample_segments(), SAMPLE_DURATION), hot)
    assert all(s["kind"] == "pause" for s in annotated["slow_sections"])


def test_empty_transcript_is_safe():
    pacing = compute_pacing([], 30.0, [])
    assert pacing["score"] == 0.0 and pacing["longest_gap"]["length"] == 30.0


if __name__ == "__main__":
    raise SystemExit(run(globals()))
