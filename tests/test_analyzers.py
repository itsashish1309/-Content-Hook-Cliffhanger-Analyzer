"""Hook, cliffhanger, emotion and verdict analysers with a rule-based fake LLM (offline)."""
from tests._runner import require, run

import config as cfg
from tests.fixtures import SAMPLE_DURATION, SAMPLE_SILENCES, fake_llm_factory, sample_segments


def _with_fake_llm():
    require("langchain_core")
    from core.llm import set_llm_factory

    set_llm_factory(fake_llm_factory())


def test_emotion_timeline():
    _with_fake_llm()
    from core.emotion import analyze_emotions

    emotion = analyze_emotions(sample_segments(), "hinglish", SAMPLE_DURATION)
    assert not emotion["fallback"] and len(emotion["points"]) == len(sample_segments())
    assert all(p["emotion"] in cfg.EMOTIONS for p in emotion["points"])
    assert any(abs(p["time"] - 106.5) < 0.01 for p in emotion["peaks"]), emotion["peaks"]  # "Haan, Meera. Main."
    assert 0 <= emotion["score"] <= 10 and len(emotion["trend"]) == len(emotion["points"])


def test_hook_analysis():
    _with_fake_llm()
    from core.hook_analyzer import analyze_hook

    hook = analyze_hook(sample_segments(), SAMPLE_DURATION, "hinglish")
    assert set(hook["sub_scores"]) == set(cfg.HOOK_WEIGHTS)
    assert hook["sub_scores"]["speed"] == 9.0               # first meaningful line at 1.2 s
    raw = sum(hook["sub_scores"][k] * w for k, w in cfg.HOOK_WEIGHTS.items())
    assert abs(hook["score"] - raw) <= 0.051                 # deterministic weighted aggregation
    assert len(hook["rewrites"]) == 3 and hook["rewrites"][0]["english"]
    assert hook["weakest"][0] == "stakes_clarity"


def test_cliffhanger_best_cut_snaps_to_segment_end():
    _with_fake_llm()
    from core.cliffhanger_analyzer import analyze_cliffhanger

    cliff = analyze_cliffhanger(sample_segments(), SAMPLE_DURATION, "hinglish")
    assert cliff["best_cut_s"] == 110.0 and cliff["trim_s"] == 6.0   # LLM said 109.7 → snapped
    assert len(cliff["endings"]) == 3 and cliff["score"] is not None


def test_verdict_ranks_fixes_by_impact():
    _with_fake_llm()
    from core.cliffhanger_analyzer import analyze_cliffhanger
    from core.emotion import analyze_emotions
    from core.hook_analyzer import analyze_hook
    from core.pacing import annotate_slow_sections, compute_pacing
    from core.verdict import build_verdict, compute_risk

    segs = sample_segments()
    emotion = analyze_emotions(segs, "hinglish", SAMPLE_DURATION)
    pacing = annotate_slow_sections(compute_pacing(segs, SAMPLE_DURATION, SAMPLE_SILENCES), emotion)
    hook = analyze_hook(segs, SAMPLE_DURATION, "hinglish", emotion)
    cliff = analyze_cliffhanger(segs, SAMPLE_DURATION, "hinglish", emotion)
    verdict = build_verdict(hook, cliff, pacing, emotion, segs, "hinglish")

    impacts = [f["impact"] for f in verdict["fixes"]]
    assert impacts == sorted(impacts, reverse=True) and 1 <= len(impacts) <= 3
    assert verdict["risk"] in ("Low", "Medium", "High") and verdict["summary"]
    assert any(f["id"] == "cliff-cut" for f in verdict["fixes"])   # cheap re-cut makes the top 3

    # Weakest-link rule: a 3/10 cliffhanger forces High even if everything else is great.
    _, band, reasons = compute_risk({"hook": 9.0, "cliffhanger": 3.0, "pacing": 9.0, "emotion": 9.0})
    assert band == "High" and any("weakest-link" in r for r in reasons)
    _, band, _ = compute_risk({"hook": 8.0, "cliffhanger": 7.5, "pacing": 7.0, "emotion": 6.0})
    assert band == "Low"


def test_llm_failure_degrades_gracefully():
    require("langchain_core")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from core.hook_analyzer import analyze_hook
    from core.llm import set_llm_factory

    set_llm_factory(lambda temperature, json_mode: FakeListChatModel(responses=["not json at all"]))
    hook = analyze_hook(sample_segments(), SAMPLE_DURATION, "hinglish")
    assert hook["score"] is None and hook["fallback"] and hook["errors"]
    assert hook["rewrites"] == []
    set_llm_factory(None)


if __name__ == "__main__":
    raise SystemExit(run(globals()))
