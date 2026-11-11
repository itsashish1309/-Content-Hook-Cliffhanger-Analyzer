"""
core/hook_analyzer.py — "Will viewers stay past the first 10 seconds?"

Two LLM calls, on purpose:
  1. DIAGNOSE (temperature 0): rubric-anchored sub-scores for conflict/tension,
     curiosity gap, emotional intensity and stakes/character clarity.
  2. PRESCRIBE (temperature 0.7): 3 rewritten openings aimed at the weakest
     dimensions, in the episode's own language + English translation.

Deterministic parts (no LLM):
  • "speed" = seconds until the first meaningful line → score via a fixed table;
  • the overall hook score = weighted mean of sub-scores (config.HOOK_WEIGHTS).
So the LLM judges each dimension; the product decides how they add up.
"""
from __future__ import annotations

import config as cfg
from core.common import (
    clamp, emotion_hint, first_spoken_line, fmt_ts, segments_in_window, speed_score, str_list,
    to_float, to_score, transcript_block, validate_alternatives, weakest_dimensions, weighted_score,
)
from core.llm import run_json_chain
from core.prompts import HOOK_REWRITE_PROMPT, HOOK_SCORE_PROMPT

HOOK_SCORE_FALLBACK = {
    "sub_scores": {dim: None for dim in cfg.HOOK_LLM_DIMENSIONS},
    "first_meaningful_line_s": None,
    "open_question": "",
    "reason": "Hook scoring is unavailable — the LLM did not return valid JSON after a retry. "
              "Run the analysis again.",
    "strengths": [],
    "weaknesses": [],
}


def validate_hook_scores(data: dict, window_s: float = cfg.HOOK_WINDOW_S) -> dict:
    """Check the scoring JSON: all four LLM dimensions + a reason are required."""
    sub = data.get("sub_scores")
    if not isinstance(sub, dict):
        raise ValueError("missing 'sub_scores' object")
    clean = {}
    for dim in cfg.HOOK_LLM_DIMENSIONS:
        if dim not in sub:
            raise ValueError(f"missing sub_scores.{dim}")
        clean[dim] = to_score(sub[dim])
    reason = str(data.get("reason") or "").strip()
    if not reason:
        raise ValueError("missing 'reason'")
    first_meaningful = to_float(data.get("first_meaningful_line_s"))
    if first_meaningful is not None:
        first_meaningful = round(clamp(first_meaningful, 0.0, window_s), 1)
    return {
        "sub_scores": clean,
        "first_meaningful_line_s": first_meaningful,
        "open_question": str(data.get("open_question") or "").strip(),
        "reason": reason,
        "strengths": str_list(data.get("strengths")),
        "weaknesses": str_list(data.get("weaknesses")),
    }


def analyze_hook(segments: list[dict], duration: float, language: str,
                 emotion: dict | None = None) -> dict:
    """Score the first HOOK_WINDOW_S seconds and propose 3 rewritten openings."""
    window_s = min(cfg.HOOK_WINDOW_S, float(duration))
    window = segments_in_window(segments, 0.0, window_s)
    after = segments_in_window(segments, window_s, window_s + 20.0)
    lang = cfg.LANGUAGES.get(language, cfg.LANGUAGES[cfg.DEFAULT_LANGUAGE])
    measured_first = first_spoken_line(segments)
    notes: list[str] = []
    errors: list[str] = []
    if not window:
        notes.append(f"No dialogue in the first {window_s:.0f} s. HookLens only hears audio, "
                     "so a purely visual hook can't be judged — treat this score with care.")

    window_text = (transcript_block(window, english=True) if window
                   else "(no dialogue in this window — only music or visuals)")

    # ── 1. Diagnose ─────────────────────────────────────────────────────────
    scoring = run_json_chain(
        HOOK_SCORE_PROMPT,
        {
            "window_s": f"{window_s:.0f}",
            "language_label": lang["prompt_label"],
            "duration": fmt_ts(duration),
            "first_line_s": f"{measured_first:.1f}" if measured_first is not None else "none",
            "emotion_hint": emotion_hint(emotion, 0.0, window_s),
            "window_transcript": window_text,
            "after_transcript": transcript_block(after) if after else "(nothing)",
        },
        validate=lambda d: validate_hook_scores(d, window_s),
        fallback=HOOK_SCORE_FALLBACK,
        name="hook_score",
        temperature=cfg.TEMPERATURE_SCORING,
    )
    llm_ok = not scoring.get("_fallback")
    if not llm_ok:
        errors.append(scoring.get("_error", "hook scoring failed"))

    # Speed is measured: the LLM only decides WHICH line is meaningful.
    first_meaningful = scoring.get("first_meaningful_line_s")
    if first_meaningful is None:
        first_meaningful = measured_first if (measured_first is not None and measured_first <= window_s) else window_s
    sub_scores = dict(scoring["sub_scores"])
    sub_scores["speed"] = speed_score(first_meaningful)
    score = weighted_score(sub_scores, cfg.HOOK_WEIGHTS) if llm_ok else None
    weakest = weakest_dimensions(sub_scores, cfg.HOOK_WEIGHTS) if llm_ok else []

    # ── 2. Prescribe ────────────────────────────────────────────────────────
    weakest_text = ", ".join(cfg.HOOK_DIMENSIONS[d].lower() for d in weakest) or "the hook overall"
    diagnosis = scoring.get("reason", "")
    if score is not None:
        diagnosis = f"Hook score {score}/10. {diagnosis}"
    rewrites = run_json_chain(
        HOOK_REWRITE_PROMPT,
        {
            "window_s": f"{window_s:.0f}",
            "write_in": lang["write_in"],
            "weakest": weakest_text,
            "window_transcript": window_text,
            "diagnosis": diagnosis,
            "episode_context": transcript_block(segments, max_chars=3000, keep="start"),
        },
        validate=lambda d: validate_alternatives(d, "rewrites", "line"),
        fallback={"rewrites": []},
        name="hook_rewrite",
        temperature=cfg.TEMPERATURE_REWRITE,
    )
    if rewrites.get("_fallback"):
        errors.append(rewrites.get("_error", "hook rewrites failed"))

    return {
        "score": score,
        "sub_scores": sub_scores,
        "weights": dict(cfg.HOOK_WEIGHTS),
        "window": [0.0, round(window_s, 2)],
        "window_segments": window,
        "window_text": " ".join(s["text"] for s in window),
        "window_text_en": " ".join((s.get("text_en") or s["text"]) for s in window),
        "first_meaningful_line_s": round(float(first_meaningful), 1),
        "measured_first_line_s": measured_first,
        "open_question": scoring.get("open_question", ""),
        "reason": scoring.get("reason", ""),
        "strengths": scoring.get("strengths", []),
        "weaknesses": scoring.get("weaknesses", []),
        "weakest": weakest,
        "rewrites": rewrites.get("rewrites", []),
        "notes": notes,
        "errors": errors,
        "fallback": not llm_ok,
    }
