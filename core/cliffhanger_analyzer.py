"""
core/cliffhanger_analyzer.py — "Will viewers unlock the next episode?"

Same diagnose → prescribe pattern as the hook analyser:
  1. DIAGNOSE (temperature 0): unresolved question, rising stakes, reveal/shock,
     emotional peak at the cut, and the "would I pay to see next?" gut check —
     plus the best timestamp to cut the episode.
  2. PRESCRIBE (temperature 0.7): 3 alternative endings in the episode's language.

The best cut is constrained to real segment end-times (the LLM picks from a
candidate list; the code snaps to the nearest one), so the suggestion is always
an edit an editor can actually make — usually the cheapest fix in the report.
"""
from __future__ import annotations

import config as cfg
from core.common import (
    emotion_hint, fmt_ts, segments_in_window, str_list, to_float, to_score, transcript_block,
    validate_alternatives, weakest_dimensions, weighted_score,
)
from core.llm import run_json_chain
from core.prompts import CLIFF_REWRITE_PROMPT, CLIFF_SCORE_PROMPT

CLIFF_SCORE_FALLBACK = {
    "sub_scores": {dim: None for dim in cfg.CLIFF_WEIGHTS},
    "best_cut_s": None,
    "cut_reason": "",
    "open_question": "",
    "reason": "Cliffhanger scoring is unavailable — the LLM did not return valid JSON after a retry. "
              "Run the analysis again.",
    "strengths": [],
    "weaknesses": [],
}


def validate_cliff_scores(data: dict) -> dict:
    """Check the scoring JSON: all five dimensions + a reason are required."""
    sub = data.get("sub_scores")
    if not isinstance(sub, dict):
        raise ValueError("missing 'sub_scores' object")
    clean = {}
    for dim in cfg.CLIFF_WEIGHTS:
        if dim not in sub:
            raise ValueError(f"missing sub_scores.{dim}")
        clean[dim] = to_score(sub[dim])
    reason = str(data.get("reason") or "").strip()
    if not reason:
        raise ValueError("missing 'reason'")
    return {
        "sub_scores": clean,
        "best_cut_s": to_float(data.get("best_cut_s")),
        "cut_reason": str(data.get("cut_reason") or "").strip(),
        "open_question": str(data.get("open_question") or "").strip(),
        "reason": reason,
        "strengths": str_list(data.get("strengths")),
        "weaknesses": str_list(data.get("weaknesses")),
    }


def cut_candidates(segments: list[dict], duration: float) -> list[float]:
    """Segment end-times in the last CUT_SEARCH_S seconds, plus the current ending."""
    zone_start = max(0.0, duration - cfg.CUT_SEARCH_S)
    ends = {round(s["end"], 1) for s in segments if zone_start <= s["end"] <= duration + 0.5}
    ends.add(round(duration, 1))
    return sorted({min(e, round(duration, 1)) for e in ends})


def snap_to_candidate(value: float | None, candidates: list[float], default: float) -> float:
    if value is None or not candidates:
        return default
    return min(candidates, key=lambda c: abs(c - value))


def analyze_cliffhanger(segments: list[dict], duration: float, language: str,
                        emotion: dict | None = None) -> dict:
    """Score the last CLIFF_WINDOW_S seconds, suggest a cut point and 3 alternative endings."""
    duration = float(duration)
    window_start = max(0.0, duration - cfg.CLIFF_WINDOW_S)
    window = segments_in_window(segments, window_start, duration + 0.01)
    lead_up = [s for s in segments if s["end"] <= window_start + 0.01]
    lang = cfg.LANGUAGES.get(language, cfg.LANGUAGES[cfg.DEFAULT_LANGUAGE])
    candidates = cut_candidates(segments, duration)
    notes: list[str] = []
    errors: list[str] = []
    if not window:
        notes.append(f"No dialogue in the last {cfg.CLIFF_WINDOW_S:.0f} s — the ending is music or "
                     "visuals, which HookLens can't hear. Judge the final beat by eye.")

    window_text = (transcript_block(window, english=True) if window
                   else "(no dialogue in this window — only music or visuals)")
    lead_up_text = transcript_block(lead_up, max_chars=3500, keep="end") if lead_up else "(none)"

    # ── 1. Diagnose ─────────────────────────────────────────────────────────
    scoring = run_json_chain(
        CLIFF_SCORE_PROMPT,
        {
            "window_s": f"{cfg.CLIFF_WINDOW_S:.0f}",
            "duration_s": f"{duration:.1f}",
            "language_label": lang["prompt_label"],
            "duration": fmt_ts(duration),
            "emotion_hint": emotion_hint(emotion, max(0.0, duration - cfg.CUT_SEARCH_S), duration + 1),
            "lead_up": lead_up_text,
            "window_start": fmt_ts(window_start, precise=True),
            "window_transcript": window_text,
            "cut_candidates": ", ".join(f"{c:.1f}" for c in candidates),
        },
        validate=validate_cliff_scores,
        fallback=CLIFF_SCORE_FALLBACK,
        name="cliffhanger_score",
        temperature=cfg.TEMPERATURE_SCORING,
    )
    llm_ok = not scoring.get("_fallback")
    if not llm_ok:
        errors.append(scoring.get("_error", "cliffhanger scoring failed"))

    sub_scores = dict(scoring["sub_scores"])
    score = weighted_score(sub_scores, cfg.CLIFF_WEIGHTS) if llm_ok else None
    weakest = weakest_dimensions(sub_scores, cfg.CLIFF_WEIGHTS) if llm_ok else []
    best_cut = snap_to_candidate(scoring.get("best_cut_s"), candidates, default=round(duration, 1))
    trim_s = round(max(0.0, duration - best_cut), 1)

    # ── 2. Prescribe ────────────────────────────────────────────────────────
    weakest_text = ", ".join(cfg.CLIFF_DIMENSIONS[d].lower() for d in weakest) or "the ending overall"
    diagnosis = scoring.get("reason", "")
    if score is not None:
        diagnosis = f"Cliffhanger score {score}/10. {diagnosis}"
    endings = run_json_chain(
        CLIFF_REWRITE_PROMPT,
        {
            "write_in": lang["write_in"],
            "weakest": weakest_text,
            "window_transcript": window_text,
            "diagnosis": diagnosis,
            "lead_up": lead_up_text,
        },
        validate=lambda d: validate_alternatives(d, "endings", "ending"),
        fallback={"endings": []},
        name="cliffhanger_rewrite",
        temperature=cfg.TEMPERATURE_REWRITE,
    )
    if endings.get("_fallback"):
        errors.append(endings.get("_error", "alternative endings failed"))

    return {
        "score": score,
        "sub_scores": sub_scores,
        "weights": dict(cfg.CLIFF_WEIGHTS),
        "window": [round(window_start, 2), round(duration, 2)],
        "window_segments": window,
        "window_text": " ".join(s["text"] for s in window),
        "window_text_en": " ".join((s.get("text_en") or s["text"]) for s in window),
        "best_cut_s": best_cut,
        "trim_s": trim_s,
        "cut_reason": scoring.get("cut_reason", ""),
        "cut_candidates": candidates,
        "open_question": scoring.get("open_question", ""),
        "reason": scoring.get("reason", ""),
        "strengths": scoring.get("strengths", []),
        "weaknesses": scoring.get("weaknesses", []),
        "weakest": weakest,
        "endings": endings.get("endings", []),
        "notes": notes,
        "errors": errors,
        "fallback": not llm_ok,
    }
