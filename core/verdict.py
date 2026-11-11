"""
core/verdict.py — overall Retention Risk + the top 3 fixes, ordered by impact.

What is deterministic (and why):
  • Weighted score = Σ weight × component score (config.RISK_WEIGHTS); missing
    components are dropped and the weights re-normalised.
  • Risk band: ≥ 7 Low, ≥ 4.5 Medium, else High — plus a weakest-link rule: if
    the hook OR the cliffhanger is below 3.5 the risk is High regardless, because
    one broken end of the funnel loses the viewer (swipe-away or no unlock).
  • Fix ranking: impact ≈ component weight × headroom to a target score of 8
    (or a sub-dimension's share of it). That is an estimate of how many
    weighted-score points the fix could recover, so "rewrite the hook" outranks
    "tighten 00:40–00:50" when the hook is the bigger leak.

What the LLM does: write the 2-line summary, a one-line logline and crisp
wording for each already-ranked fix. If it fails, templates are used instead.
"""
from __future__ import annotations

import config as cfg
from core.common import fmt_ts, transcript_block, truncate, weighted_score
from core.llm import run_json_chain
from core.prompts import VERDICT_PROMPT


def compute_risk(components: dict) -> tuple[float | None, str, list[str]]:
    """Weighted score → (overall, band, reasons)."""
    overall = weighted_score(components, cfg.RISK_WEIGHTS)
    if overall is None:
        return None, "Unknown", ["Not enough of the analysis succeeded to compute a risk score."]
    if overall >= cfg.RISK_LOW_FROM:
        band = "Low"
    elif overall >= cfg.RISK_MEDIUM_FROM:
        band = "Medium"
    else:
        band = "High"
    reasons = [f"Weighted score {overall:.1f}/10 → {band} risk band."]
    for key, label in (("hook", "Hook"), ("cliffhanger", "Cliffhanger")):
        value = components.get(key)
        if value is not None and value < cfg.CRITICAL_COMPONENT:
            band = "High"
            reasons.append(f"{label} is {value:.1f}/10 (below {cfg.CRITICAL_COMPONENT}) — weakest-link rule: "
                           "one broken end of the funnel is enough to lose the viewer.")
    missing = [k for k in cfg.RISK_WEIGHTS if components.get(k) is None]
    if missing:
        reasons.append("Not scored (weights re-normalised): " + ", ".join(missing) + ".")
    return overall, band, reasons


def rank_fixes(hook: dict, cliff: dict, pacing: dict, emotion: dict, max_fixes: int = 3) -> list[dict]:
    """Deterministic fix candidates with an impact estimate, highest impact first."""
    w, target = cfg.RISK_WEIGHTS, cfg.FIX_TARGET_SCORE
    candidates: list[dict] = []

    hook_score = hook.get("score")
    if hook_score is not None:
        first_line = hook.get("first_meaningful_line_s")
        speed = (hook.get("sub_scores") or {}).get("speed")
        if first_line is not None and first_line > 2.5 and speed is not None:
            candidates.append({
                "id": "hook-speed", "component": "hook",
                "title": f"Open on the first line (cut {first_line:.1f}s)",
                "action": f"Trim the first {first_line:.1f} s so the episode starts on its first story line.",
                "impact": w["hook"] * cfg.HOOK_WEIGHTS["speed"] * (10 - speed),
                "timestamp": 0.0,
            })
        if hook_score < target:
            best = (hook.get("rewrites") or [{}])[0]
            weak = ", ".join(cfg.HOOK_DIMENSIONS[d].lower() for d in hook.get("weakest", []))
            action = f"Rewrite the first 10 s to fix {weak or 'the weakest dimensions'}"
            if best.get("line"):
                action += f" — e.g. “{truncate(best['line'], 110)}”"
            candidates.append({
                "id": "hook-rewrite", "component": "hook", "title": "Rewrite the opening 10 seconds",
                "action": action + ".", "impact": w["hook"] * (target - hook_score), "timestamp": 0.0,
            })

    cliff_score = cliff.get("score")
    if cliff_score is not None:
        trim = cliff.get("trim_s") or 0.0
        cut = cliff.get("best_cut_s")
        if trim >= 1.0 and cut is not None:
            # A re-cut is a pure edit (no reshoot). It moves two dimensions: the
            # emotional peak at the cut and, with it, "would I pay to see next?".
            peak = (cliff.get("sub_scores") or {}).get("emotional_peak_at_cut") or 0.0
            reason = cliff.get("cut_reason") or "end on the peak, before the reaction fizzles"
            moved = cfg.CLIFF_WEIGHTS["emotional_peak_at_cut"] + cfg.CLIFF_WEIGHTS["would_pay_next"]
            candidates.append({
                "id": "cliff-cut", "component": "cliffhanger",
                "title": f"End the episode at {fmt_ts(cut, True)}",
                "action": f"Cut the last {trim:.1f} s — {reason}",
                "impact": w["cliffhanger"] * moved * max(0.0, target - peak),
                "timestamp": cut,
            })
        if cliff_score < target:
            best = (cliff.get("endings") or [{}])[0]
            weak = ", ".join(cfg.CLIFF_DIMENSIONS[d].lower() for d in cliff.get("weakest", []))
            action = f"Rework the final beat to add {weak or 'an unresolved question'}"
            if best.get("ending"):
                action += f" — e.g. “{truncate(best['ending'], 110)}”"
            candidates.append({
                "id": "cliff-rewrite", "component": "cliffhanger", "title": "Sharpen the cliffhanger",
                "action": action + ".", "impact": w["cliffhanger"] * (target - cliff_score),
                "timestamp": (cliff.get("window") or [None])[0],
            })

    pacing_score = pacing.get("score")
    if pacing_score is not None and pacing_score < target:
        sections = pacing.get("slow_sections") or []
        drags = [s for s in sections if s.get("kind") != "pause"] or sections
        worst = max(drags, key=lambda s: s["end"] - s["start"]) if drags else None
        if worst:
            mood = " and emotion is low" if worst.get("kind") == "drag" else ""
            candidates.append({
                "id": "pacing", "component": "pacing",
                "title": f"Tighten {fmt_ts(worst['start'])}–{fmt_ts(worst['end'])}",
                "action": f"Dialogue drops to {worst.get('avg_wps', 0):.1f} words/s here{mood} — trim pauses "
                          "or pull the next beat earlier.",
                "impact": w["pacing"] * (target - pacing_score), "timestamp": worst["start"],
            })
        else:
            gap = pacing.get("longest_gap") or {}
            candidates.append({
                "id": "pacing", "component": "pacing", "title": "Raise the dialogue density",
                "action": f"Overall {pacing.get('wps', 0):.1f} words/s; the longest silent stretch is "
                          f"{gap.get('length', 0):.1f}s at {fmt_ts(gap.get('start'))}.",
                "impact": w["pacing"] * (target - pacing_score), "timestamp": gap.get("start"),
            })

    emotion_score = emotion.get("score")
    if emotion_score is not None and emotion_score < target:
        zones = emotion.get("flat_zones") or []
        zone = max(zones, key=lambda z: z["length"]) if zones else None
        if zone:
            title = f"Add a spike around {fmt_ts(zone['start'])}"
            action = (f"{zone['length']:.0f}s of flat emotion ({fmt_ts(zone['start'])}–{fmt_ts(zone['end'])}, "
                      f"avg {zone['mean']:.1f}/10) — add a confrontation, reveal or reaction beat.")
            stamp = zone["start"]
        else:
            stats = emotion.get("stats") or {}
            title = "End on a stronger emotional beat"
            action = (f"Average intensity is {stats.get('mean', 0):.1f}/10 and the final 15 s average "
                      f"{stats.get('end_mean', 0):.1f}/10 — escalate before the cut.")
            stamp = None
        candidates.append({"id": "emotion", "component": "emotion", "title": title, "action": action,
                           "impact": w["emotion"] * (target - emotion_score), "timestamp": stamp})

    for candidate in candidates:
        candidate["impact"] = round(candidate["impact"], 2)
    candidates = [c for c in candidates if c["impact"] >= 0.05]
    candidates.sort(key=lambda c: c["impact"], reverse=True)
    return candidates[:max_fixes]


def _validate_verdict(data: dict, n_fixes: int) -> dict:
    summary = str(data.get("summary") or "").strip()
    if not summary:
        raise ValueError("missing 'summary'")
    fixes = data.get("fixes") if isinstance(data.get("fixes"), list) else []
    clean = []
    for item in fixes[:n_fixes]:
        if isinstance(item, dict):
            clean.append({"title": truncate(item.get("title") or "", 70),
                          "action": truncate(item.get("action") or "", 300)})
    return {"summary": truncate(summary, 400), "logline": truncate(data.get("logline") or "", 220),
            "fixes": clean}


def build_verdict(hook: dict, cliff: dict, pacing: dict, emotion: dict,
                  segments: list[dict], language: str) -> dict:
    """Combine all analyses into the final verdict shown on the Overview tab."""
    components = {
        "hook": hook.get("score"),
        "cliffhanger": cliff.get("score"),
        "pacing": pacing.get("score"),
        "emotion": emotion.get("score"),
    }
    overall, risk, reasons = compute_risk(components)
    fixes = rank_fixes(hook, cliff, pacing, emotion)

    def fmt(value):
        return "n/a" if value is None else f"{value:.1f}"

    fallback = {
        "summary": (f"Retention risk is {risk} (weighted {fmt(overall)}/10). "
                    + (f"Biggest lever: {fixes[0]['title'].lower()}." if fixes else "No major fixes needed.")),
        "logline": "",
        "fixes": [],
    }
    errors: list[str] = []
    text = fallback
    if fixes or overall is not None:
        text = run_json_chain(
            VERDICT_PROMPT,
            {
                "n_fixes": str(len(fixes)),
                "risk": risk,
                "overall": fmt(overall),
                "hook": fmt(components["hook"]),
                "cliff": fmt(components["cliffhanger"]),
                "pacing": fmt(components["pacing"]),
                "emotion": fmt(components["emotion"]),
                "hook_reason": hook.get("reason") or "n/a",
                "cliff_reason": cliff.get("reason") or "n/a",
                "pacing_notes": " ".join(pacing.get("notes", [])) or "n/a",
                "fix_candidates": "\n".join(
                    f"{i}. [{f['component']}, impact +{f['impact']:.2f}] {f['title']}: {f['action']}"
                    for i, f in enumerate(fixes, start=1)) or "(none — the episode is in good shape)",
                "transcript_excerpt": transcript_block(segments, max_chars=2500, keep="start"),
            },
            validate=lambda d: _validate_verdict(d, len(fixes)),
            fallback=fallback,
            name="verdict",
            temperature=cfg.TEMPERATURE_SUMMARY,
        )
        if text.get("_fallback"):
            errors.append(text.get("_error", "verdict text failed"))

    # Merge the LLM's wording into the deterministic ranking (order never changes).
    for i, fix in enumerate(fixes):
        llm_fix = text.get("fixes", [])[i] if i < len(text.get("fixes", [])) else {}
        fix["template_title"], fix["template_action"] = fix["title"], fix["action"]
        if llm_fix.get("title"):
            fix["title"] = llm_fix["title"]
        if llm_fix.get("action"):
            fix["action"] = llm_fix["action"]

    return {
        "overall": overall,
        "risk": risk,
        "risk_reasons": reasons,
        "components": components,
        "weights": dict(cfg.RISK_WEIGHTS),
        "summary": text.get("summary") or fallback["summary"],
        "logline": text.get("logline", ""),
        "fixes": fixes,
        "errors": errors,
        "fallback": bool(text.get("_fallback")),
    }
