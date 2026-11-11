"""
core/pacing.py — "Where does the episode drag?"  Pure numbers: pydub + numpy, NO LLM.

Two complementary signals:
  • Acoustic silence (pydub.detect_silence, threshold relative to the episode's
    own loudness) → dead air.
  • Dialogue gaps from the timestamped transcript → "nobody is talking", which
    still catches drag when a music bed hides the silence (common in micro-dramas).

Outputs: words-per-second, speaking rate, silence ratio, dialogue coverage,
longest gap without dialogue, dialogue density per 10-second window, and
flagged slow sections with timestamps. annotate_slow_sections() later uses the
emotion timeline to tell a *drag* (slow AND low emotion → fix it) from an
*intentional pause* (slow but emotionally loaded → probably a dramatic beat).
"""
from __future__ import annotations

import numpy as np

import config as cfg
from core.common import clamp, fmt_ts, word_count


def detect_silences(wav_path: str) -> list[list[float]]:
    """Silent stretches as [[start_s, end_s], …] using a loudness-relative threshold."""
    from pydub import AudioSegment
    from pydub.silence import detect_silence

    audio = AudioSegment.from_wav(wav_path)
    if audio.dBFS == float("-inf"):
        return [[0.0, len(audio) / 1000.0]]
    ranges = detect_silence(
        audio,
        min_silence_len=cfg.SILENCE_MIN_MS,
        silence_thresh=audio.dBFS - cfg.SILENCE_REL_DB,
        seek_step=cfg.SILENCE_SEEK_MS,
    )
    return [[round(start / 1000, 2), round(end / 1000, 2)] for start, end in ranges]


def _union_length(intervals: list[tuple[float, float]]) -> float:
    """Total length covered by possibly-overlapping intervals."""
    total, current_start, current_end = 0.0, None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


def dialogue_gaps(segments: list[dict], duration: float) -> list[dict]:
    """Every stretch without dialogue: before the first line, between lines, after the last."""
    gaps, cursor = [], 0.0
    for seg in sorted(segments, key=lambda s: s["start"]):
        if seg["start"] > cursor:
            gaps.append({"start": round(cursor, 2), "end": round(seg["start"], 2),
                         "length": round(seg["start"] - cursor, 2)})
        cursor = max(cursor, seg["end"])
    if duration > cursor:
        gaps.append({"start": round(cursor, 2), "end": round(duration, 2), "length": round(duration - cursor, 2)})
    return gaps


def compute_pacing(segments: list[dict], duration: float, silences: list[list[float]] | None = None) -> dict:
    """All pacing numbers for one episode (deterministic, unit-tested)."""
    duration = max(float(duration), 0.1)
    silences = silences or []
    words = np.array([word_count(s["text"]) for s in segments], dtype=float)
    total_words = int(words.sum()) if len(words) else 0

    speech_time = _union_length([(s["start"], s["end"]) for s in segments]) if segments else 0.0
    silence_total = float(sum(max(0.0, end - start) for start, end in silences))
    gaps = dialogue_gaps(segments, duration)
    longest = max(gaps, key=lambda g: g["length"]) if gaps else {"start": 0.0, "end": 0.0, "length": 0.0}

    # ── Dialogue density per window (words spread proportionally over each line) ──
    edges = list(np.arange(0.0, duration, cfg.DENSITY_BIN_S)) + [duration]
    bins = []
    for b_start, b_end in zip(edges[:-1], edges[1:]):
        length = b_end - b_start
        if length <= 0:
            continue
        bin_words, covered = 0.0, []
        for seg, n_words in zip(segments, words):
            overlap = min(seg["end"], b_end) - max(seg["start"], b_start)
            if overlap <= 0:
                continue
            seg_len = max(seg["end"] - seg["start"], 1e-6)
            bin_words += n_words * overlap / seg_len
            covered.append((max(seg["start"], b_start), min(seg["end"], b_end)))
        # Cast numpy scalars to plain Python types: results must stay JSON-serialisable.
        bins.append({
            "start": round(float(b_start), 2),
            "end": round(float(b_end), 2),
            "words": round(float(bin_words), 1),
            "wps": round(float(bin_words / length), 2),
            "coverage": round(float(_union_length(covered) / length), 2),
            "full": bool(length >= cfg.DENSITY_BIN_S * cfg.MIN_BIN_FRACTION),
        })

    full_bins = [b for b in bins if b["full"]]
    median_wps = float(np.median([b["wps"] for b in full_bins])) if full_bins else 0.0
    for b in bins:
        reasons = []
        if b["full"]:
            if b["wps"] < cfg.SLOW_WPS:
                reasons.append(f"{b['wps']:.1f} words/s (< {cfg.SLOW_WPS:.1f})")
            elif median_wps > 0 and b["wps"] < cfg.SLOW_RELATIVE * median_wps:
                reasons.append(f"{b['wps']:.1f} words/s — under half this episode's median ({median_wps:.1f})")
        b["slow"] = bool(reasons)
        b["reason"] = "; ".join(reasons)

    # ── Slow sections = adjacent slow windows merged, plus long dialogue gaps ──
    sections: list[dict] = []
    for b in bins:
        if not b["slow"]:
            continue
        if sections and abs(sections[-1]["end"] - b["start"]) < 0.01 and sections[-1]["source"] == "density":
            sections[-1]["end"] = b["end"]
            sections[-1]["bins"].append(b)
        else:
            sections.append({"start": b["start"], "end": b["end"], "bins": [b], "source": "density"})
    for s in sections:
        s["avg_wps"] = round(float(np.mean([b["wps"] for b in s.pop("bins")])), 2)
        s["reasons"] = [f"low dialogue density ({s['avg_wps']:.1f} words/s)"]
    long_gaps = [g for g in gaps if g["length"] > cfg.LONG_GAP_S]
    for gap in long_gaps:
        inside = next((s for s in sections if s["start"] <= gap["start"] and gap["end"] <= s["end"]), None)
        if inside:
            inside["reasons"].append(f"{gap['length']:.1f}s without dialogue")
        else:
            sections.append({"start": gap["start"], "end": gap["end"], "avg_wps": 0.0, "source": "gap",
                             "reasons": [f"{gap['length']:.1f}s without dialogue"]})
    sections.sort(key=lambda s: s["start"])
    for s in sections:
        s["kind"] = "slow"           # refined to drag / pause by annotate_slow_sections()
        s["emotion_mean"] = None

    # ── Score: density vs target, minus gap and slow-window penalties ──
    wps = total_words / duration
    slow_fraction = (sum(1 for b in full_bins if b["slow"]) / len(full_bins)) if full_bins else 0.0
    density_component = min(10.0, 10.0 * wps / cfg.TARGET_WPS)
    gap_penalty = min(3.0, max(0.0, longest["length"] - cfg.LONG_GAP_S) * 0.75)
    slow_penalty = 4.0 * slow_fraction
    score = round(clamp(density_component - gap_penalty - slow_penalty, 0.0, 10.0), 1)

    first_line = float(segments[0]["start"]) if segments else None
    notes = [
        f"{wps:.2f} words/s overall (target ≈ {cfg.TARGET_WPS}); speaking rate "
        f"{(total_words / speech_time) if speech_time else 0:.2f} words/s while talking.",
        f"Dialogue covers {speech_time / duration:.0%} of the runtime; acoustic silence {silence_total / duration:.0%}.",
        f"Longest stretch without dialogue: {longest['length']:.1f}s at {fmt_ts(longest['start'])}.",
    ]
    if sections:
        notes.append("Slow sections: " + ", ".join(f"{fmt_ts(s['start'])}–{fmt_ts(s['end'])}" for s in sections[:4]))

    return {
        "score": score,
        "duration": round(duration, 2),
        "total_words": total_words,
        "wps": round(wps, 2),
        "speaking_rate": round(total_words / speech_time, 2) if speech_time else 0.0,
        "silence_total_s": round(silence_total, 1),
        "silence_ratio": round(silence_total / duration, 3),
        "dialogue_coverage": round(speech_time / duration, 3),
        "longest_gap": longest,
        "long_gaps": long_gaps,
        "bins": bins,
        "median_wps": round(median_wps, 2),
        "slow_sections": sections,
        "slow_fraction": round(slow_fraction, 3),
        "first_line_s": first_line,
        "silences": silences,
        "score_parts": {"density": round(density_component, 2), "gap_penalty": round(gap_penalty, 2),
                        "slow_penalty": round(slow_penalty, 2)},
        "notes": notes,
    }


def annotate_slow_sections(pacing: dict, emotion: dict | None) -> dict:
    """Cross-check slow sections with emotion: low emotion → 'drag', high → 'pause'."""
    points = (emotion or {}).get("points") or []
    for section in pacing.get("slow_sections", []):
        # Lines overlapping the section, plus the lines right next to a silent gap.
        nearby = [p for p in points
                  if p["end"] > section["start"] - 2 and p["start"] < section["end"] + 2
                  and p.get("intensity") is not None]
        if not nearby:
            continue
        mean = round(float(np.mean([p["intensity"] for p in nearby])), 1)
        section["emotion_mean"] = mean
        if mean <= 4.0:
            section["kind"] = "drag"
        elif mean >= 6.0:
            section["kind"] = "pause"
    return pacing
