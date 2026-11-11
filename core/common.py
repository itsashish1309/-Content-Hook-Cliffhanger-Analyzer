"""
core/common.py — small, dependency-free helpers shared by every analyser.

Kept pure (no LLM, no I/O) so they are trivially unit-testable: timestamp
formatting, window selection, score parsing/aggregation and the validators
for the LLM's "3 alternatives" outputs.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

import config as cfg

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


# ── Formatting ────────────────────────────────────────────────────────────────
def fmt_ts(seconds: float | None, precise: bool = False) -> str:
    """65.3 → '01:05' (or '01:05.3' when precise)."""
    if seconds is None:
        return "--:--"
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "--:--"
    if math.isnan(value):
        return "--:--"
    value = max(0.0, round(value, 1))
    minutes, secs = divmod(value, 60)
    if precise:
        return f"{int(minutes):02d}:{secs:04.1f}"
    return f"{int(minutes):02d}:{int(secs):02d}"


def fmt_range(start: float, end: float, precise: bool = True) -> str:
    return f"{fmt_ts(start, precise)}–{fmt_ts(end, precise)}"


def truncate(text: Any, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def word_count(text: str) -> int:
    return len((text or "").split())


# ── Segments ──────────────────────────────────────────────────────────────────
def segments_in_window(segments: Iterable[dict], start: float, end: float) -> list[dict]:
    """Segments that overlap [start, end) — a line crossing the boundary is included."""
    return [s for s in segments if s["end"] > start and s["start"] < end]


def transcript_block(segments: Iterable[dict], *, english: bool = False,
                     max_chars: int | None = None, keep: str = "end") -> str:
    """Render segments as '[mm:ss.s–mm:ss.s] text' lines for prompts."""
    lines = []
    for seg in segments:
        line = f"[{fmt_range(seg['start'], seg['end'])}] {seg['text']}"
        text_en = (seg.get("text_en") or "").strip()
        if english and text_en and text_en.lower() != seg["text"].strip().lower():
            line += f"   (EN: {text_en})"
        lines.append(line)
    text = "\n".join(lines)
    if max_chars and len(text) > max_chars:
        text = ("…\n" + text[-max_chars:]) if keep == "end" else (text[:max_chars] + "\n…")
    return text or "(no dialogue)"


def first_spoken_line(segments: list[dict], min_words: int = 2) -> float | None:
    """Start time of the first line with at least `min_words` words."""
    for seg in segments:
        if word_count(seg["text"]) >= min_words:
            return float(seg["start"])
    return float(segments[0]["start"]) if segments else None


def emotion_hint(emotion: dict | None, start: float, end: float, limit: int = 12) -> str:
    """Compact 'mm:ss label n/10' list of emotion ratings inside a window, for prompts."""
    if not emotion or not emotion.get("points"):
        return "not available"
    points = [p for p in emotion["points"]
              if p["end"] > start and p["start"] < end and p.get("intensity") is not None]
    if not points:
        return "no rated lines in this window"
    return ", ".join(f"{fmt_ts(p['start'])} {p['emotion']} {p['intensity']:.0f}/10" for p in points[:limit])


# ── Numbers & scores ──────────────────────────────────────────────────────────
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def to_float(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return default if math.isnan(float(value)) else float(value)
    if isinstance(value, str):
        match = _NUMBER.search(value)
        if match:
            return float(match.group())
    return default


def to_score(value: Any) -> float:
    """Parse an LLM score (7, 7.5, "7", "7/10") into a 0–10 float; raise if impossible."""
    number = to_float(value)
    if number is None:
        raise ValueError(f"not a numeric score: {value!r}")
    return round(clamp(number, 0.0, 10.0), 1)


def weighted_score(scores: dict, weights: dict) -> float | None:
    """Weighted mean over the dimensions that have a score (weights re-normalised)."""
    total, weight_sum = 0.0, 0.0
    for key, weight in weights.items():
        value = scores.get(key)
        if value is None:
            continue
        total += weight * float(value)
        weight_sum += weight
    return round(total / weight_sum, 1) if weight_sum > 0 else None


def weakest_dimensions(sub_scores: dict, weights: dict, n: int = 2) -> list[str]:
    """Lowest-scoring dimensions; ties broken by the higher weight (bigger lever)."""
    ranked = sorted(
        (value, -weights.get(key, 0.0), key) for key, value in sub_scores.items() if value is not None
    )
    return [key for _, _, key in ranked[:n]]


def speed_score(first_line_s: float | None) -> float | None:
    """Seconds until the first meaningful line → 0–10 (see config.SPEED_SCORE_STEPS)."""
    if first_line_s is None:
        return None
    for limit, score in cfg.SPEED_SCORE_STEPS:
        if first_line_s <= limit:
            return score
    return 1.0


def str_list(value: Any, max_items: int = 4, max_len: int = 160) -> list[str]:
    """Coerce an LLM field into a short list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [part for part in re.split(r"[\n;•]+", value) if part.strip()]
    if not isinstance(value, (list, tuple)):
        value = [value]
    out = []
    for item in value:
        text = truncate(item, max_len)
        if text:
            out.append(text)
    return out[:max_items]


# ── Transcript exports ────────────────────────────────────────────────────────
def _srt_time(seconds: float) -> str:
    total_ms = int(round(max(0.0, float(seconds)) * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def transcript_to_srt(segments: list[dict], english: bool = False) -> str:
    """Subtitles file — writers can drop it straight into an editor."""
    blocks = []
    for i, seg in enumerate(segments, start=1):
        text = (seg.get("text_en") or seg["text"]) if english else seg["text"]
        blocks.append(f"{i}\n{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n{text}\n")
    return "\n".join(blocks)


def transcript_to_txt(segments: list[dict], title: str = "") -> str:
    lines = [title, "=" * len(title), ""] if title else []
    for seg in segments:
        lines.append(f"[{fmt_range(seg['start'], seg['end'])}] {seg['text']}")
        text_en = (seg.get("text_en") or "").strip()
        if text_en and text_en.lower() != seg["text"].strip().lower():
            lines.append(f"    EN: {text_en}")
    return "\n".join(lines) + "\n"


# ── Validators for "give me 3 alternatives" outputs ───────────────────────────
def validate_alternatives(data: dict, key: str, text_key: str, expected: int = 3) -> dict:
    """
    Validate {"<key>": [{"<text_key>": ..., "english": ..., "technique": ..., "why": ...}]}.
    Raises ValueError (→ one repair retry) unless `expected` usable items exist.
    """
    items = data.get(key)
    if not isinstance(items, list):
        raise ValueError(f"'{key}' must be a list")
    clean = []
    for item in items:
        if isinstance(item, str):
            item = {text_key: item}
        if not isinstance(item, dict):
            continue
        text = str(item.get(text_key) or "").strip()
        if not text:
            continue
        clean.append({
            text_key: text,
            "english": str(item.get("english") or "").strip() or text,
            "technique": truncate(item.get("technique") or "", 60),
            "why": truncate(item.get("why") or "", 220),
        })
    if len(clean) < expected:
        raise ValueError(f"expected {expected} items in '{key}', got {len(clean)}")
    return {key: clean[:expected]}
