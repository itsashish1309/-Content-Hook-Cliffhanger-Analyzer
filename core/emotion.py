"""
core/emotion.py — emotion timeline: intensity 0–10 + label for every line.

How it works:
  • Segments are sent to the LLM in batches (EMOTION_BATCH_SIZE per call) with
    their ids and timestamps; temperature 0; strict JSON validated per id.
  • Missing ratings are imputed from neighbours (linear interpolation) and
    flagged, so one dropped line never breaks the chart.
  • Peaks, dips and "flat zones" are found with numpy — no LLM — so they are
    reproducible and explainable.
  • emotion score = 0.6 × mean intensity + 0.4 × intensity of the final 15 s
    ("is there energy overall, and does it end hot?").
"""
from __future__ import annotations

import numpy as np

import config as cfg
from core.common import fmt_ts, to_score, truncate
from core.llm import run_json_chain
from core.prompts import EMOTION_PROMPT


def normalise_emotion(label: object) -> str:
    text = str(label or "").strip().lower()
    if text in cfg.EMOTIONS:
        return text
    if text in cfg.EMOTION_SYNONYMS:
        return cfg.EMOTION_SYNONYMS[text]
    for word in text.replace("/", " ").replace(",", " ").split():
        if word in cfg.EMOTIONS:
            return word
        if word in cfg.EMOTION_SYNONYMS:
            return cfg.EMOTION_SYNONYMS[word]
    return "calm"


def validate_ratings(data: dict, expected_ids: set[int], min_coverage: float = 0.6) -> dict:
    """Keep ratings for known ids; demand that most of the batch is covered."""
    items = data.get("ratings")
    if not isinstance(items, list):
        raise ValueError("'ratings' must be a list")
    clean: dict[int, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            seg_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if seg_id not in expected_ids:
            continue
        try:
            intensity = to_score(item.get("intensity"))
        except ValueError:
            continue
        clean[seg_id] = {"id": seg_id, "intensity": intensity, "emotion": normalise_emotion(item.get("emotion"))}
    if expected_ids and len(clean) < max(1, int(min_coverage * len(expected_ids))):
        raise ValueError(f"ratings returned for {len(clean)} of {len(expected_ids)} segments")
    return {"ratings": list(clean.values())}


def _segments_block(batch: list[dict]) -> str:
    return "\n".join(f"id={s['id']} [{fmt_ts(s['start'], True)}] {s['text']}" for s in batch)


def find_peaks_and_dips(values: list[float], max_items: int = 5) -> tuple[list[int], list[int]]:
    """Local maxima ≥ max(PEAK_MIN, mean+0.75σ) and local minima ≤ min(DIP_MAX, mean−0.75σ)."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 3:
        return [], []
    mean, std = float(x.mean()), float(x.std())
    peak_threshold = max(cfg.PEAK_MIN, mean + 0.75 * std)
    dip_threshold = min(cfg.DIP_MAX, mean - 0.75 * std)
    peaks, dips = [], []
    for i in range(n):
        left = x[i - 1] if i > 0 else -np.inf
        right = x[i + 1] if i < n - 1 else -np.inf
        if x[i] >= peak_threshold and x[i] >= left and x[i] > right:
            peaks.append(i)
        left_d = x[i - 1] if i > 0 else np.inf
        right_d = x[i + 1] if i < n - 1 else np.inf
        if 0 < i < n - 1 and x[i] <= dip_threshold and x[i] <= left_d and x[i] < right_d:
            dips.append(i)
    peaks = sorted(sorted(peaks, key=lambda i: -x[i])[:max_items])
    dips = sorted(sorted(dips, key=lambda i: x[i])[:max_items])
    return peaks, dips


def flat_zones(points: list[dict]) -> list[dict]:
    """Runs of consecutive low-intensity lines lasting ≥ FLAT_MIN_S — energy sags."""
    zones, run = [], []
    for point in points + [None]:
        if point is not None and point["intensity"] is not None and point["intensity"] <= cfg.FLAT_MAX:
            run.append(point)
            continue
        if run:
            start, end = run[0]["start"], run[-1]["end"]
            if end - start >= cfg.FLAT_MIN_S:
                zones.append({
                    "start": round(start, 1), "end": round(end, 1), "length": round(end - start, 1),
                    "mean": round(float(np.mean([p["intensity"] for p in run])), 1),
                })
            run = []
    return zones


def rolling_mean(values: list[float], window: int = 3) -> list[float]:
    x = np.asarray(values, dtype=float)
    if len(x) < window:
        return [round(float(v), 2) for v in x]
    padded = np.pad(x, (window // 2, window - 1 - window // 2), mode="edge")
    smooth = np.convolve(padded, np.ones(window) / window, mode="valid")
    return [round(float(v), 2) for v in smooth]


def analyze_emotions(segments: list[dict], language: str, duration: float | None = None) -> dict:
    """Rate every segment, then derive peaks, dips, flat zones and an emotion score."""
    lang = cfg.LANGUAGES.get(language, cfg.LANGUAGES[cfg.DEFAULT_LANGUAGE])
    ratings: dict[int, dict] = {}
    errors: list[str] = []
    batches = [segments[i:i + cfg.EMOTION_BATCH_SIZE] for i in range(0, len(segments), cfg.EMOTION_BATCH_SIZE)]

    for number, batch in enumerate(batches, start=1):
        ids = {s["id"] for s in batch}
        result = run_json_chain(
            EMOTION_PROMPT,
            {"emotions": ", ".join(cfg.EMOTIONS), "language_label": lang["prompt_label"],
             "segments_block": _segments_block(batch)},
            validate=lambda d, ids=ids: validate_ratings(d, ids),
            fallback={"ratings": []},
            name=f"emotion_batch_{number}",
            temperature=cfg.TEMPERATURE_SCORING,
        )
        if result.get("_fallback"):
            errors.append(result.get("_error", f"emotion batch {number} failed"))
        for rating in result.get("ratings", []):
            ratings[rating["id"]] = rating

    points = []
    for seg in segments:
        rating = ratings.get(seg["id"])
        points.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "mid": round((seg["start"] + seg["end"]) / 2, 2),
            "intensity": rating["intensity"] if rating else None,
            "emotion": rating["emotion"] if rating else "calm",
            "imputed": rating is None,
            "text": seg["text"],
        })

    rated = [p for p in points if p["intensity"] is not None]
    if not rated:
        return {"points": points, "peaks": [], "dips": [], "flat_zones": [], "trend": [],
                "stats": {}, "score": None, "errors": errors or ["no emotion ratings"], "fallback": True}

    # Impute missing intensities from neighbours so the timeline stays continuous.
    if len(rated) < len(points):
        known_t = [p["mid"] for p in rated]
        known_v = [p["intensity"] for p in rated]
        for p in points:
            if p["intensity"] is None:
                p["intensity"] = round(float(np.interp(p["mid"], known_t, known_v)), 1)

    values = [p["intensity"] for p in points]
    peak_idx, dip_idx = find_peaks_and_dips(values)

    def describe(i: int) -> dict:
        p = points[i]
        return {"id": p["id"], "time": p["start"], "intensity": p["intensity"],
                "emotion": p["emotion"], "text": truncate(p["text"], 120)}

    total = float(duration or (segments[-1]["end"] if segments else 0.0))
    ending_from = max(0.0, total - cfg.CLIFF_WINDOW_S)
    ending = [p["intensity"] for p in points if p["end"] > ending_from] or values[-1:]

    # Share of spoken time per emotion (for the "emotion mix" chart).
    time_by_emotion = {label: 0.0 for label in cfg.EMOTIONS}
    for p in points:
        time_by_emotion[p["emotion"]] += max(0.0, p["end"] - p["start"])
    spoken = sum(time_by_emotion.values()) or 1.0
    shares = {label: round(seconds / spoken, 3) for label, seconds in time_by_emotion.items()}

    mean = float(np.mean(values))
    end_mean = float(np.mean(ending))
    weights = cfg.EMOTION_ARC_WEIGHTS
    score = round(min(10.0, weights["mean"] * mean + weights["ending"] * end_mean), 1)

    return {
        "points": points,
        "trend": rolling_mean(values),
        "peaks": [describe(i) for i in peak_idx],
        "dips": [describe(i) for i in dip_idx],
        "flat_zones": flat_zones(points),
        "stats": {
            "mean": round(mean, 1),
            "max": round(float(np.max(values)), 1),
            "min": round(float(np.min(values)), 1),
            "end_mean": round(end_mean, 1),
            "dominant": max(shares, key=shares.get),
            "shares": shares,
            "imputed": sum(1 for p in points if p["imputed"]),
        },
        "score": score,
        "errors": errors,
        "fallback": False,
    }
