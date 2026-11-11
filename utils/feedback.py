"""
utils/feedback.py — thumbs up/down on every suggestion, stored in data/feedback.json.

Why it matters: an LLM tool is only useful if writers actually adopt its
suggestions. Logging 👍/👎 per suggestion type (hook rewrite, ending, fix)
gives an approval rate we can track over time — the first step to evaluating
and improving the prompts with real users instead of guessing.

Storage is a plain JSON list (no extra database, per the stack constraint).
Writes are atomic (temp file + os.replace) and guarded by a lock so two
Streamlit sessions can't corrupt the file. A changed vote replaces the old one.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import config as cfg

_LOCK = threading.Lock()


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "[]")
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        # Never lose the app over a damaged file: keep a copy and start fresh.
        try:
            path.replace(path.with_suffix(".corrupt.json"))
        except OSError:
            pass
        return []


def _atomic_write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".feedback-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_feedback(*, episode_id: str, episode_title: str, suggestion_id: str, suggestion_type: str,
                  suggestion_text: str, rating: int, language: str = "",
                  path: Path | None = None) -> dict:
    """Record a vote (+1 / -1). Re-voting on the same suggestion replaces the old vote."""
    path = Path(path or cfg.FEEDBACK_FILE)
    record = {
        "episode_id": episode_id,
        "episode_title": episode_title,
        "suggestion_id": suggestion_id,
        "suggestion_type": suggestion_type,
        "suggestion_text": suggestion_text[:1000],
        "rating": 1 if rating > 0 else -1,
        "language": language,
        "app_version": cfg.APP_VERSION,
        "model": cfg.MISTRAL_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with _LOCK:
        records = [r for r in _read(path)
                   if not (r.get("episode_id") == episode_id and r.get("suggestion_id") == suggestion_id)]
        records.append(record)
        _atomic_write(path, records)
    return record


def remove_feedback(episode_id: str, suggestion_id: str, path: Path | None = None) -> None:
    """Delete a vote (the writer un-clicked the thumb)."""
    path = Path(path or cfg.FEEDBACK_FILE)
    with _LOCK:
        records = _read(path)
        kept = [r for r in records
                if not (r.get("episode_id") == episode_id and r.get("suggestion_id") == suggestion_id)]
        if len(kept) != len(records):
            _atomic_write(path, kept)


def load_feedback(path: Path | None = None) -> list[dict]:
    with _LOCK:
        return _read(Path(path or cfg.FEEDBACK_FILE))


def feedback_stats(path: Path | None = None) -> dict:
    """Totals and approval rate overall and per suggestion type."""
    records = load_feedback(path)

    def summarise(rows: list[dict]) -> dict:
        up = sum(1 for r in rows if r.get("rating", 0) > 0)
        down = sum(1 for r in rows if r.get("rating", 0) < 0)
        total = up + down
        return {"up": up, "down": down, "total": total,
                "approval_rate": round(up / total, 3) if total else None}

    by_type: dict[str, list[dict]] = {}
    for record in records:
        by_type.setdefault(record.get("suggestion_type", "other"), []).append(record)
    return {
        **summarise(records),
        "episodes": len({r.get("episode_id") for r in records}),
        "by_type": {kind: summarise(rows) for kind, rows in sorted(by_type.items())},
    }
