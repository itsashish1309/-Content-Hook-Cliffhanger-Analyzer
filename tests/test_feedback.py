"""Feedback JSON store: save, re-vote (upsert), remove, stats — on a temp file."""
import tempfile
from pathlib import Path

from tests._runner import run

from utils.feedback import feedback_stats, load_feedback, remove_feedback, save_feedback


def _path() -> Path:
    return Path(tempfile.mkdtemp()) / "feedback.json"


def _vote(path, suggestion_id, rating, kind="hook_rewrite"):
    return save_feedback(episode_id="ep1", episode_title="Test", suggestion_id=suggestion_id,
                         suggestion_type=kind, suggestion_text="line", rating=rating, path=path)


def test_save_and_upsert():
    path = _path()
    _vote(path, "hook-rw-0", 1)
    _vote(path, "hook-rw-0", -1)          # writer changed their mind → replaces, not duplicates
    _vote(path, "fix-pacing", 1, "fix")
    records = load_feedback(path)
    assert len(records) == 2
    assert next(r for r in records if r["suggestion_id"] == "hook-rw-0")["rating"] == -1


def test_stats_and_remove():
    path = _path()
    _vote(path, "a", 1)
    _vote(path, "b", 1)
    _vote(path, "c", -1, "ending")
    stats = feedback_stats(path)
    assert stats["total"] == 3 and stats["approval_rate"] == round(2 / 3, 3)
    assert stats["by_type"]["ending"]["down"] == 1
    remove_feedback("ep1", "a", path)
    assert feedback_stats(path)["total"] == 2


def test_corrupt_file_is_recovered():
    path = _path()
    path.write_text("{not json", encoding="utf-8")
    assert load_feedback(path) == []
    _vote(path, "x", 1)
    assert len(load_feedback(path)) == 1


if __name__ == "__main__":
    raise SystemExit(run(globals()))
