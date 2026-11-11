"""
Whole pipeline, offline: media + transcription + silence detection are patched
with the sample episode; the LLM is the rule-based fake; embeddings are fake
and ChromaDB (if installed) uses a temp folder. Exercises every step and the
failure policy (optional steps may fail without losing the analysis).
"""
import json
import tempfile
from pathlib import Path

from tests._runner import require, run

import config as cfg
from tests.fixtures import (
    SAMPLE_DURATION, SAMPLE_SILENCES, FakeEmbeddings, fake_llm_factory, sample_segments,
)


def run_offline_pipeline(build_chat: bool = True):
    require("langchain_core")
    from core import pipeline
    from core.llm import set_llm_factory

    set_llm_factory(fake_llm_factory())
    cfg.CHROMA_DIR = Path(tempfile.mkdtemp())
    try:
        from core import rag_engine

        rag_engine.get_persistent_client.cache_clear()
    except Exception:  # noqa: BLE001 — chromadb missing: benchmark step will fail gracefully
        pass

    pipeline.prepare_media = lambda source, display_name=None: {
        "source": source, "source_type": "file", "title": "Locker Ka Raaz · Ep 3", "media_id": "file-offline",
        "youtube_url": None, "video_path": None, "is_video": True, "uploader": None,
        "wav_path": "unused.wav", "chunks": ["unused.wav"], "chunk_seconds": 600, "duration": SAMPLE_DURATION,
    }
    pipeline.transcribe_all = lambda chunks, language, **kw: {
        "segments": sample_segments(), "text": " ".join(s["text"] for s in sample_segments()),
        "language": language, "engine": "fake:test", "warnings": [],
    }
    pipeline.detect_silences = lambda wav_path: SAMPLE_SILENCES
    steps = []
    result, chain = pipeline.run_pipeline(
        "episode.mp4", "hinglish", embeddings_loader=FakeEmbeddings, build_chat=build_chat,
        on_step=lambda key, state: steps.append((key, state)),
    )
    result["_steps"] = steps
    return result, chain


def test_pipeline_end_to_end():
    result, chain = run_offline_pipeline()
    for key in ("hook", "cliffhanger", "pacing", "emotion", "verdict", "transcript", "benchmark"):
        assert key in result, key
    assert result["verdict"]["risk"] in ("Low", "Medium", "High")
    assert result["hook"]["score"] is not None and result["cliffhanger"]["best_cut_s"] == 110.0
    json.dumps(result, ensure_ascii=False)                 # must be JSON-serialisable for downloads
    states = dict(result.pop("_steps"))
    assert all(states.get(k) in ("done", "error") for k in ("audio", "transcript", "hook", "verdict"))


def test_chat_chain_answers_when_chromadb_available():
    require("chromadb")
    from core.rag_engine import ask_question

    _, chain = run_offline_pipeline()
    assert chain is not None
    assert "[01:46]" in ask_question(chain, "Which line is the emotional peak?")


if __name__ == "__main__":
    raise SystemExit(run(globals()))
