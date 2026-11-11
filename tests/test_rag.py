"""RAG: chat chunking (pure) + benchmark seeding / similarity search on a temp ChromaDB."""
import tempfile
from pathlib import Path

from tests._runner import require, run

import config as cfg
from tests.fixtures import FakeEmbeddings, sample_segments


def _isolated_chroma():
    """Point the persistent store at a temp folder so tests never touch data/chroma_db."""
    require("chromadb")
    require("langchain_core")
    from core import rag_engine

    cfg.CHROMA_DIR = Path(tempfile.mkdtemp())
    rag_engine.get_persistent_client.cache_clear()
    return rag_engine


def test_chat_documents_are_time_windows():
    require("langchain_core")
    from core.rag_engine import build_chat_documents

    docs = build_chat_documents(sample_segments(), window_s=20.0)
    assert len(docs) >= 5
    assert docs[0]["text"].startswith("[00:01]") and docs[0]["start"] == 1.2
    assert all(d["end"] > d["start"] for d in docs)


def test_benchmark_seed_and_similarity():
    rag = _isolated_chroma()
    emb = FakeEmbeddings()
    collection = rag.get_benchmark_collection(emb)
    assert collection.count() == 24                      # 12 hooks + 12 cliffhangers (synthetic seeds)
    similar = rag.find_similar("someone cut the brakes of your wife's car", "hook", embeddings=emb)
    assert similar and similar[0]["source"] == "seed" and 0 <= similar[0]["similarity"] <= 1
    assert all(item["episode_id"].startswith("seed-") for item in similar)


def test_add_episode_and_exclude_itself():
    rag = _isolated_chroma()
    emb = FakeEmbeddings()
    hook = {"score": 6.5, "window_text": "Arjun ye locker ki chaabi", "window_text_en": "Arjun this locker key",
            "sub_scores": {"conflict_tension": 7}}
    cliff = {"score": 7.4, "window_text": "Haan Meera. Main.", "window_text_en": "Yes Meera. Me.", "sub_scores": {}}
    assert rag.add_to_benchmark("file-test", "Test episode", "hinglish", hook, cliff, embeddings=emb) == 2
    assert rag.add_to_benchmark("file-test", "Test episode", "hinglish", hook, cliff, embeddings=emb) == 2
    assert rag.benchmark_stats("hook", emb)["analysed"] == 1          # upsert, not duplicate
    found = rag.find_similar("Arjun this locker key", "hook", exclude_episode="file-test", embeddings=emb)
    assert all(item["episode_id"] != "file-test" for item in found)


if __name__ == "__main__":
    raise SystemExit(run(globals()))
