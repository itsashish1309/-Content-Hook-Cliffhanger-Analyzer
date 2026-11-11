"""
core/rag_engine.py — two RAG features on ChromaDB + sentence-transformers.

1. BENCHMARK STORE (persistent, data/chroma_db)
   Every analysed episode's hook and cliffhanger is embedded and stored with
   its scores, so a writer sees "the most similar past hooks and how they
   scored". On first run the store is seeded with clearly-labelled SYNTHETIC
   examples (data/seed_benchmark.json) so the feature works on day one; real
   episodes accumulate on top. Texts are embedded through their English
   translation when available, so a Hindi hook can match a similar Hinglish one.

2. CHAT WITH THIS EPISODE (in-memory)
   The transcript is grouped into ~20-second windows of consecutive lines (so
   every chunk carries a timestamp range the bot can cite), embedded into an
   in-memory Chroma collection, and wired into an LCEL chain. The public API is
   reused from AI Video Assistant: build_rag_chain() and ask_question().

Why chromadb directly instead of a LangChain vector-store wrapper: the
maintained wrapper is a separate package (langchain-chroma) outside the agreed
stack, and the langchain-community one is deprecated. Talking to chromadb with
our own embeddings (langchain-huggingface) is a few lines and has fewer moving
parts. (AI Video Assistant's vector_store.py is merged into this file.)
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import config as cfg
from core.common import fmt_range, fmt_ts
from core.llm import text_llm
from core.prompts import CHAT_PROMPT
from utils.errors import HookLensError


# ── Embeddings & clients ──────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_embeddings():
    """sentence-transformers via langchain-huggingface (normalised → cosine-ready)."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise HookLensError("langchain-huggingface is not installed.",
                            hint="Run: pip install -r requirements.txt") from exc
    return HuggingFaceEmbeddings(
        model_name=cfg.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _chromadb():
    try:
        import chromadb
        from chromadb.config import Settings
    except Exception as exc:  # also catches Chroma's "sqlite3 too old" RuntimeError
        raise HookLensError(f"ChromaDB is unavailable: {exc}") from exc
    return chromadb, Settings


@lru_cache(maxsize=1)
def get_persistent_client():
    chromadb, Settings = _chromadb()
    return chromadb.PersistentClient(path=str(cfg.CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


@lru_cache(maxsize=1)
def get_memory_client():
    chromadb, Settings = _chromadb()
    return chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))


def _clean_metadata(meta: dict) -> dict:
    """Chroma metadata values must be str / int / float / bool (no None, no nesting)."""
    clean = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = json.dumps(value, ensure_ascii=False)
    return clean


# ── 1. Benchmark store ────────────────────────────────────────────────────────
def get_benchmark_collection(embeddings=None):
    """Open (and seed on first use) the persistent benchmark collection."""
    client = get_persistent_client()
    collection = client.get_or_create_collection(
        name=cfg.BENCHMARK_COLLECTION,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,  # we always pass our own vectors
    )
    if collection.count() == 0:
        _seed_benchmark(collection, embeddings or get_embeddings())
    return collection


def _seed_benchmark(collection, embeddings) -> int:
    if not cfg.SEED_BENCHMARK_FILE.exists():
        return 0
    seed = json.loads(cfg.SEED_BENCHMARK_FILE.read_text(encoding="utf-8"))
    items = seed.get("items", [])
    if not items:
        return 0
    ids, docs, metas, to_embed = [], [], [], []
    for item in items:
        ids.append(f"seed:{item['id']}")
        docs.append(item["text"])
        to_embed.append(item.get("text_en") or item["text"])
        metas.append(_clean_metadata({
            "kind": item["kind"], "score": float(item["score"]), "title": item["title"],
            "language": item["language"], "source": "seed", "episode_id": f"seed-{item['id']}",
            "text_en": item.get("text_en", ""), "sub_scores": item.get("sub_scores", {}),
            "added_at": seed.get("created", ""),
        }))
    collection.upsert(ids=ids, embeddings=embeddings.embed_documents(to_embed), documents=docs, metadatas=metas)
    return len(ids)


def add_to_benchmark(episode_id: str, title: str, language: str, hook: dict, cliff: dict,
                     embeddings=None) -> int:
    """Upsert this episode's hook and cliffhanger (re-analysing replaces, never duplicates)."""
    embeddings = embeddings or get_embeddings()
    collection = get_benchmark_collection(embeddings)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ids, docs, metas, to_embed = [], [], [], []
    for kind, analysis in (("hook", hook), ("cliffhanger", cliff)):
        text = (analysis.get("window_text") or "").strip()
        if not text or analysis.get("score") is None:
            continue
        ids.append(f"{episode_id}:{kind}")
        docs.append(text)
        to_embed.append(analysis.get("window_text_en") or text)
        metas.append(_clean_metadata({
            "kind": kind, "score": float(analysis["score"]), "title": title, "language": language,
            "source": "analysed", "episode_id": episode_id, "text_en": analysis.get("window_text_en", ""),
            "sub_scores": analysis.get("sub_scores", {}), "added_at": now,
        }))
    if ids:
        collection.upsert(ids=ids, embeddings=embeddings.embed_documents(to_embed), documents=docs, metadatas=metas)
    return len(ids)


def find_similar(text: str, kind: str, *, exclude_episode: str | None = None,
                 k: int = cfg.SIMILAR_K, embeddings=None) -> list[dict]:
    """Most similar stored hooks/cliffhangers with their scores."""
    text = (text or "").strip()
    if not text:
        return []
    embeddings = embeddings or get_embeddings()
    collection = get_benchmark_collection(embeddings)
    total = collection.count()
    if total == 0:
        return []
    where = {"kind": kind}
    if exclude_episode:
        where = {"$and": [{"kind": kind}, {"episode_id": {"$ne": exclude_episode}}]}
    result = collection.query(
        query_embeddings=[embeddings.embed_query(text)],
        n_results=min(k, total),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    similar = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        meta = meta or {}
        similar.append({
            "text": doc,
            "text_en": meta.get("text_en", ""),
            "score": meta.get("score"),
            "title": meta.get("title", "Untitled"),
            "language": meta.get("language", ""),
            "source": meta.get("source", "analysed"),
            "episode_id": meta.get("episode_id", ""),
            "similarity": round(max(0.0, 1.0 - float(dist)), 3),   # cosine distance → similarity
        })
    return similar


def benchmark_stats(kind: str, embeddings=None) -> dict:
    """Count, average score and seed/analysed split for one kind."""
    collection = get_benchmark_collection(embeddings)
    rows = collection.get(where={"kind": kind}, include=["metadatas"])
    metas = rows.get("metadatas") or []
    scores = [m.get("score") for m in metas if isinstance(m.get("score"), (int, float))]
    return {
        "count": len(metas),
        "analysed": sum(1 for m in metas if m.get("source") == "analysed"),
        "seed": sum(1 for m in metas if m.get("source") == "seed"),
        "mean_score": round(sum(scores) / len(scores), 1) if scores else None,
    }


# ── 2. Chat with this episode ─────────────────────────────────────────────────
def build_chat_documents(segments: list[dict], window_s: float = cfg.CHAT_WINDOW_S) -> list[dict]:
    """Group consecutive lines into ~window_s chunks, each tagged with its time range."""
    docs, current = [], []
    for seg in segments:
        current.append(seg)
        if current[-1]["end"] - current[0]["start"] >= window_s:
            docs.append(current)
            current = [current[-1]]          # 1-line overlap keeps context across chunks
    if current and (not docs or current != docs[-1][-1:]):
        docs.append(current)
    out = []
    for group in docs:
        lines = []
        for seg in group:
            line = f"[{fmt_ts(seg['start'])}] {seg['text']}"
            text_en = (seg.get("text_en") or "").strip()
            if text_en and text_en.lower() != seg["text"].lower():
                line += f"  (EN: {text_en})"
            lines.append(line)
        out.append({"text": "\n".join(lines), "start": group[0]["start"], "end": group[-1]["end"]})
    return out


def _segments_from_text(transcript: str) -> list[dict]:
    """Backward compatibility: a plain transcript string → pseudo-segments (no real times)."""
    words, segments, step = transcript.split(), [], 60
    for i in range(0, len(words), step):
        segments.append({"id": len(segments), "start": float(len(segments) * 20),
                         "end": float(len(segments) * 20 + 20), "text": " ".join(words[i:i + step])})
    return segments


def format_docs(results: list[tuple[str, dict]]) -> str:
    ordered = sorted(results, key=lambda item: item[1].get("start", 0.0))
    return "\n\n".join(f"({fmt_range(meta.get('start', 0), meta.get('end', 0), precise=False)})\n{doc}"
                       for doc, meta in ordered)


def build_rag_chain(transcript_or_segments, *, analysis_brief: str = "", episode_id: str | None = None,
                    embeddings=None):
    """
    Reused API from AI Video Assistant. Accepts timestamped segments (preferred)
    or a plain transcript string, and returns an LCEL chain: question → answer.
    """
    segments = (_segments_from_text(transcript_or_segments)
                if isinstance(transcript_or_segments, str) else list(transcript_or_segments))
    if not segments:
        raise HookLensError("There is no transcript to chat with.")
    embeddings = embeddings or get_embeddings()
    docs = build_chat_documents(segments)

    client = get_memory_client()
    suffix = hashlib.sha1(f"{episode_id}-{uuid.uuid4()}".encode()).hexdigest()[:16]
    collection = client.create_collection(name=f"chat-{suffix}", metadata={"hnsw:space": "cosine"},
                                          embedding_function=None)
    collection.add(
        ids=[f"w{i}" for i in range(len(docs))],
        embeddings=embeddings.embed_documents([d["text"] for d in docs]),
        documents=[d["text"] for d in docs],
        metadatas=[{"start": float(d["start"]), "end": float(d["end"])} for d in docs],
    )

    def retrieve(question: str) -> str:
        result = collection.query(
            query_embeddings=[embeddings.embed_query(question)],
            n_results=min(cfg.CHAT_TOP_K, len(docs)),
            include=["documents", "metadatas"],
        )
        found = list(zip((result.get("documents") or [[]])[0], (result.get("metadatas") or [[]])[0]))
        return format_docs(found) or "(no matching lines)"

    prompt = CHAT_PROMPT.partial(analysis_brief=analysis_brief or "(no analysis available)")
    # Full LCEL RAG pipeline (same shape as AI Video Assistant's)
    rag_chain = (
        {"context": RunnableLambda(retrieve), "question": RunnablePassthrough()}
        | prompt
        | text_llm()
        | StrOutputParser()
    )
    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    """Reused API: run one question through the chain."""
    return rag_chain.invoke(question).strip()
