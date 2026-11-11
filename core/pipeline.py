"""
core/pipeline.py — the single analysis pipeline shared by the UI and the CLI.

    audio → transcript → pacing → emotion → hook → cliffhanger → verdict
          → benchmark (RAG) → chat index

Why one shared function: app.py and main.py both call run_pipeline(), so the
Streamlit UI and the command line can never drift apart. The UI passes an
`on_step(key, state)` callback to drive its live status bars; the CLI passes
one that prints. Expensive models are injected as loaders so Streamlit can
cache them with st.cache_resource and only load what a language needs.

Failure policy:
  • Problems the writer can fix (bad link, no audio, missing key) raise
    HookLensError immediately, before any slow work where possible.
  • A failed LLM analysis degrades to a fallback (shown as "unavailable").
  • Optional steps (benchmark, chat) can fail without losing the analysis.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable

import config as cfg
from core.cliffhanger_analyzer import analyze_cliffhanger
from core.common import fmt_ts
from core.emotion import analyze_emotions
from core.hook_analyzer import analyze_hook
from core.llm import require_mistral_key
from core.pacing import annotate_slow_sections, compute_pacing, detect_silences
from core.transcriber import transcribe_all
from core.verdict import build_verdict
from utils.audio_processor import prepare_media
from utils.errors import HookLensError

# (key, icon, label) — also used by app.py to render the status bars.
STEPS = [
    ("audio", "🔊", "Audio processing"),
    ("transcript", "📝", "Transcription + timestamps"),
    ("pacing", "⏱️", "Pacing (numbers only)"),
    ("emotion", "💓", "Emotion timeline"),
    ("hook", "🎣", "Hook analysis"),
    ("cliffhanger", "🧗", "Cliffhanger analysis"),
    ("verdict", "⚖️", "Overall verdict"),
    ("benchmark", "📚", "RAG benchmark"),
    ("chat", "💬", "Episode chat index"),
]

StepCallback = Callable[[str, str], None]


class _Stepper:
    """Context manager per step: reports active/done/error and records timings."""

    def __init__(self, on_step: StepCallback | None):
        self.on_step = on_step or (lambda key, state: None)
        self.timings: dict[str, float] = {}
        self.warnings: list[str] = []

    @contextmanager
    def __call__(self, key: str, optional: bool = False):
        self.on_step(key, "active")
        started = time.perf_counter()
        try:
            yield
        except Exception as exc:
            self.timings[key] = round(time.perf_counter() - started, 2)
            self.on_step(key, "error")
            if optional:
                self.warnings.append(f"{key}: {exc}")
                return  # swallow — optional step must not lose the analysis
            raise
        self.timings[key] = round(time.perf_counter() - started, 2)
        self.on_step(key, "done")


def build_analysis_brief(result: dict) -> str:
    """Compact text summary injected into the chat prompt (so chat can explain the scores)."""
    hook, cliff = result["hook"], result["cliffhanger"]
    pacing, emotion, verdict = result["pacing"], result["emotion"], result["verdict"]

    def s(value):
        return "n/a" if value is None else f"{value:.1f}/10"

    hook_sub = ", ".join(f"{cfg.HOOK_DIMENSIONS[k]} {v:.0f}" for k, v in hook["sub_scores"].items() if v is not None)
    cliff_sub = ", ".join(f"{cfg.CLIFF_DIMENSIONS[k]} {v:.0f}" for k, v in cliff["sub_scores"].items() if v is not None)
    peaks = ", ".join(f"{fmt_ts(p['time'])} {p['emotion']} {p['intensity']:.0f}" for p in emotion.get("peaks", []))
    slow = ", ".join(f"{fmt_ts(x['start'])}-{fmt_ts(x['end'])} ({x['kind']})" for x in pacing.get("slow_sections", []))
    fixes = " | ".join(f"{i}. {f['title']}: {f['action']}" for i, f in enumerate(verdict.get("fixes", []), 1))
    return "\n".join([
        f"Title: {result['title']} · language {result['language']} · runtime {fmt_ts(result['duration'])}",
        f"Retention risk: {verdict['risk']} (weighted {s(verdict.get('overall'))})",
        f"Hook {s(hook['score'])} — {hook_sub}. Diagnosis: {hook.get('reason', '')}",
        f"Cliffhanger {s(cliff['score'])} — {cliff_sub}. Diagnosis: {cliff.get('reason', '')} "
        f"Suggested cut at {fmt_ts(cliff.get('best_cut_s'), True)}.",
        f"Pacing {s(pacing['score'])}: {pacing['wps']:.2f} words/s, longest gap "
        f"{pacing['longest_gap']['length']:.1f}s at {fmt_ts(pacing['longest_gap']['start'])}. Slow: {slow or 'none'}.",
        f"Emotion {s(emotion.get('score'))}: peaks {peaks or 'none'}.",
        f"Top fixes: {fixes or 'none'}",
    ])


def run_pipeline(
    source: str,
    language: str = cfg.DEFAULT_LANGUAGE,
    *,
    display_name: str | None = None,
    translate: bool = True,
    save_to_benchmark: bool = True,
    build_chat: bool = True,
    whisper_loader: Callable[[], object] | None = None,
    embeddings_loader: Callable[[], object] | None = None,
    on_step: StepCallback | None = None,
):
    """
    Analyse one episode end-to-end.
    Returns (result, chat_chain): `result` is a JSON-serialisable dict (session
    state, downloads, PDF); `chat_chain` is the LCEL RAG chain or None.
    """
    language = language if language in cfg.LANGUAGES else cfg.DEFAULT_LANGUAGE
    require_mistral_key()  # fail fast — before minutes of download/transcription
    step = _Stepper(on_step)

    with step("audio"):
        media = prepare_media(source, display_name=display_name)

    with step("transcript"):
        transcript = transcribe_all(media["chunks"], language, chunk_seconds=media["chunk_seconds"],
                                    whisper_loader=whisper_loader, translate=translate)
        if not transcript["segments"]:
            raise HookLensError("No dialogue was detected in this episode.",
                                hint="Check the language setting, or that the file has audible speech.")
    segments, duration = transcript["segments"], float(media["duration"])

    with step("pacing"):
        try:
            silences = detect_silences(media["wav_path"])
        except Exception as exc:  # pacing still works from transcript gaps alone
            step.warnings.append(f"Silence detection failed ({exc}); pacing uses transcript gaps only.")
            silences = []
        pacing = compute_pacing(segments, duration, silences)

    with step("emotion"):
        emotion = analyze_emotions(segments, language, duration)
        pacing = annotate_slow_sections(pacing, emotion)

    with step("hook"):
        hook = analyze_hook(segments, duration, language, emotion)

    with step("cliffhanger"):
        cliff = analyze_cliffhanger(segments, duration, language, emotion)

    with step("verdict"):
        verdict = build_verdict(hook, cliff, pacing, emotion, segments, language)

    result = {
        "app_version": cfg.APP_VERSION,
        "episode_id": media["media_id"],
        "title": media["title"],
        "language": language,
        "analysed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration": duration,
        "source": {
            "type": media["source_type"],
            "url": media.get("youtube_url"),
            "video_path": media.get("video_path"),
            "is_video": media.get("is_video", False),
            "uploader": media.get("uploader"),
        },
        "transcript": transcript,
        "pacing": pacing,
        "emotion": emotion,
        "hook": hook,
        "cliffhanger": cliff,
        "verdict": verdict,
        "benchmark": {"similar_hooks": [], "similar_cliffhangers": [], "saved": False, "error": None},
        "models": {"llm": cfg.MISTRAL_MODEL, "stt": transcript["engine"], "embeddings": cfg.EMBEDDING_MODEL},
    }

    embeddings = None
    with step("benchmark", optional=True):
        from core.rag_engine import add_to_benchmark, benchmark_stats, find_similar

        embeddings = embeddings_loader() if embeddings_loader else None
        bench = result["benchmark"]
        bench["similar_hooks"] = find_similar(hook.get("window_text_en") or hook.get("window_text", ""), "hook",
                                              exclude_episode=media["media_id"], embeddings=embeddings)
        bench["similar_cliffhangers"] = find_similar(cliff.get("window_text_en") or cliff.get("window_text", ""),
                                                     "cliffhanger", exclude_episode=media["media_id"],
                                                     embeddings=embeddings)
        if save_to_benchmark:
            bench["saved"] = add_to_benchmark(media["media_id"], media["title"], language, hook, cliff,
                                              embeddings=embeddings) > 0
        bench["hook_stats"] = benchmark_stats("hook", embeddings)
        bench["cliff_stats"] = benchmark_stats("cliffhanger", embeddings)
    if step.warnings and step.warnings[-1].startswith("benchmark"):
        result["benchmark"]["error"] = step.warnings[-1]

    chat_chain = None
    if build_chat:
        with step("chat", optional=True):
            from core.rag_engine import build_rag_chain

            if embeddings is None and embeddings_loader:
                embeddings = embeddings_loader()
            chat_chain = build_rag_chain(segments, analysis_brief=build_analysis_brief(result),
                                         episode_id=media["media_id"], embeddings=embeddings)

    collected = list(transcript.get("warnings", [])) + step.warnings
    for part in (hook, cliff, emotion, verdict):
        collected.extend(part.get("notes", []))
        collected.extend(e for e in part.get("errors", []) if e)
    result["warnings"] = list(dict.fromkeys(w for w in collected if w))   # de-duplicate, keep order
    result["timings"] = step.timings
    return result, chat_chain
