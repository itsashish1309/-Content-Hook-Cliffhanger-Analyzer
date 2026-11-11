"""
core/transcriber.py — speech-to-text WITH timestamps.

Extended from AI Video Assistant (load_model, transcribe_chunk_whisper,
_send_to_sarvam, transcribe_chunk_sarvam, transcribe_chunk, transcribe_all):

  • transcribe_all() now returns segments [{id, start, end, text, text_en}] plus
    the full text. HookLens has to know exactly what is said in the first 10 s
    and the last 15 s, so a plain string is no longer enough.
  • Whisper: native segment timestamps, fp16 off on CPU, and a guard against
    the hallucinated lines Whisper invents over background music (micro-dramas
    are full of music beds).
  • Sarvam (Hindi / Hinglish): the existing integration is kept — same
    api-subscription-key header, same ≤25 s pieces, same error printing — but
    moved to Sarvam's current /speech-to-text endpoint because the old
    saaras:v2.5 /speech-to-text-translate path is being retired. Two upgrades:
      - mode="translit" keeps Hinglish in natural Roman script (the original
        language) instead of translating it to English;
      - with_timestamps=true returns phrase-level timestamps.
    Pieces are now cut at pauses instead of hard 25 s marks so words aren't split.
  • Optional English translation per segment via deep-translator.

Routing:
  english  → Whisper (local)
  hindi    → Sarvam if SARVAM_API_KEY is set, otherwise Whisper (language="hi")
  hinglish → Sarvam if SARVAM_API_KEY is set, otherwise Whisper (language="hi")
"""
from __future__ import annotations

import os
import time
from typing import Callable

import requests
from tqdm import tqdm

import config as cfg
from utils.errors import HookLensError

_model = None          # process-wide cache for the CLI (the UI uses st.cache_resource)
_model_name = None


# ── Whisper ───────────────────────────────────────────────────────────────────
def load_model(model_name: str | None = None):
    """Load (once) and return the Whisper model. Reused from AI Video Assistant."""
    global _model, _model_name
    name = model_name or cfg.WHISPER_MODEL
    if _model is None or _model_name != name:
        try:
            import whisper
        except ImportError as exc:
            raise HookLensError("openai-whisper is not installed.", hint="Run: pip install -r requirements.txt") from exc
        print(f"Loading Whisper model: {name} ...")
        _model = whisper.load_model(name)
        _model_name = name
        print("Whisper model loaded.")
    return _model


def _on_gpu(model) -> bool:
    return str(getattr(model, "device", "cpu")).startswith("cuda")


def _is_hallucination(seg: dict) -> bool:
    """
    Whisper's known failure modes over music/silence:
      • low-confidence text where it also thinks there is no speech;
      • repetition loops (very high compression ratio).
    """
    no_speech = float(seg.get("no_speech_prob", 0.0) or 0.0)
    avg_logprob = float(seg.get("avg_logprob", 0.0) or 0.0)
    compression = float(seg.get("compression_ratio", 0.0) or 0.0)
    return (no_speech > 0.6 and avg_logprob < -1.0) or compression > 2.6


def transcribe_chunk_whisper(chunk_path: str, language: str = "english", model=None) -> list[dict]:
    """Transcribe one chunk with Whisper → segments with chunk-relative timestamps."""
    model = model or load_model()
    lang_code = cfg.LANGUAGES.get(language, cfg.LANGUAGES["english"])["whisper"]
    result = model.transcribe(
        chunk_path,
        task="transcribe",
        language=lang_code,
        fp16=_on_gpu(model),               # fp16 is GPU-only; avoids the CPU warning
        condition_on_previous_text=False,  # stops one bad line from snowballing
        verbose=None,
    )
    segments = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text or _is_hallucination(seg):
            continue
        segments.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": text})
    return segments


# ── Sarvam ────────────────────────────────────────────────────────────────────
def _split_on_pauses(audio, max_ms: int, min_ms: int = 2000) -> list[tuple[int, int]]:
    """
    Cut audio into pieces ≤ max_ms, preferring to cut in the middle of a pause.
    (The old version cut at hard 25 s marks, which could split a word.)
    """
    from pydub.silence import detect_silence

    total = len(audio)
    if total <= max_ms:
        return [(0, total)]
    threshold = audio.dBFS - cfg.SILENCE_REL_DB if audio.dBFS != float("-inf") else -50
    silences = detect_silence(audio, min_silence_len=300, silence_thresh=threshold,
                              seek_step=cfg.SILENCE_SEEK_MS)
    cut_points = [(start + end) // 2 for start, end in silences]

    pieces, start = [], 0
    while total - start > max_ms:
        candidates = [c for c in cut_points if start + min_ms < c <= start + max_ms]
        cut = candidates[-1] if candidates else start + max_ms
        pieces.append((start, cut))
        start = cut
    pieces.append((start, total))
    return pieces


def _send_to_sarvam(piece_path: str, mode: str = "translit", language_code: str = "hi-IN") -> dict:
    """Send one ≤30 s WAV to Sarvam /speech-to-text and return the JSON body."""
    api_key = cfg.sarvam_api_key()
    if not api_key:
        raise HookLensError("SARVAM_API_KEY is not set in environment / .env")

    headers = {"api-subscription-key": api_key}
    data = {
        "model": cfg.SARVAM_STT_MODEL,
        "mode": mode,
        "language_code": language_code,
        "with_timestamps": "true",
    }
    response = None
    for attempt in range(3):
        with open(piece_path, "rb") as f:
            files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
            try:
                response = requests.post(cfg.SARVAM_STT_URL, headers=headers, files=files,
                                         data=data, timeout=120)
            except requests.RequestException as exc:
                if attempt == 2:
                    raise HookLensError("Couldn't reach Sarvam's speech-to-text API.",
                                        hint="Check your internet connection and try again.") from exc
                time.sleep(2 * (attempt + 1))
                continue
        if response.status_code in (429, 503) and attempt < 2:   # rate limit / overloaded
            time.sleep(2 * (attempt + 1))
            continue
        break

    if response is None:
        raise HookLensError("Sarvam did not respond.")
    if response.status_code in (401, 403):
        raise HookLensError("Sarvam rejected the API key.", hint="Check SARVAM_API_KEY in .env / Secrets.")
    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        raise HookLensError(
            f"Sarvam speech-to-text failed (HTTP {response.status_code}).",
            hint="Try again in a minute, or choose English to use Whisper.",
        )
    return response.json()


def _parse_sarvam(body: dict, offset_s: float, speech_start_s: float, speech_end_s: float) -> list[dict]:
    """Sarvam chunk timestamps → absolute segments; falls back to one segment per piece."""
    timestamps = body.get("timestamps") or {}
    phrases = timestamps.get("words") or timestamps.get("chunks") or []
    starts = timestamps.get("start_time_seconds") or []
    ends = timestamps.get("end_time_seconds") or []
    segments = []
    if phrases and len(phrases) == len(starts) == len(ends):
        for phrase, start, end in zip(phrases, starts, ends):
            text = (phrase or "").strip()
            if text:
                segments.append({"start": offset_s + float(start), "end": offset_s + float(end), "text": text})
    if not segments:
        text = (body.get("transcript") or "").strip()
        if text:
            segments.append({"start": speech_start_s, "end": speech_end_s, "text": text})
    return segments


def transcribe_chunk_sarvam(chunk_path: str, language: str = "hinglish", offset_s: float = 0.0) -> list[dict]:
    """
    Sarvam's REST API only accepts ≤30 s audio, so the chunk is split into
    ≤25 s pieces at natural pauses, each piece is sent separately, and the
    timestamped phrases are stitched back together.
    """
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    if not cfg.sarvam_api_key():
        raise HookLensError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    mode = cfg.SARVAM_MODE.get(language, "transcribe")
    language_code = cfg.LANGUAGES.get(language, cfg.LANGUAGES["hinglish"])["sarvam"]
    pieces = _split_on_pauses(audio, cfg.SARVAM_PIECE_SECONDS * 1000)

    segments: list[dict] = []
    for i, (start_ms, end_ms) in enumerate(tqdm(pieces, desc="Sarvam", unit="piece", disable=None)):
        piece = audio[start_ms:end_ms]
        if piece.dBFS == float("-inf"):
            continue  # pure silence — don't spend an API call on it
        # Tighten the fallback segment bounds to where speech actually is.
        threshold = piece.dBFS - cfg.SILENCE_REL_DB
        voiced = detect_nonsilent(piece, min_silence_len=300, silence_thresh=threshold,
                                  seek_step=cfg.SILENCE_SEEK_MS)
        speech_start = offset_s + (start_ms + (voiced[0][0] if voiced else 0)) / 1000
        speech_end = offset_s + (start_ms + (voiced[-1][1] if voiced else len(piece))) / 1000

        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")
        try:
            body = _send_to_sarvam(piece_path, mode=mode, language_code=language_code)
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)
        segments.extend(_parse_sarvam(body, offset_s + start_ms / 1000, speech_start, speech_end))
    return segments


# ── Router (reused) ───────────────────────────────────────────────────────────
def uses_sarvam(language: str) -> bool:
    return language in ("hindi", "hinglish") and bool(cfg.sarvam_api_key())


def transcribe_chunk(chunk_path: str, language: str = "english", offset_s: float = 0.0,
                     whisper_model=None) -> list[dict]:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english          → Whisper (local model)
    - hindi / hinglish → Sarvam (original language, timestamped) when a key is set
    """
    if uses_sarvam(language):
        return transcribe_chunk_sarvam(chunk_path, language=language, offset_s=offset_s)
    segments = transcribe_chunk_whisper(chunk_path, language=language, model=whisper_model)
    for seg in segments:
        seg["start"] += offset_s
        seg["end"] += offset_s
    return segments


# ── Translation ───────────────────────────────────────────────────────────────
def translate_segments(segments: list[dict], language: str) -> list[str]:
    """
    Add segment["text_en"] using deep-translator (Google). Lines are sent in
    newline-joined blocks (≤4500 chars) so a 3-minute episode needs 1–2 requests
    instead of ~50; if the line count comes back wrong, that block falls back to
    line-by-line. Translation is a convenience — failures never stop the analysis.
    Returns a list of warnings.
    """
    if language == "english":
        for seg in segments:
            seg["text_en"] = seg["text"]
        return []
    for seg in segments:
        seg.setdefault("text_en", "")
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        return ["deep-translator is not installed — English translations skipped."]

    translator = GoogleTranslator(source="auto", target="en")
    blocks, current, size = [], [], 0
    for seg in segments:
        line = seg["text"].replace("\n", " ").strip()
        if current and size + len(line) + 1 > 4500:
            blocks.append(current)
            current, size = [], 0
        current.append(seg)
        size += len(line) + 1
    if current:
        blocks.append(current)

    failures = 0
    for block in tqdm(blocks, desc="Translate", unit="block", disable=None):
        lines = [seg["text"].replace("\n", " ").strip() for seg in block]
        translated: list[str] = []
        try:
            out = translator.translate("\n".join(lines)) or ""
            translated = [t.strip() for t in out.split("\n")]
            if len(translated) != len(lines):
                raise ValueError("line count mismatch")
        except Exception:
            translated = []
            for line in lines:
                try:
                    translated.append((translator.translate(line) or "").strip())
                except Exception:
                    translated.append("")
                    failures += 1
        for seg, text_en in zip(block, translated):
            seg["text_en"] = text_en
    if failures:
        return [f"English translation failed for {failures} line(s) (deep-translator)."]
    return []


# ── Clean-up ──────────────────────────────────────────────────────────────────
def _clean_segments(segments: list[dict]) -> list[dict]:
    """Sort, fix inverted/zero-length times, round, and assign sequential ids."""
    cleaned = []
    for seg in sorted(segments, key=lambda s: (s["start"], s["end"])):
        text = " ".join(seg["text"].split())
        if not text:
            continue
        start = max(0.0, float(seg["start"]))
        end = max(start + 0.1, float(seg["end"]))
        cleaned.append({"start": round(start, 2), "end": round(end, 2), "text": text})
    for i, seg in enumerate(cleaned):
        seg["id"] = i
    return cleaned


# ── Public API ────────────────────────────────────────────────────────────────
def transcribe_all(
    chunks: list,
    language: str = "english",
    *,
    chunk_seconds: float = cfg.CHUNK_MINUTES * 60,
    whisper_loader: Callable[[], object] | None = None,
    translate: bool = True,
) -> dict:
    """
    Transcribe every chunk and return:
        {
          "segments": [{"id", "start", "end", "text", "text_en"}],
          "text": "full transcript",
          "language": "hinglish",
          "engine": "sarvam:saaras:v3/translit" | "whisper:base",
          "warnings": [...]
        }
    `whisper_loader` lets the Streamlit app inject its st.cache_resource model
    (and only load Whisper when it's actually needed).
    """
    language = language if language in cfg.LANGUAGES else cfg.DEFAULT_LANGUAGE
    warnings: list[str] = []
    sarvam = uses_sarvam(language)
    if language in ("hindi", "hinglish") and not sarvam:
        warnings.append(
            f"SARVAM_API_KEY not set — used Whisper for {cfg.LANGUAGES[language]['label']}; "
            "accuracy on Indian-language dialogue will be lower."
        )
    model = None if sarvam else (whisper_loader() if whisper_loader else load_model())

    engine = (f"sarvam:{cfg.SARVAM_STT_MODEL}/{cfg.SARVAM_MODE.get(language, 'transcribe')}"
              if sarvam else f"whisper:{cfg.WHISPER_MODEL}")
    print(f"Using {engine} for transcription.")

    segments: list[dict] = []
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        segments.extend(transcribe_chunk(chunk, language=language, offset_s=i * chunk_seconds,
                                         whisper_model=model))
    segments = _clean_segments(segments)
    if translate:
        warnings.extend(translate_segments(segments, language))
    else:
        for seg in segments:
            seg["text_en"] = seg["text"] if language == "english" else ""
    print("Transcription complete.")

    return {
        "segments": segments,
        "text": " ".join(seg["text"] for seg in segments).strip(),
        "language": language,
        "engine": engine,
        "warnings": warnings,
    }
