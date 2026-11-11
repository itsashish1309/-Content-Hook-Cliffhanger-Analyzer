"""
config.py — every tunable number in HookLens lives here.

Why one file: the hook/cliffhanger windows, score weights, risk bands and
pacing thresholds are *product decisions*, not code. A content lead can change
a weight here and see the effect on the next analysis without touching any
analyser. Secrets come from the environment (.env locally, Streamlit Secrets
on Community Cloud) and are never hard-coded.

Import rule: call load_dotenv() BEFORE importing this module (app.py, main.py
and the tests all do), because model names below are read at import time.
API keys are read lazily through functions, so they never go stale.
"""
from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "HookLens"
APP_VERSION = "1.0.0"
APP_TAGLINE = "Episode hook & cliffhanger analyzer for micro-dramas"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"          # writer uploads (auto-cleaned after 24 h)
DOWNLOAD_DIR = DATA_DIR / "downloads"      # yt-dlp audio + normalised 16 kHz WAVs
CHROMA_DIR = DATA_DIR / "chroma_db"        # persistent benchmark vector store
REPORTS_DIR = DATA_DIR / "reports"         # PDFs written by the CLI
FEEDBACK_FILE = DATA_DIR / "feedback.json"
SEED_BENCHMARK_FILE = DATA_DIR / "seed_benchmark.json"

for _folder in (DATA_DIR, UPLOAD_DIR, DOWNLOAD_DIR, CHROMA_DIR, REPORTS_DIR):
    _folder.mkdir(parents=True, exist_ok=True)


# ── Secrets / environment ─────────────────────────────────────────────────────
def env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable at call time; blank values count as unset."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def mistral_api_key() -> str | None:
    return env("MISTRAL_API_KEY")


def sarvam_api_key() -> str | None:
    return env("SARVAM_API_KEY")


# ── Models ────────────────────────────────────────────────────────────────────
MISTRAL_MODEL = env("MISTRAL_MODEL", "mistral-small-latest")
WHISPER_MODEL = env("WHISPER_MODEL", "base")  # "base" fits Streamlit Community Cloud RAM
EMBEDDING_MODEL = env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Temperatures: 0 for anything that produces a score (consistency), higher for
# creative rewrites (variety), in-between for summaries and chat.
TEMPERATURE_SCORING = 0.0
TEMPERATURE_REWRITE = 0.7
TEMPERATURE_SUMMARY = 0.3
LLM_MAX_TOKENS = 2048
LLM_TIMEOUT_S = 90
LLM_MAX_RETRIES = 3          # network / rate-limit retries inside langchain-mistralai
LLM_SEED = 42                # fixed seed for temperature-0 scoring calls

# ── Sarvam (Hindi / Hinglish speech-to-text) ──────────────────────────────────
# Sarvam's current REST endpoint. `mode` controls the output script:
#   transcribe → original script (Devanagari for Hindi)
#   translit   → Roman script (natural written Hinglish)
#   codemix    → English words in Latin, Hindi words in Devanagari
#   translate  → English (what the old /speech-to-text-translate endpoint did)
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_STT_MODEL = env("SARVAM_STT_MODEL", "saaras:v3")
_LEGACY_SARVAM_MODELS = {"saaras:v2", "saaras:v2.5", "saarika:v1", "saarika:v2",
                         "saarika:v2.5", "saarika:flash"}
if SARVAM_STT_MODEL in _LEGACY_SARVAM_MODELS:
    # Older models ignore `mode` and are being retired; an old .env value
    # (the AI Video Assistant used saaras:v2.5) is upgraded transparently.
    SARVAM_STT_MODEL = "saaras:v3"
SARVAM_MODE = {
    "english": "transcribe",
    "hindi": env("SARVAM_HINDI_MODE", "transcribe"),
    "hinglish": env("SARVAM_HINGLISH_MODE", "translit"),
}
SARVAM_PIECE_SECONDS = 25    # REST limit is 30 s; 5 s safety margin (as before)

# ── Languages ─────────────────────────────────────────────────────────────────
LANGUAGES: dict[str, dict[str, str]] = {
    "english": {
        "label": "English",
        "whisper": "en",
        "sarvam": "en-IN",
        "prompt_label": "English",
        "write_in": "natural spoken English",
    },
    "hindi": {
        "label": "Hindi",
        "whisper": "hi",
        "sarvam": "hi-IN",
        "prompt_label": "Hindi (Devanagari script)",
        "write_in": "natural spoken Hindi in Devanagari script",
    },
    "hinglish": {
        "label": "Hinglish",
        "whisper": "hi",
        "sarvam": "hi-IN",
        "prompt_label": "Hinglish (Hindi-English code-mix, usually Roman script)",
        "write_in": ("natural Hinglish in Roman (Latin) script, the way people text — "
                     "e.g. 'Tumne mujhse jhooth kyun bola?' — never Devanagari"),
    },
}
DEFAULT_LANGUAGE = "english"

# ── Analysis windows ──────────────────────────────────────────────────────────
HOOK_WINDOW_S = 10.0         # "will they stay past the first 10 seconds?"
CLIFF_WINDOW_S = 15.0        # "will they unlock the next episode?"
CUT_SEARCH_S = 30.0          # best-cut candidates are searched in the last 30 s

# ── Hook scoring ──────────────────────────────────────────────────────────────
HOOK_DIMENSIONS = {
    "conflict_tension": "Conflict / tension",
    "curiosity_gap": "Curiosity gap",
    "emotional_intensity": "Emotional intensity",
    "stakes_clarity": "Stakes & character clarity",
    "speed": "Speed to first line",
}
HOOK_LLM_DIMENSIONS = ("conflict_tension", "curiosity_gap", "emotional_intensity", "stakes_clarity")
HOOK_WEIGHTS = {
    "conflict_tension": 0.25,
    "curiosity_gap": 0.25,
    "emotional_intensity": 0.20,
    "stakes_clarity": 0.15,
    "speed": 0.15,
}
# "Speed" is measured, not guessed: seconds until the first meaningful line → score.
SPEED_SCORE_STEPS = (
    (0.5, 10.0), (1.5, 9.0), (2.5, 8.0), (3.5, 6.0), (5.0, 4.0), (7.0, 3.0), (float("inf"), 1.0),
)

# ── Cliffhanger scoring ───────────────────────────────────────────────────────
CLIFF_DIMENSIONS = {
    "unresolved_question": "Unresolved question",
    "rising_stakes": "Rising stakes",
    "reveal_shock": "Reveal / shock",
    "emotional_peak_at_cut": "Emotional peak at cut",
    "would_pay_next": "Would I pay to see next?",
}
CLIFF_WEIGHTS = {
    "unresolved_question": 0.25,
    "rising_stakes": 0.20,
    "reveal_shock": 0.20,
    "emotional_peak_at_cut": 0.15,
    "would_pay_next": 0.20,
}

# ── Overall verdict ───────────────────────────────────────────────────────────
# Hook + cliffhanger carry 75 % because they map directly to the two funnel
# metrics that matter for micro-drama revenue: early drop-off and unlocks.
RISK_WEIGHTS = {"hook": 0.40, "cliffhanger": 0.35, "pacing": 0.15, "emotion": 0.10}
RISK_LOW_FROM = 7.0          # weighted score ≥ 7.0 → Low risk
RISK_MEDIUM_FROM = 4.5       # 4.5–6.9 → Medium, below → High
CRITICAL_COMPONENT = 3.5     # weakest-link rule: hook or cliffhanger below this → High
FIX_TARGET_SCORE = 8.0       # assumed score after a good fix (used to estimate impact)

# Colour bands used everywhere (UI cards, PDF): red < 4, amber 4–6.9, green ≥ 7.
SCORE_RED_BELOW = 4.0
SCORE_GREEN_FROM = 7.0


def score_band(score: float | None) -> str:
    """Map a 0–10 score to 'red' | 'amber' | 'green' | 'none'."""
    if score is None:
        return "none"
    if score < SCORE_RED_BELOW:
        return "red"
    if score < SCORE_GREEN_FROM:
        return "amber"
    return "green"


RISK_BAND_COLOUR = {"Low": "green", "Medium": "amber", "High": "red", "Unknown": "none"}

# ── Pacing (pure numbers: pydub + numpy, no LLM) ──────────────────────────────
DENSITY_BIN_S = 10.0         # dialogue density is reported per 10-second window
SILENCE_MIN_MS = 700         # a pause must last ≥ 0.7 s to count as silence
SILENCE_REL_DB = 16.0        # silence = quieter than (average loudness − 16 dB)
SILENCE_SEEK_MS = 10         # detect_silence step; 10 ms is ~10x faster than 1 ms
SLOW_WPS = 1.0               # a window with < 1 word/s of dialogue is "slow"…
SLOW_RELATIVE = 0.5          # …or below 50 % of this episode's median density
LONG_GAP_S = 3.0             # > 3 s without dialogue gets flagged
TARGET_WPS = 2.2             # healthy overall dialogue density for a micro-drama
MIN_BIN_FRACTION = 0.5       # ignore a trailing window shorter than half a window

# ── Emotion timeline ──────────────────────────────────────────────────────────
EMOTIONS = ("anger", "shock", "sadness", "romance", "suspense", "calm")
EMOTION_COLORS = {
    "anger": "#ef4444",
    "shock": "#f59e0b",
    "sadness": "#3b82f6",
    "romance": "#ec4899",
    "suspense": "#9f67ff",
    "calm": "#64748b",
}
EMOTION_SYNONYMS = {
    "angry": "anger", "rage": "anger", "fury": "anger", "frustration": "anger",
    "surprise": "shock", "surprised": "shock", "shocked": "shock", "disbelief": "shock",
    "sad": "sadness", "grief": "sadness", "heartbreak": "sadness", "hurt": "sadness",
    "love": "romance", "romantic": "romance", "longing": "romance", "affection": "romance",
    "fear": "suspense", "tension": "suspense", "tense": "suspense", "mystery": "suspense",
    "dread": "suspense", "anxiety": "suspense", "threat": "suspense",
    "neutral": "calm", "joy": "calm", "happy": "calm", "happiness": "calm", "relief": "calm",
}
EMOTION_BATCH_SIZE = 25      # segments per LLM call
PEAK_MIN = 7.0               # a peak must be at least this intense
DIP_MAX = 3.0                # a dip must be at most this intense
FLAT_MAX = 3.0               # "flat zone" = consecutive lines at or below this…
FLAT_MIN_S = 15.0            # …lasting at least 15 s
EMOTION_ARC_WEIGHTS = {"mean": 0.6, "ending": 0.4}   # overall energy + "does it end hot?"

# ── Media ─────────────────────────────────────────────────────────────────────
ALLOWED_UPLOAD_TYPES = ("mp4", "mov", "mkv", "mp3", "wav", "m4a")
VIDEO_EXTENSIONS = {"mp4", "mov", "mkv"}
MAX_EPISODE_SECONDS = int(env("MAX_EPISODE_SECONDS", "900") or 900)  # 15 min guard rail
MIN_EPISODE_SECONDS = 5.0
CHUNK_MINUTES = 10           # reused from AI Video Assistant (episodes are 1 chunk)
UPLOAD_TTL_HOURS = 24

# ── RAG ───────────────────────────────────────────────────────────────────────
BENCHMARK_COLLECTION = "hooklens_benchmark"
SIMILAR_K = 3
CHAT_WINDOW_S = 20.0         # chat chunks = ~20 s of consecutive lines (timestamped)
CHAT_TOP_K = 4
