"""
app.py — HookLens Streamlit UI.

A Kuku TV writer pastes a YouTube link or uploads an episode and gets, BEFORE
release: a Hook score (first 10 s), a Cliffhanger score (last 15 s), pacing
and emotion timelines, a Retention Risk verdict with the top 3 fixes, rewrite
suggestions, a "similar past hooks" benchmark, chat and a PDF scorecard.

UI rules followed here:
  • The visual theme from AI Video Assistant (dark, purple/cyan, Syne +
    JetBrains Mono), polished with score cards, bars and an emotion strip.
  • SECURITY: every piece of transcript / LLM / user text is passed through
    esc() (html.escape) before it touches st.markdown(unsafe_allow_html=True).
  • st.cache_resource for the Whisper model and embeddings; st.session_state
    for results, so reruns (clicks, feedback, chat) never re-process the episode.
  • Friendly errors: HookLensError messages are shown as-is; anything else is a
    generic message with the traceback tucked into an expander.
"""
from __future__ import annotations

import html
import json
import os
import re
import time
import traceback
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # MUST run before core imports — config reads env at import time

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HookLens — Hook & Cliffhanger Analyzer",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _bridge_secrets_to_env() -> None:
    """Streamlit Cloud Secrets → os.environ, so core/ can read keys with os.getenv."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float, bool)):
                os.environ.setdefault(key, str(value))
    except Exception:
        pass  # no secrets.toml locally — .env is used instead


_bridge_secrets_to_env()

import config as cfg  # noqa: E402
from core.common import (  # noqa: E402
    fmt_range, fmt_ts, transcript_to_srt, transcript_to_txt, truncate,
)
from core.pipeline import STEPS, build_analysis_brief, run_pipeline  # noqa: E402
from utils.audio_processor import cleanup_old_files, is_youtube_url, save_uploaded_file  # noqa: E402
from utils.errors import HookLensError  # noqa: E402
from utils.feedback import feedback_stats, remove_feedback, save_feedback  # noqa: E402


# ─── Cached resources (loaded once per server process) ────────────────────────
@st.cache_resource(show_spinner="Loading the Whisper model (first run only)…")
def get_whisper_model():
    from core.transcriber import load_model

    return load_model(cfg.WHISPER_MODEL)


@st.cache_resource(show_spinner="Loading the embedding model (first run only)…")
def get_embedding_model():
    from core.rag_engine import get_embeddings

    return get_embeddings()


@st.cache_resource(show_spinner=False)
def startup_cleanup() -> int:
    """Once per process: delete uploads / downloads older than 24 h."""
    return cleanup_old_files(cfg.UPLOAD_DIR) + cleanup_old_files(cfg.DOWNLOAD_DIR)


startup_cleanup()

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@500;600;700;800&family=JetBrains+Mono:wght@300;400;500;700&family=Noto+Sans:wght@400;500;600&family=Noto+Sans+Devanagari:wght@400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --surface-3: #222232;
    --border: #2a2a3a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #8a8ab0;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --content-font: 'Noto Sans', 'Noto Sans Devanagari', system-ui, sans-serif;
}

/* ── Global ── */
html, body, [class*="css"] { font-family: 'JetBrains Mono', 'Noto Sans Devanagari', monospace; }
.stApp { background: var(--bg) !important; color: var(--text); }
.stApp::before {
    content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124, 58, 237, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 58, 237, 0.035) 1px, transparent 1px);
    background-size: 40px 40px; pointer-events: none; z-index: 0;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 2rem !important; max-width: 1400px; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
h1, h2, h3, h4, h5, h6 { font-family: 'Syne', sans-serif !important; color: var(--text) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text); }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.2rem 0 !important; }

/* ── Hero ── */
.hero-title {
    font-family: 'Syne', sans-serif; font-size: clamp(2rem, 5vw, 3.3rem); font-weight: 800; line-height: 1.05; margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub { font-size: 0.78rem; color: var(--text-muted); letter-spacing: 0.2em; text-transform: uppercase; margin-top: 0.5rem; }
.brand-title { font-family: 'Syne', sans-serif; font-size: 1.75rem; font-weight: 800; line-height: 1;
    background: linear-gradient(135deg, #fff 0%, var(--accent-glow) 55%, var(--accent-2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.brand-sub { font-size: 0.66rem; color: var(--text-muted); letter-spacing: 0.18em; text-transform: uppercase; margin: .35rem 0 .9rem; }

/* ── Cards ── */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1.2rem 1.35rem;
    margin-bottom: 1rem; position: relative; overflow: hidden; transition: border-color 0.2s; }
.card:hover { border-color: rgba(124, 58, 237, .6); }
.card::before { content: ''; position: absolute; top: 0; left: 0; width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2)); }
.card-title { font-family: 'Syne', sans-serif; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.7rem; }
.card-content { font-family: var(--content-font); font-size: 0.9rem; line-height: 1.7; color: var(--text); }
.muted { color: var(--text-muted); font-size: 0.8rem; line-height: 1.6; }
.section-title { font-family: 'Syne', sans-serif; font-size: 1.08rem; font-weight: 700; color: var(--text);
    margin: 1.5rem 0 0.75rem; display: flex; align-items: baseline; gap: .6rem; flex-wrap: wrap; }
.section-title small { font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; color: var(--text-muted);
    font-weight: 400; letter-spacing: .06em; text-transform: uppercase; }

/* ── Badges ── */
.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.64rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; white-space: nowrap; }
.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.35); }
.badge-cyan   { background: rgba(6,182,212,0.15); color: var(--accent-2); border: 1px solid rgba(6,182,212,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success); border: 1px solid rgba(16,185,129,0.3); }
.badge-amber  { background: rgba(245,158,11,0.15); color: var(--warning); border: 1px solid rgba(245,158,11,0.3); }
.badge-red    { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.35); }
.badge-muted  { background: var(--surface-2); color: var(--text-muted); border: 1px solid var(--border); text-transform: none; letter-spacing: .02em; }
.badge-row { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .8rem; }

/* ── Episode header ── */
.episode-title { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 700; color: var(--text); line-height: 1.25; }
.episode-logline { font-family: var(--content-font); color: var(--text-muted); font-size: 0.9rem; margin-top: .35rem; }

/* ── Score cards (colour bands: red < 4, amber 4–6.9, green ≥ 7) ── */
.band-green { --band: var(--success); --band-soft: rgba(16,185,129,.16); }
.band-amber { --band: var(--warning); --band-soft: rgba(245,158,11,.16); }
.band-red   { --band: var(--danger);  --band-soft: rgba(239,68,68,.16); }
.band-none  { --band: var(--text-muted); --band-soft: rgba(138,138,176,.10); }
.score-card { background: linear-gradient(160deg, var(--band-soft) 0%, var(--surface) 55%); border: 1px solid var(--border);
    border-top: 3px solid var(--band); border-radius: 16px; padding: 1.15rem 1.35rem 1.25rem; margin-bottom: 1rem;
    position: relative; overflow: hidden; }
.score-label { font-family: 'Syne', sans-serif; font-size: .7rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--text-muted); }
.score-value { font-family: 'Syne', sans-serif; font-size: 3.1rem; font-weight: 800; color: var(--band); line-height: 1.05; margin-top: .35rem; }
.score-value span { font-size: 1rem; color: var(--text-muted); font-weight: 600; margin-left: .2rem; }
.score-value.word { font-size: 2.5rem; letter-spacing: .02em; }
.score-sub { font-size: .74rem; color: var(--text-muted); margin-top: .3rem; line-height: 1.5; }
.score-sub b { color: var(--band); font-weight: 700; }
.score-meter { height: 6px; background: var(--surface-2); border-radius: 3px; margin-top: .85rem; overflow: hidden; }
.score-meter > div { height: 100%; background: var(--band); border-radius: 3px; }

/* ── Sub-score bars ── */
.sub-row { display: grid; grid-template-columns: minmax(150px, 1.25fr) 2fr 46px; align-items: center; gap: .8rem;
    padding: .45rem 0; border-bottom: 1px dashed rgba(42,42,58,.8); }
.sub-row:last-child { border-bottom: none; }
.sub-label { font-size: .76rem; color: var(--text); }
.sub-w { color: var(--text-muted); font-size: .64rem; margin-left: .35rem; }
.sub-track { height: 8px; background: var(--surface-2); border-radius: 4px; overflow: hidden; }
.sub-fill { height: 100%; border-radius: 4px; }
.fill-green { background: linear-gradient(90deg, #059669, var(--success)); }
.fill-amber { background: linear-gradient(90deg, #d97706, var(--warning)); }
.fill-red   { background: linear-gradient(90deg, #b91c1c, var(--danger)); }
.fill-none  { background: var(--border); }
.sub-val { text-align: right; font-family: 'Syne', sans-serif; font-weight: 700; font-size: .95rem; }

/* ── Lists inside cards ── */
.pill-list { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .4rem; }
.pill { font-family: var(--content-font); font-size: .74rem; padding: .18rem .55rem; border-radius: 999px; }
.pill-good { background: rgba(16,185,129,.12); color: #6ee7b7; border: 1px solid rgba(16,185,129,.25); }
.pill-bad  { background: rgba(239,68,68,.10); color: #fca5a5; border: 1px solid rgba(239,68,68,.25); }
.question { font-family: var(--content-font); font-size: .9rem; color: var(--accent-2); margin-top: .6rem; }

/* ── Fix cards ── */
.fix-card { display: flex; gap: 1rem; background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 1rem 1.15rem; margin-bottom: .35rem; }
.fix-num { width: 2.1rem; height: 2.1rem; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, var(--accent), var(--accent-2)); font-family: 'Syne', sans-serif; font-weight: 800; color: #fff; }
.fix-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: .98rem; color: var(--text); display: flex; gap: .5rem; flex-wrap: wrap; align-items: center; }
.fix-action { font-family: var(--content-font); font-size: .86rem; color: var(--text); line-height: 1.6; margin-top: .3rem; }

/* ── Alternatives (rewrites / endings) ── */
.alt-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.1rem; min-height: 190px; margin-bottom: .35rem; }
.alt-card:hover { border-color: rgba(6,182,212,.5); }
.alt-top { display: flex; gap: .4rem; flex-wrap: wrap; }
.alt-line { font-family: var(--content-font); font-size: 1rem; line-height: 1.55; color: #fff; margin: .7rem 0 .45rem; }
.alt-en { font-family: var(--content-font); font-size: .8rem; color: var(--accent-2); line-height: 1.5; }
.alt-why { font-size: .72rem; color: var(--text-muted); margin-top: .55rem; line-height: 1.5; }

/* ── Transcript ── */
.transcript-wrap { max-height: 580px; overflow-y: auto; border: 1px solid var(--border); border-radius: 12px; background: var(--surface); }
.seg { display: grid; grid-template-columns: 132px 1fr 118px; gap: .75rem; padding: .55rem .8rem;
    border-left: 3px solid transparent; border-bottom: 1px solid rgba(42,42,58,.6); }
.seg:last-child { border-bottom: none; }
.seg-hook  { border-left-color: var(--accent-2); background: rgba(6,182,212,.05); }
.seg-cliff { border-left-color: var(--accent-glow); background: rgba(124,58,237,.07); }
.seg-time { font-size: .7rem; color: var(--text-muted); padding-top: .2rem; }
.seg-text { font-family: var(--content-font); font-size: .9rem; line-height: 1.55; color: var(--text); }
.seg-en { font-family: var(--content-font); font-size: .78rem; color: var(--text-muted); margin-top: .15rem; }
.seg-emo { text-align: right; }
/* Narrow panels (hook/cliff windows, peaks/dips): time + chip on one row, text full width below */
.compact .seg { grid-template-columns: 1fr auto; row-gap: .25rem; }
.compact .seg-time { grid-column: 1; grid-row: 1; padding-top: 0; }
.compact .seg-emo { grid-column: 2; grid-row: 1; }
.compact .seg-main { grid-column: 1 / -1; grid-row: 2; }
.emo-chip { display: inline-block; padding: .12rem .5rem; border-radius: 999px; font-size: .64rem; border: 1px solid var(--emo);
    color: var(--emo); background: color-mix(in srgb, var(--emo) 14%, transparent); white-space: nowrap; }
.legend-row { display: flex; gap: 1rem; flex-wrap: wrap; font-size: .7rem; color: var(--text-muted); margin: .2rem 0 .6rem; }
.legend-row i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: .35rem; vertical-align: -1px; }

/* ── Emotion strip ── */
.strip { position: relative; height: 56px; background: var(--surface-2); border-radius: 8px; margin: .5rem 0 .25rem; overflow: hidden; }
.strip-seg { position: absolute; top: 19px; height: 18px; border-radius: 3px; }
.strip-window { position: absolute; top: 0; bottom: 0; border: 1px dashed; border-radius: 6px; }
.strip-window.hook  { border-color: rgba(6,182,212,.75); background: rgba(6,182,212,.07); }
.strip-window.cliff { border-color: rgba(159,103,255,.85); background: rgba(124,58,237,.09); }
.strip-mark { position: absolute; transform: translateX(-50%); font-size: .72rem; line-height: 1; }
.strip-mark.peak { top: 3px; color: var(--danger); }
.strip-mark.dip  { bottom: 3px; color: var(--accent-2); }
.strip-ticks { position: relative; height: 16px; font-size: .62rem; color: var(--text-muted); }
.strip-ticks span { position: absolute; transform: translateX(-50%); }

/* ── Benchmark (RAG) ── */
.sim-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: .85rem 1rem; margin-bottom: .6rem; }
.sim-head { display: flex; align-items: center; gap: .65rem; }
.sim-score { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.25rem; color: var(--band); min-width: 2.6rem; }
.sim-title { font-weight: 600; font-size: .8rem; flex: 1; color: var(--text); }
.sim-pct { font-size: .7rem; color: var(--accent-2); white-space: nowrap; }
.sim-text { font-family: var(--content-font); font-size: .86rem; color: var(--text); margin-top: .45rem; line-height: 1.55; }
.sim-en { font-family: var(--content-font); font-size: .76rem; color: var(--text-muted); margin-top: .2rem; }
.compare { font-family: var(--content-font); font-size: .9rem; color: var(--text); margin-bottom: .8rem; }
.compare b { font-family: 'Syne', sans-serif; }

/* ── Status bars (reused from AI Video Assistant) ── */
.status-bar { display: flex; align-items: center; gap: 0.7rem; padding: 0.55rem 0.85rem; background: var(--surface-2);
    border-radius: 8px; margin: 0.35rem 0; border: 1px solid var(--border); font-size: 0.76rem; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot-active  { background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.5s infinite; }
.dot-done    { background: var(--success); }
.dot-error   { background: var(--warning); box-shadow: 0 0 6px var(--warning); }
.dot-pending { background: var(--border); }
.status-time { margin-left: auto; color: var(--text-muted); font-size: .66rem; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
.key-row { display: flex; justify-content: space-between; font-size: .72rem; padding: .25rem 0; color: var(--text-muted); }
.key-ok { color: var(--success); } .key-miss { color: var(--warning); }

/* ── Chat (reused) ── */
.chat-container { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem;
    max-height: 460px; overflow-y: auto; margin-bottom: 1rem; }
.chat-msg { margin-bottom: 1rem; display: flex; flex-direction: column; gap: 0.2rem; }
.chat-label { font-size: 0.64rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; }
.chat-bubble { display: inline-block; padding: 0.65rem 1rem; border-radius: 10px; font-family: var(--content-font);
    font-size: 0.88rem; line-height: 1.6; max-width: 90%; }
.user-label  { color: var(--accent-glow); }
.bot-label   { color: var(--accent-2); }
.user-bubble { background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.25); align-self: flex-end; }
.bot-bubble  { background: rgba(6,182,212,0.08); border: 1px solid rgba(6,182,212,0.2); align-self: flex-start; }

/* ── Error card ── */
.error-card { background: rgba(239,68,68,.08); border: 1px solid rgba(239,68,68,.35); border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1rem; }
.error-title { font-family: 'Syne', sans-serif; font-weight: 700; color: #fca5a5; }
.error-hint { font-size: .82rem; color: var(--text); margin-top: .35rem; font-family: var(--content-font); }

/* ── Empty state ── */
.empty-wrap { text-align: center; padding: 3.5rem 1rem 1rem; }
.empty-steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin-top: 2rem; text-align: left; }
.empty-step { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1.1rem; }
.empty-step b { font-family: 'Syne', sans-serif; font-size: .95rem; display: block; margin: .35rem 0; color: var(--text); }
.empty-step span { font-size: .78rem; color: var(--text-muted); line-height: 1.6; }

/* ── Native widgets ── */
.stTextInput > div > div > input, .stSelectbox > div > div { background: var(--surface-2) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important; color: var(--text) !important; }
.stTextInput > div > div > input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
    border-radius: 8px !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    letter-spacing: 0.04em !important; transition: all 0.2s !important; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important; color: white !important; border: none !important;
    text-transform: uppercase !important; padding: 0.6rem 1.4rem !important; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 25px rgba(124,58,237,0.35) !important; }
.stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { font-family: 'Syne', sans-serif; font-weight: 700; padding: .55rem .9rem; border-radius: 8px 8px 0 0; }
.stTabs [aria-selected="true"] { background: rgba(124,58,237,.14) !important; }
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: .75rem 1rem; }
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif; }
[data-testid="stMetricLabel"] p { color: var(--text-muted) !important; font-size: .7rem !important; text-transform: uppercase; letter-spacing: .08em; }
.stProgress > div > div > div { background: var(--accent) !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
@media (max-width: 900px) { .empty-steps { grid-template-columns: 1fr; } .seg { grid-template-columns: 96px 1fr; } .seg-emo { display: none; } }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ───────────────────────────────────────────────────────
for _key, _factory in {
    "result": lambda: None,
    "chat_chain": lambda: None,
    "chat_history": list,
    "pipeline_steps": dict,
    "step_times": dict,
    "error": lambda: None,
    "pdf_cache": dict,
    "toast": lambda: None,
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _factory()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def esc(value) -> str:
    """Escape ANY transcript / LLM / user text before it goes into unsafe HTML."""
    return html.escape("" if value is None else str(value), quote=True)


def html_block(markup: str) -> None:
    """
    Render trusted HTML templates (with escaped values inside).
    Collapsed to a single line and wrapped in a <div> so Markdown never turns
    indentation into code blocks or interprets *, _, $ inside our content.
    """
    one_line = " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(f'<div class="hl">{one_line}</div>', unsafe_allow_html=True)


def fmt_score(score) -> str:
    return "—" if score is None else f"{score:.1f}"


def band_word(band: str) -> str:
    return {"green": "Strong", "amber": "Needs work", "red": "Weak", "none": "Unavailable"}[band]


def section(title: str, note: str = "") -> None:
    small = f"<small>{esc(note)}</small>" if note else ""
    html_block(f'<div class="section-title">{esc(title)} {small}</div>')


def file_slug(result: dict) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", result.get("title", "")).strip("_")[:40]
    return slug or result.get("episode_id", "episode")


# ─── Pipeline status (reused step-status bars, now live) ──────────────────────
def dot_class(state: str) -> str:
    return {"active": "dot-active", "done": "dot-done", "error": "dot-error"}.get(state, "dot-pending")


def render_status(placeholder) -> None:
    steps = st.session_state.pipeline_steps
    if not steps:
        placeholder.empty()
        return
    rows = []
    for key, icon, label in STEPS:
        state = steps.get(key, "pending")
        seconds = st.session_state.step_times.get(key)
        timing = f'<span class="status-time">{seconds:.1f}s</span>' if seconds is not None and state != "active" else ""
        rows.append(f'<div class="status-bar"><div class="status-dot {dot_class(state)}"></div>'
                    f'<span>{icon} {esc(label)}</span>{timing}</div>')
    placeholder.markdown('<div class="hl"><span class="badge badge-green">Pipeline status</span>'
                         + "".join(rows) + "</div>", unsafe_allow_html=True)


def make_step_callback(placeholder):
    started: dict[str, float] = {}

    def on_step(key: str, state: str) -> None:
        st.session_state.pipeline_steps[key] = state
        if state == "active":
            started[key] = time.perf_counter()
        elif key in started:
            st.session_state.step_times[key] = time.perf_counter() - started[key]
        render_status(placeholder)

    return on_step


# ─── Feedback (thumbs on every suggestion → data/feedback.json) ───────────────
def _on_feedback(key, episode_id, title, language, suggestion_id, suggestion_type, text) -> None:
    value = st.session_state.get(key)
    if value is None:  # thumb un-clicked
        remove_feedback(episode_id, suggestion_id)
        return
    save_feedback(episode_id=episode_id, episode_title=title, suggestion_id=suggestion_id,
                  suggestion_type=suggestion_type, suggestion_text=text,
                  rating=1 if value == 1 else -1, language=language)
    st.session_state.toast = "Thanks — feedback saved 👍" if value == 1 else "Thanks — noted 👎"


def feedback_widget(result: dict, suggestion_id: str, suggestion_type: str, text: str) -> None:
    key = f"fb::{result['episode_id']}::{suggestion_id}"
    st.feedback("thumbs", key=key, on_change=_on_feedback,
                args=(key, result["episode_id"], result["title"], result["language"],
                      suggestion_id, suggestion_type, text))


# ─── HTML components ──────────────────────────────────────────────────────────
def score_card_html(label: str, score, question: str) -> str:
    band = cfg.score_band(score)
    pct = 0 if score is None else score * 10
    return (f'<div class="score-card band-{band}"><div class="score-label">{esc(label)}</div>'
            f'<div class="score-value">{fmt_score(score)}<span>/10</span></div>'
            f'<div class="score-sub">{esc(question)} · <b>{band_word(band)}</b></div>'
            f'<div class="score-meter"><div style="width:{pct:.0f}%"></div></div></div>')


def risk_card_html(verdict: dict) -> str:
    risk = verdict.get("risk", "Unknown")
    band = cfg.RISK_BAND_COLOUR.get(risk, "none")
    overall = verdict.get("overall")
    pct = 0 if overall is None else overall * 10
    return (f'<div class="score-card band-{band}"><div class="score-label">⚖️ Retention risk</div>'
            f'<div class="score-value word">{esc(risk.upper())}</div>'
            f'<div class="score-sub">Weighted score <b>{fmt_score(overall)}/10</b> · hook, cliffhanger, '
            f'pacing &amp; emotion</div>'
            f'<div class="score-meter"><div style="width:{pct:.0f}%"></div></div></div>')


def subscores_html(title: str, sub_scores: dict, labels: dict, weights: dict) -> str:
    rows = []
    for key, label in labels.items():
        value = sub_scores.get(key)
        band = cfg.score_band(value)
        pct = 0 if value is None else value * 10
        rows.append(f'<div class="sub-row"><div class="sub-label">{esc(label)}'
                    f'<span class="sub-w">×{weights.get(key, 0):.2f}</span></div>'
                    f'<div class="sub-track"><div class="sub-fill fill-{band}" style="width:{pct:.0f}%"></div></div>'
                    f'<div class="sub-val">{fmt_score(value)}</div></div>')
    return f'<div class="card"><div class="card-title">{esc(title)}</div>{"".join(rows)}</div>'


def diagnosis_html(analysis: dict, question_label: str) -> str:
    good = "".join(f'<span class="pill pill-good">{esc(s)}</span>' for s in analysis.get("strengths", []))
    bad = "".join(f'<span class="pill pill-bad">{esc(s)}</span>' for s in analysis.get("weaknesses", []))
    question = analysis.get("open_question")
    question_html = f'<div class="question">❓ {esc(question_label)}: {esc(question)}</div>' if question else ""
    pills = ""
    if good or bad:
        pills = f'<div class="pill-list">{good}{bad}</div>'
    return (f'<div class="card"><div class="card-title">🩺 Diagnosis</div>'
            f'<div class="card-content">{esc(analysis.get("reason", ""))}</div>{question_html}{pills}</div>')


def segment_row_html(seg: dict, point: dict | None = None, extra_class: str = "") -> str:
    text_en = (seg.get("text_en") or "").strip()
    en_html = (f'<div class="seg-en">{esc(text_en)}</div>'
               if text_en and text_en.lower() != seg["text"].strip().lower() else "")
    chip = ""
    if point and point.get("intensity") is not None:
        colour = cfg.EMOTION_COLORS.get(point["emotion"], "#64748b")
        chip = (f'<span class="emo-chip" style="--emo:{colour}">{esc(point["emotion"])} · '
                f'{point["intensity"]:.0f}</span>')
    return (f'<div class="seg {extra_class}"><div class="seg-time">{esc(fmt_range(seg["start"], seg["end"]))}</div>'
            f'<div class="seg-main"><div class="seg-text">{esc(seg["text"])}</div>{en_html}</div>'
            f'<div class="seg-emo">{chip}</div></div>')


def window_html(title: str, segments: list[dict], points: dict, css: str, footer: str = "") -> str:
    rows = "".join(segment_row_html(s, points.get(s["id"]), css) for s in segments)
    rows = rows or '<div class="muted" style="padding:.8rem">No dialogue in this window.</div>'
    foot = f'<div class="muted" style="margin-top:.7rem">{esc(footer)}</div>' if footer else ""
    return (f'<div class="card"><div class="card-title">{esc(title)}</div>'
            f'<div class="transcript-wrap compact" style="max-height:360px">{rows}</div>{foot}</div>')


def alternative_html(index: int, label: str, text: str, english: str, technique: str, why: str) -> str:
    technique_html = f'<span class="badge badge-cyan">{esc(technique)}</span>' if technique else ""
    english_html = (f'<div class="alt-en">EN · {esc(english)}</div>'
                    if english and english.strip() != text.strip() else "")
    why_html = f'<div class="alt-why">💡 {esc(why)}</div>' if why else ""
    return (f'<div class="alt-card"><div class="alt-top"><span class="badge badge-purple">{esc(label)} {index + 1}</span>'
            f'{technique_html}</div><div class="alt-line">“{esc(text)}”</div>{english_html}{why_html}</div>')


def fix_html(index: int, fix: dict) -> str:
    stamp = fix.get("timestamp")
    stamp_html = f'<span class="badge badge-muted">⏱ {esc(fmt_ts(stamp))}</span>' if stamp is not None else ""
    return (f'<div class="fix-card"><div class="fix-num">{index + 1}</div><div>'
            f'<div class="fix-title">{esc(fix.get("title", ""))}'
            f'<span class="badge badge-cyan">+{fix.get("impact", 0):.2f} pts</span>'
            f'<span class="badge badge-muted">{esc(fix.get("component", ""))}</span>{stamp_html}</div>'
            f'<div class="fix-action">{esc(fix.get("action", ""))}</div></div></div>')


def emotion_strip_html(result: dict) -> str:
    emotion, duration = result["emotion"], max(float(result["duration"]), 0.1)

    def pct(t: float) -> float:
        return max(0.0, min(100.0, 100.0 * float(t) / duration))

    parts = [
        f'<div class="strip-window hook" style="left:0%;width:{pct(cfg.HOOK_WINDOW_S):.2f}%"></div>',
        f'<div class="strip-window cliff" style="left:{pct(duration - cfg.CLIFF_WINDOW_S):.2f}%;'
        f'width:{100 - pct(duration - cfg.CLIFF_WINDOW_S):.2f}%"></div>',
    ]
    for p in emotion.get("points", []):
        if p.get("intensity") is None:
            continue
        left = pct(p["start"])
        width = max(0.4, min(100.0 - left, pct(p["end"]) - left))
        colour = cfg.EMOTION_COLORS.get(p["emotion"], "#64748b")
        opacity = 0.25 + 0.75 * float(p["intensity"]) / 10.0
        tip = esc(f"{fmt_ts(p['start'])} · {p['emotion']} {p['intensity']:.0f}/10 — {truncate(p['text'], 90)}")
        parts.append(f'<div class="strip-seg" title="{tip}" style="left:{left:.2f}%;width:{width:.2f}%;'
                     f'background:{colour};opacity:{opacity:.2f}"></div>')
    for peak in emotion.get("peaks", []):
        tip = esc(f"Peak {fmt_ts(peak['time'])} · {peak['emotion']} {peak['intensity']:.0f}/10")
        parts.append(f'<div class="strip-mark peak" title="{tip}" style="left:{pct(peak["time"]):.2f}%">▲</div>')
    for dip in emotion.get("dips", []):
        tip = esc(f"Dip {fmt_ts(dip['time'])} · {dip['emotion']} {dip['intensity']:.0f}/10")
        parts.append(f'<div class="strip-mark dip" title="{tip}" style="left:{pct(dip["time"]):.2f}%">▼</div>')

    step = 30 if duration > 90 else 15
    ticks = "".join(f'<span style="left:{pct(t):.2f}%">{fmt_ts(t)}</span>'
                    for t in range(0, int(duration) + 1, step) if 0 < t < duration - 3)
    legend = "".join(f'<span><i style="background:{colour}"></i>{esc(name)}</span>'
                     for name, colour in cfg.EMOTION_COLORS.items())
    return (f'<div class="card"><div class="card-title">Emotion strip · ▲ peaks · ▼ dips · dashed = hook / '
            f'cliffhanger windows</div><div class="strip">{"".join(parts)}</div>'
            f'<div class="strip-ticks">{ticks}</div><div class="legend-row">{legend}</div></div>')


# ─── Shared renderers ─────────────────────────────────────────────────────────
def render_player(result: dict) -> None:
    source = result.get("source") or {}
    try:
        if source.get("type") == "youtube" and source.get("url"):
            st.video(source["url"])
            return
        path = source.get("video_path")
        if path and Path(path).exists():
            if source.get("is_video"):
                st.video(path)
                st.caption("If the preview doesn't play, your browser may not support this container "
                           "(e.g. .mkv). The analysis is unaffected.")
            else:
                st.audio(path)
            return
    except Exception:
        pass
    html_block('<div class="card"><div class="card-title">🎬 Preview</div>'
               '<div class="muted">Preview unavailable — the uploaded file was cleaned up. '
               'Results are still available.</div></div>')


def render_similar(result: dict, kind: str) -> None:
    bench = result.get("benchmark") or {}
    items = bench.get("similar_hooks" if kind == "hook" else "similar_cliffhangers") or []
    stats = bench.get("hook_stats" if kind == "hook" else "cliff_stats") or {}
    current = result["hook" if kind == "hook" else "cliffhanger"].get("score")
    noun = "hooks" if kind == "hook" else "cliffhangers"
    section(f"📚 Benchmark — most similar past {noun}", "ChromaDB · sentence-transformers")
    if bench.get("error") and not items:
        html_block(f'<div class="card"><div class="muted">Benchmark unavailable: {esc(bench["error"])}</div></div>')
        return
    if not items:
        html_block('<div class="card"><div class="muted">No similar examples yet — they appear as more '
                   'episodes are analysed.</div></div>')
        return
    scores = [i["score"] for i in items if isinstance(i.get("score"), (int, float))]
    if scores and current is not None:
        avg = sum(scores) / len(scores)
        delta = current - avg
        best = max(items, key=lambda i: i.get("score") or 0)
        word = "above" if delta >= 0 else "below"
        html_block(f'<div class="compare">This {kind} scores <b>{current:.1f}</b> vs <b>{avg:.1f}</b> for the '
                   f'{len(scores)} most similar — <b>{abs(delta):.1f} {word}</b>. Best match: '
                   f'“{esc(truncate(best["text"], 70))}” ({best["score"]:.1f}).</div>')
    for item in items:
        band = cfg.score_band(item.get("score"))
        source_badge = ('<span class="badge badge-muted">Seed · synthetic</span>' if item.get("source") == "seed"
                        else '<span class="badge badge-green">Analysed</span>')
        en = item.get("text_en") or ""
        en_html = (f'<div class="sim-en">EN · {esc(en)}</div>'
                   if en and en.strip().lower() != item["text"].strip().lower() else "")
        html_block(f'<div class="sim-card band-{band}"><div class="sim-head">'
                   f'<span class="sim-score">{fmt_score(item.get("score"))}</span>'
                   f'<span class="sim-title">{esc(item.get("title", ""))}</span>{source_badge}'
                   f'<span class="sim-pct">{item.get("similarity", 0):.0%} similar</span></div>'
                   f'<div class="sim-text">{esc(item["text"])}</div>{en_html}</div>')
    if stats:
        mean = stats.get("mean_score")
        st.caption(f"Library: {stats.get('count', 0)} {noun} ({stats.get('analysed', 0)} analysed, "
                   f"{stats.get('seed', 0)} synthetic seeds) · average {mean if mean is not None else '—'}/10")


# ─── Tabs ─────────────────────────────────────────────────────────────────────
def render_overview(result: dict) -> None:
    hook, cliff, verdict = result["hook"], result["cliffhanger"], result["verdict"]
    pacing, emotion = result["pacing"], result["emotion"]

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        html_block(score_card_html("🎣 Hook · first 10 s", hook.get("score"),
                                   "Will viewers stay past the first 10 seconds?"))
    with c2:
        html_block(score_card_html("🧗 Cliffhanger · last 15 s", cliff.get("score"),
                                   "Will viewers unlock the next episode?"))
    with c3:
        html_block(risk_card_html(verdict))

    left, right = st.columns([3, 2], gap="large")
    with left:
        html_block(f'<div class="card"><div class="card-title">⚖️ Verdict</div>'
                   f'<div class="card-content">{esc(verdict.get("summary", ""))}</div></div>')
        section("🛠️ Top fixes", "ordered by estimated impact on the weighted score")
        fixes = verdict.get("fixes") or []
        if not fixes:
            html_block('<div class="card"><div class="muted">No major fixes — this episode is in good shape. '
                       '✨</div></div>')
        for i, fix in enumerate(fixes):
            html_block(fix_html(i, fix))
            feedback_widget(result, f"fix-{fix['id']}", "fix", f"{fix['title']}: {fix['action']}")
        with st.expander("How the risk was decided"):
            reasons = "".join(f"<li>{esc(r)}</li>" for r in verdict.get("risk_reasons", []))
            weights = " · ".join(f"{esc(k)} {v:.0%}" for k, v in verdict.get("weights", {}).items())
            html_block(f'<div class="muted"><ul>{reasons}</ul>Weights: {weights}. Bands: ≥ {cfg.RISK_LOW_FROM} Low, '
                       f'≥ {cfg.RISK_MEDIUM_FROM} Medium, otherwise High.</div>')
    with right:
        render_player(result)
        m1, m2 = st.columns(2)
        m1.metric("Pacing", f"{fmt_score(pacing.get('score'))}/10", help="Numbers only: words/s, gaps, slow windows")
        m2.metric("Emotion arc", f"{fmt_score(emotion.get('score'))}/10",
                  help="0.6 × mean intensity + 0.4 × intensity of the final 15 s")
        m3, m4 = st.columns(2)
        m3.metric("Words / sec", f"{pacing.get('wps', 0):.2f}", help=f"Target ≈ {cfg.TARGET_WPS}")
        gap = pacing.get("longest_gap") or {}
        m4.metric("Longest gap", f"{gap.get('length', 0):.1f}s", help=f"Starts at {fmt_ts(gap.get('start'))}")


def render_hook_tab(result: dict) -> None:
    hook = result["hook"]
    points = {p["id"]: p for p in result["emotion"].get("points", [])}
    left, right = st.columns([3, 2], gap="large")
    with left:
        html_block(score_card_html("🎣 Hook score · first 10 seconds", hook.get("score"),
                                   "Weighted mean of the five sub-scores below"))
        html_block(subscores_html("Sub-scores (weights from config.py)", hook.get("sub_scores", {}),
                                  cfg.HOOK_DIMENSIONS, cfg.HOOK_WEIGHTS))
        html_block(diagnosis_html(hook, "Viewer's question"))
    with right:
        html_block(window_html(
            "🕐 Hook window · 0–10 s", hook.get("window_segments", []), points, "seg-hook",
            footer=f"First meaningful line at {hook.get('first_meaningful_line_s', 0):.1f}s → speed score "
                   f"{fmt_score(hook.get('sub_scores', {}).get('speed'))}/10 (measured from timestamps)."))
        for note in hook.get("notes", []):
            st.info(note)

    section("✏️ Rewritten openings", f"in {cfg.LANGUAGES[result['language']]['label']} · English underneath")
    rewrites = hook.get("rewrites") or []
    if not rewrites:
        html_block('<div class="card"><div class="muted">Rewrites unavailable for this run — try analysing '
                   'again.</div></div>')
    else:
        cols = st.columns(len(rewrites), gap="medium")
        for i, (col, rw) in enumerate(zip(cols, rewrites)):
            with col:
                html_block(alternative_html(i, "Opening", rw["line"], rw.get("english", ""),
                                            rw.get("technique", ""), rw.get("why", "")))
                feedback_widget(result, f"hook-rw-{i}", "hook_rewrite", rw["line"])
    render_similar(result, "hook")


def render_cliffhanger_tab(result: dict) -> None:
    cliff = result["cliffhanger"]
    points = {p["id"]: p for p in result["emotion"].get("points", [])}
    left, right = st.columns([3, 2], gap="large")
    with left:
        html_block(score_card_html("🧗 Cliffhanger score · last 15 seconds", cliff.get("score"),
                                   "Weighted mean of the five sub-scores below"))
        html_block(subscores_html("Sub-scores (weights from config.py)", cliff.get("sub_scores", {}),
                                  cfg.CLIFF_DIMENSIONS, cfg.CLIFF_WEIGHTS))
        html_block(diagnosis_html(cliff, "Question they'll pay to answer"))
    with right:
        cut, trim = cliff.get("best_cut_s"), cliff.get("trim_s") or 0.0
        if cut is not None and trim >= 0.5:
            cut_line = f"✂️ Cut at <b>{esc(fmt_ts(cut, True))}</b> — trim the last {trim:.1f} s."
        else:
            cut_line = "✅ The current ending is already the best cut point."
        reason = f'<div class="muted" style="margin-top:.4rem">{esc(cliff.get("cut_reason", ""))}</div>' \
            if cliff.get("cut_reason") else ""
        html_block(f'<div class="card"><div class="card-title">Best timestamp to end the episode</div>'
                   f'<div class="card-content">{cut_line}</div>{reason}</div>')
        window = cliff.get("window") or [0, result["duration"]]
        html_block(window_html(f"🕐 Cliffhanger window · {fmt_ts(window[0])}–{fmt_ts(window[1])}",
                               cliff.get("window_segments", []), points, "seg-cliff"))
        for note in cliff.get("notes", []):
            st.info(note)

    section("🎬 Alternative endings", f"in {cfg.LANGUAGES[result['language']]['label']} · English underneath")
    endings = cliff.get("endings") or []
    if not endings:
        html_block('<div class="card"><div class="muted">Alternative endings unavailable for this run — try '
                   'analysing again.</div></div>')
    else:
        cols = st.columns(len(endings), gap="medium")
        for i, (col, end) in enumerate(zip(cols, endings)):
            with col:
                html_block(alternative_html(i, "Ending", end["ending"], end.get("english", ""),
                                            end.get("technique", ""), end.get("why", "")))
                feedback_widget(result, f"cliff-end-{i}", "ending", end["ending"])
    render_similar(result, "cliffhanger")


def render_pacing_emotion_tab(result: dict) -> None:
    pacing, emotion = result["pacing"], result["emotion"]
    gap = pacing.get("longest_gap") or {}
    m = st.columns(5)
    m[0].metric("Pacing score", f"{fmt_score(pacing.get('score'))}/10")
    m[1].metric("Words / sec", f"{pacing.get('wps', 0):.2f}", help=f"Target ≈ {cfg.TARGET_WPS} words/s overall")
    m[2].metric("Dialogue coverage", f"{pacing.get('dialogue_coverage', 0):.0%}",
                help="Share of the runtime with someone speaking")
    m[3].metric("Silence ratio", f"{pacing.get('silence_ratio', 0):.0%}",
                help=f"pydub detect_silence: ≥ {cfg.SILENCE_MIN_MS} ms, {cfg.SILENCE_REL_DB:.0f} dB below average")
    m[4].metric("Longest gap", f"{gap.get('length', 0):.1f}s", help=f"No dialogue from {fmt_ts(gap.get('start'))}")

    section("⏱️ Dialogue density per 10 seconds", "red = slow window")
    bins = pacing.get("bins") or []
    if bins:
        st.bar_chart(
            {
                "Window": [fmt_ts(b["start"]) for b in bins],
                "Words/sec": [0.0 if b["slow"] else b["wps"] for b in bins],
                "Slow": [b["wps"] if b["slow"] else 0.0 for b in bins],
            },
            x="Window", y=["Words/sec", "Slow"], color=["#7c3aed", "#ef4444"], height=250,
            x_label="Window start", y_label="Words per second",
        )
    sections = pacing.get("slow_sections") or []
    if sections:
        cards = []
        for s in sections:
            kind = s.get("kind", "slow")
            badge = {"drag": '<span class="badge badge-red">Drag · fix it</span>',
                     "pause": '<span class="badge badge-purple">Intentional pause?</span>'}.get(
                kind, '<span class="badge badge-amber">Slow</span>')
            emo = (f" · emotion {s['emotion_mean']:.1f}/10" if s.get("emotion_mean") is not None else "")
            cards.append(f'<div class="fix-card"><div><div class="fix-title">'
                         f'{esc(fmt_ts(s["start"]))}–{esc(fmt_ts(s["end"]))} {badge}</div>'
                         f'<div class="fix-action">{esc("; ".join(s.get("reasons", [])))}{esc(emo)}</div></div></div>')
        html_block("".join(cards))
        st.caption("Drag = slow AND low emotion (trim it). Intentional pause = slow but emotionally loaded "
                   "(probably a dramatic beat — check it by eye).")
    else:
        html_block('<div class="card"><div class="muted">No slow sections flagged. 🚀</div></div>')

    section("💓 Emotion timeline", "LLM-rated intensity per line · peaks & dips found with numpy")
    points = emotion.get("points") or []
    if emotion.get("fallback") or not points:
        html_block('<div class="card"><div class="muted">Emotion timeline unavailable for this run.</div></div>')
        return
    st.line_chart(
        {"Time (s)": [p["mid"] for p in points], "Intensity": [p["intensity"] for p in points],
         "Trend": emotion.get("trend") or [p["intensity"] for p in points]},
        x="Time (s)", y=["Intensity", "Trend"], color=["#9f67ff", "#06b6d4"], height=280,
        x_label="Time (seconds)", y_label="Intensity (0–10)",
    )
    html_block(emotion_strip_html(result))

    col_peaks, col_dips = st.columns(2, gap="large")

    def moments_html(title: str, moments: list[dict], symbol: str) -> str:
        rows = "".join(
            f'<div class="seg"><div class="seg-time">{symbol} {esc(fmt_ts(m["time"], True))}</div>'
            f'<div class="seg-main"><div class="seg-text">{esc(m["text"])}</div></div>'
            f'<div class="seg-emo"><span class="emo-chip" style="--emo:{cfg.EMOTION_COLORS.get(m["emotion"], "#64748b")}">'
            f'{esc(m["emotion"])} · {m["intensity"]:.0f}</span></div></div>' for m in moments)
        rows = rows or '<div class="muted" style="padding:.8rem">None found.</div>'
        return (f'<div class="card"><div class="card-title">{esc(title)}</div>'
                f'<div class="transcript-wrap compact">{rows}</div></div>')

    with col_peaks:
        html_block(moments_html("▲ Peaks", emotion.get("peaks", []), "▲"))
    with col_dips:
        html_block(moments_html("▼ Dips", emotion.get("dips", []), "▼"))

    shares = (emotion.get("stats") or {}).get("shares") or {}
    if shares:
        section("🎭 Emotion mix", "share of dialogue time")
        st.bar_chart({"Emotion": list(shares), "Share (%)": [round(v * 100, 1) for v in shares.values()]},
                     x="Emotion", y="Share (%)", horizontal=True, height=230, color="#7c3aed")


def render_transcript_tab(result: dict) -> None:
    segments = result["transcript"]["segments"]
    points = {p["id"]: p for p in result["emotion"].get("points", [])}
    hook_end = result["hook"]["window"][1]
    cliff_start = result["cliffhanger"]["window"][0]
    top_l, top_r = st.columns([3, 2])
    with top_l:
        query = st.text_input("Search the transcript", placeholder="Find a line…", label_visibility="collapsed")
    with top_r:
        html_block(f'<div class="legend-row" style="justify-content:flex-end;margin-top:.6rem">'
                   f'<span><i style="background:var(--accent-2)"></i>hook window</span>'
                   f'<span><i style="background:var(--accent-glow)"></i>cliffhanger window</span>'
                   f'<span>{len(segments)} lines · {esc(result["transcript"]["engine"])}</span></div>')
    rows = []
    needle = (query or "").strip().lower()
    for seg in segments:
        haystack = f"{seg['text']} {seg.get('text_en', '')}".lower()
        if needle and needle not in haystack:
            continue
        css = "seg-hook" if seg["start"] < hook_end else ("seg-cliff" if seg["end"] > cliff_start else "")
        rows.append(segment_row_html(seg, points.get(seg["id"]), css))
    html_block('<div class="transcript-wrap">' + ("".join(rows) or
               '<div class="muted" style="padding:.8rem">No lines match your search.</div>') + "</div>")

    d1, d2, d3 = st.columns(3)
    slug = file_slug(result)
    d1.download_button("⬇️ Transcript (.txt)", transcript_to_txt(segments, result["title"]),
                       file_name=f"{slug}_transcript.txt", mime="text/plain", width="stretch")
    d2.download_button("⬇️ Subtitles (.srt)", transcript_to_srt(segments),
                       file_name=f"{slug}.srt", mime="application/x-subrip", width="stretch")
    if result["language"] != "english":
        d3.download_button("⬇️ English subtitles (.srt)", transcript_to_srt(segments, english=True),
                           file_name=f"{slug}_en.srt", mime="application/x-subrip", width="stretch")


def render_chat_tab(result: dict) -> None:
    from core.rag_engine import ask_question, build_rag_chain

    section("💬 Chat with this episode", "RAG over timestamped transcript windows + the analysis")
    if st.session_state.chat_chain is None:
        html_block('<div class="card"><div class="muted">The chat index isn\'t built for this session '
                   '(it may have failed during analysis, or the app restarted).</div></div>')
        if st.button("🔄 Build chat index", key="rebuild_chat"):
            built = False
            try:
                with st.spinner("Indexing the transcript…"):
                    st.session_state.chat_chain = build_rag_chain(
                        result["transcript"]["segments"], analysis_brief=build_analysis_brief(result),
                        episode_id=result["episode_id"], embeddings=get_embedding_model())
                built = True
            except HookLensError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001 — show any failure to the user
                st.error(f"Couldn't build the chat index: {exc}")
            if built:
                st.rerun()  # outside try: st.rerun() works by raising a control-flow exception
        return

    suggestions = ["Why did the hook get this score?", "Which line is the emotional peak?",
                   "Rewrite the final line to hit harder"]
    clicked = None
    for col, question in zip(st.columns(len(suggestions)), suggestions):
        if col.button(question, key=f"suggest::{question}", width="stretch"):
            clicked = question

    history = st.session_state.chat_history
    if history:
        bubbles = []
        for msg in history:
            content = esc(msg["content"]).replace("\n", "<br>")
            if msg["role"] == "user":
                bubbles.append(f'<div class="chat-msg" style="align-items:flex-end"><span class="chat-label user-label">'
                               f'You</span><div class="chat-bubble user-bubble">{content}</div></div>')
            else:
                bubbles.append(f'<div class="chat-msg" style="align-items:flex-start"><span class="chat-label bot-label">'
                               f'🤖 HookLens</span><div class="chat-bubble bot-bubble">{content}</div></div>')
        html_block('<div class="chat-container">' + "".join(bubbles) + "</div>")
    else:
        html_block('<div class="card" style="text-align:center;padding:1.6rem"><div style="font-size:1.8rem">💬</div>'
                   '<div class="muted">Ask anything about this episode — lines, characters, scores or rewrites. '
                   'Answers cite timestamps.</div></div>')

    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1], gap="small")
        typed = c1.text_input("Your question", placeholder="e.g. Where does the episode lose energy?",
                              label_visibility="collapsed")
        sent = c2.form_submit_button("Send →", width="stretch")
    question = clicked or (typed.strip() if sent else "")
    if question:
        with st.spinner("Thinking…"):
            try:
                answer = ask_question(st.session_state.chat_chain, question)
            except HookLensError as exc:
                answer = f"⚠️ {exc}"
            except Exception as exc:  # noqa: BLE001
                answer = f"⚠️ Chat failed: {exc}"
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        st.rerun()
    if history and st.button("🗑️ Clear chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()


def render_report_tab(result: dict) -> None:
    from core.report import build_pdf_report

    left, right = st.columns([3, 2], gap="large")
    slug = file_slug(result)
    with left:
        html_block('<div class="card"><div class="card-title">📄 One-page scorecard</div><div class="card-content">'
                   'Hook, cliffhanger and retention-risk cards · sub-scores with bars · pacing stats · emotion '
                   'sparkline · top 3 fixes · best opening rewrite, best alternative ending and suggested cut.'
                   '</div><div class="muted" style="margin-top:.5rem">Devanagari lines are shown in English in the '
                   'PDF (built-in PDF fonts are Latin-only); the app shows the original script.</div></div>')
        cache_key = f"{result['episode_id']}::{result['analysed_at']}"
        try:
            if cache_key not in st.session_state.pdf_cache:
                st.session_state.pdf_cache = {cache_key: build_pdf_report(result)}
            st.download_button("📄 Download PDF scorecard", st.session_state.pdf_cache[cache_key],
                               file_name=f"hooklens_{slug}.pdf", mime="application/pdf", type="primary",
                               width="stretch")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Couldn't build the PDF: {exc}")
        st.download_button("🧾 Download full analysis (JSON)", json.dumps(result, ensure_ascii=False, indent=2),
                           file_name=f"hooklens_{slug}.json", mime="application/json", width="stretch")
        with st.expander("How the scores are computed"):
            hook_w = " · ".join(f"{esc(cfg.HOOK_DIMENSIONS[k])} {v:.0%}" for k, v in cfg.HOOK_WEIGHTS.items())
            cliff_w = " · ".join(f"{esc(cfg.CLIFF_DIMENSIONS[k])} {v:.0%}" for k, v in cfg.CLIFF_WEIGHTS.items())
            risk_w = " · ".join(f"{esc(k)} {v:.0%}" for k, v in cfg.RISK_WEIGHTS.items())
            html_block(f'<div class="muted"><b>Hook</b> = weighted mean: {hook_w}.<br><b>Cliffhanger</b> = weighted '
                       f'mean: {cliff_w}.<br><b>Retention risk</b> = {risk_w}; ≥ {cfg.RISK_LOW_FROM} Low, '
                       f'≥ {cfg.RISK_MEDIUM_FROM} Medium, else High; hook or cliffhanger below '
                       f'{cfg.CRITICAL_COMPONENT} → High (weakest-link rule).<br>The LLM ({esc(cfg.MISTRAL_MODEL)}, '
                       f'temperature 0, rubric with 2/5/9 anchors) scores each dimension; aggregation, pacing, '
                       f'peaks and risk bands are deterministic code.</div>')
    with right:
        stats = feedback_stats()
        rate = stats.get("approval_rate")
        html_block(f'<div class="card"><div class="card-title">👍 Writer feedback (all episodes)</div>'
                   f'<div class="score-value" style="font-size:2.2rem;color:var(--accent-2)">'
                   f'{"—" if rate is None else f"{rate:.0%}"}</div>'
                   f'<div class="muted">approval across {stats.get("total", 0)} votes on '
                   f'{stats.get("episodes", 0)} episode(s) · stored in data/feedback.json</div></div>')
        by_type = stats.get("by_type") or {}
        if by_type:
            labels = {"fix": "Top fixes", "hook_rewrite": "Hook rewrites", "ending": "Alternative endings"}
            rows = []
            for kind, v in by_type.items():
                approval = "—" if v["approval_rate"] is None else f"{v['approval_rate']:.0%}"
                rows.append(f'<div class="key-row"><span>{esc(labels.get(kind, kind))}</span>'
                            f'<span>👍 {v["up"]} · 👎 {v["down"]} · {approval}</span></div>')
            rows = "".join(rows)
            html_block(f'<div class="card"><div class="card-title">By suggestion type</div>{rows}</div>')
        st.caption("Why: approval rate per suggestion type is the first evaluation signal for prompt changes — "
                   "if writers reject most endings, the ending prompt gets fixed first.")


def render_empty_state() -> None:
    html_block("""
    <div class="empty-wrap">
      <div style="font-size:3.4rem">🎣</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;margin:.4rem 0">Score your episode before it ships</div>
      <div class="muted" style="max-width:560px;margin:0 auto">Paste a YouTube link or upload a 1–3 minute micro-drama episode
      (English, Hindi or Hinglish). HookLens tells you whether viewers will stay past 10 seconds, whether they'll unlock the
      next episode, where it drags — and exactly how to fix it.</div>
      <div class="empty-steps">
        <div class="empty-step">🎣<b>Hook · first 10 s</b><span>Conflict, curiosity gap, emotion, stakes and speed to the
        first line — plus 3 rewritten openings.</span></div>
        <div class="empty-step">🧗<b>Cliffhanger · last 15 s</b><span>Unresolved question, rising stakes, reveal, peak at the
        cut, "would I pay?" — plus the best timestamp to cut.</span></div>
        <div class="empty-step">⚖️<b>Retention risk</b><span>Weighted verdict, top 3 fixes by impact, pacing and emotion
        timelines, similar past hooks and a PDF scorecard.</span></div>
      </div>
    </div>""")


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    html_block('<div class="brand-title">🎣 HookLens</div>'
               '<div class="brand-sub">Hook &amp; cliffhanger analyzer · Kuku TV</div>')
    html_block('<span class="badge badge-purple">Input</span>')
    input_mode = st.radio("Source", ["YouTube link", "Upload file"], horizontal=True, label_visibility="collapsed")
    url, uploaded = "", None
    if input_mode == "YouTube link":
        url = st.text_input("YouTube link", placeholder="https://youtube.com/shorts/…")
        st.caption("If YouTube blocks the server download, use Upload instead.")
    else:
        uploaded = st.file_uploader("Episode file", type=list(cfg.ALLOWED_UPLOAD_TYPES),
                                    help="Video or audio of a 1–3 minute episode (.mp4 .mov .mkv .mp3 .wav .m4a)")
    language = st.selectbox("Episode language", list(cfg.LANGUAGES),
                            format_func=lambda key: cfg.LANGUAGES[key]["label"])
    with st.expander("Options"):
        translate = st.checkbox("English translation per line", value=True,
                                help="deep-translator; only used for Hindi / Hinglish episodes")
        save_bench = st.checkbox("Save to benchmark library", value=True,
                                 help="Adds this hook and cliffhanger (with scores) to the ChromaDB benchmark")
    run_btn = st.button("⚡  Analyse episode", type="primary", width="stretch")
    status_placeholder = st.empty()
    render_status(status_placeholder)

    html_block("<hr>")
    mistral_ok, sarvam_ok = bool(cfg.mistral_api_key()), bool(cfg.sarvam_api_key())
    html_block(
        '<span class="badge badge-muted">Keys</span>'
        f'<div class="key-row"><span>Mistral (required)</span><span class="{"key-ok" if mistral_ok else "key-miss"}">'
        f'{"✓ set" if mistral_ok else "✗ missing"}</span></div>'
        f'<div class="key-row"><span>Sarvam (Hindi/Hinglish)</span><span class="{"key-ok" if sarvam_ok else "key-miss"}">'
        f'{"✓ set" if sarvam_ok else "– Whisper fallback"}</span></div>')
    st.caption(f"{cfg.APP_NAME} v{cfg.APP_VERSION} · Whisper {cfg.WHISPER_MODEL} · {cfg.MISTRAL_MODEL} · "
               "ChromaDB · Streamlit")

# ─── Main Area ────────────────────────────────────────────────────────────────
if st.session_state.toast:
    st.toast(st.session_state.toast)
    st.session_state.toast = None

html_block('<div class="hero-title">HookLens</div>'
           '<div class="hero-sub">Hook · Cliffhanger · Pacing · Emotion — before you publish</div>')
html_block("<hr>")

# ── Run Pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    st.session_state.error = None
    source, display_name = None, None
    if input_mode == "YouTube link":
        if not url.strip():
            st.session_state.error = ("Paste a YouTube link first.", None, None)
        elif not is_youtube_url(url):
            st.session_state.error = ("That doesn't look like a YouTube video link.",
                                      "Use a link like https://youtube.com/watch?v=…, https://youtu.be/… or a "
                                      "/shorts/ link.", None)
        else:
            source = url.strip()
    elif uploaded is None:
        st.session_state.error = ("Drop an episode file into the uploader first.", None, None)
    else:
        try:
            source, display_name = str(save_uploaded_file(uploaded)), uploaded.name
        except HookLensError as exc:
            st.session_state.error = (exc.message, exc.hint, None)

    if source:
        st.session_state.update(result=None, chat_chain=None, chat_history=[], pipeline_steps={}, step_times={})
        notice = st.empty()
        notice.info("⚙️ Analysing — live progress is in the sidebar. A 2-minute episode usually takes 1–3 minutes.")
        finished = False
        try:
            result, chain = run_pipeline(
                source, language, display_name=display_name, translate=translate, save_to_benchmark=save_bench,
                whisper_loader=get_whisper_model, embeddings_loader=get_embedding_model,
                on_step=make_step_callback(status_placeholder),
            )
            st.session_state.result, st.session_state.chat_chain = result, chain
            finished = True
        except HookLensError as exc:
            st.session_state.error = (exc.message, exc.hint, None)
        except Exception:  # noqa: BLE001 — unexpected: generic message + details for developers
            st.session_state.error = ("Something went wrong while analysing this episode.",
                                      "Please try again. If it keeps happening, open the details below.",
                                      traceback.format_exc())
        notice.empty()
        if finished:
            st.rerun()  # outside try: st.rerun() works by raising a control-flow exception

# ── Errors ────────────────────────────────────────────────────────────────────
if st.session_state.error:
    message, hint, details = st.session_state.error
    hint_html = f'<div class="error-hint">{esc(hint)}</div>' if hint else ""
    html_block(f'<div class="error-card"><div class="error-title">⚠️ {esc(message)}</div>{hint_html}</div>')
    if details:
        with st.expander("Technical details"):
            st.code(details)

# ── Results ───────────────────────────────────────────────────────────────────
result = st.session_state.result
if result:
    lang_label = cfg.LANGUAGES.get(result["language"], {}).get("label", result["language"])
    logline = result["verdict"].get("logline")
    logline_html = f'<div class="episode-logline">{esc(logline)}</div>' if logline else ""
    html_block(
        f'<div class="card"><div class="card-title">📌 Episode</div><div class="episode-title">{esc(result["title"])}</div>'
        f'{logline_html}<div class="badge-row"><span class="badge badge-purple">{esc(lang_label)}</span>'
        f'<span class="badge badge-cyan">{esc(fmt_ts(result["duration"]))} runtime</span>'
        f'<span class="badge badge-muted">{esc(result["transcript"]["engine"])}</span>'
        f'<span class="badge badge-muted">{esc(result["models"]["llm"])}</span></div></div>')
    if result.get("warnings"):
        with st.expander(f"⚠️ {len(result['warnings'])} note(s) about this analysis"):
            html_block('<div class="muted"><ul>' + "".join(f"<li>{esc(w)}</li>" for w in result["warnings"])
                       + "</ul></div>")

    tabs = st.tabs(["📊 Overview", "🎣 Hook", "🧗 Cliffhanger", "⏱️ Pacing & Emotion", "📝 Transcript",
                    "💬 Chat", "📄 Report"])
    with tabs[0]:
        render_overview(result)
    with tabs[1]:
        render_hook_tab(result)
    with tabs[2]:
        render_cliffhanger_tab(result)
    with tabs[3]:
        render_pacing_emotion_tab(result)
    with tabs[4]:
        render_transcript_tab(result)
    with tabs[5]:
        render_chat_tab(result)
    with tabs[6]:
        render_report_tab(result)
elif not st.session_state.error:
    render_empty_state()
