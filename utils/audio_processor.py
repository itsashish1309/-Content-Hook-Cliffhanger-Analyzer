"""
utils/audio_processor.py — turn any episode source into clean 16 kHz mono WAV.

Reused from AI Video Assistant: download_youtube_audio, convert_to_wav,
chunk_audio and process_input keep their names and jobs. Extended for HookLens:

  • save_uploaded_file(): Streamlit uploads → data/uploads with a safe name.
  • Every source is normalised to 16 kHz mono WAV. Whisper and Sarvam both work
    best at 16 kHz, and the pacing maths needs one consistent format.
  • YouTube downloads are named by video id (titles can contain characters that
    break file paths) and return metadata (title, duration) for the UI.
  • Guard rails: 15-minute limit, live streams rejected, ffprobe pre-check so
    "this video has no audio track" is a friendly message, not a stack trace.
  • prepare_media(): one call that returns everything the pipeline needs.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import config as cfg
from utils.errors import HookLensError

DOWNLOAD_DIR = cfg.DOWNLOAD_DIR

_YT_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtu.be", "www.youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com",
}
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


# ── Source detection ──────────────────────────────────────────────────────────
def is_url(source: str) -> bool:
    return (source or "").strip().lower().startswith(("http://", "https://"))


def youtube_video_id(url: str) -> str | None:
    """Return the 11-character video id for watch / youtu.be / shorts / embed / live links."""
    try:
        parsed = urlparse((url or "").strip())
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    if host not in _YT_HOSTS:
        return None
    if host.endswith("youtu.be"):
        vid = parsed.path.lstrip("/").split("/")[0]
    else:
        vid = (parse_qs(parsed.query).get("v") or [""])[0]
        if not vid:
            match = re.match(r"^/(?:shorts|embed|live|v)/([^/?#]+)", parsed.path)
            vid = match.group(1) if match else ""
    return vid if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid or "") else None


def is_youtube_url(url: str) -> bool:
    return youtube_video_id(url) is not None


# ── Uploads ───────────────────────────────────────────────────────────────────
def save_uploaded_file(uploaded_file, upload_dir: Path = cfg.UPLOAD_DIR) -> Path:
    """Persist a Streamlit UploadedFile so the reused path-based pipeline can read it."""
    name = getattr(uploaded_file, "name", "") or "episode"
    ext = Path(name).suffix.lower().lstrip(".")
    if ext not in cfg.ALLOWED_UPLOAD_TYPES:
        raise HookLensError(
            f"“.{ext or '?'}” files aren't supported.",
            hint="Use one of: " + ", ".join(f".{e}" for e in cfg.ALLOWED_UPLOAD_TYPES),
        )
    data = uploaded_file.getbuffer()
    if len(data) == 0:
        raise HookLensError("The uploaded file is empty.")
    stem = _SAFE_CHARS.sub("_", Path(name).stem)[:60].strip("._") or "episode"
    path = Path(upload_dir) / f"{uuid.uuid4().hex[:10]}_{stem}.{ext}"
    path.write_bytes(data)
    return path


def cleanup_old_files(folder: Path = cfg.UPLOAD_DIR, max_age_hours: float = cfg.UPLOAD_TTL_HOURS) -> int:
    """Delete files older than max_age_hours (keeps the server disk from filling up)."""
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for path in Path(folder).glob("*"):
        if path.name == ".gitkeep" or not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def file_fingerprint(path: str | Path) -> str:
    """Short SHA-1 of the file bytes → stable episode id (same file = same id)."""
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


# ── FFmpeg helpers ────────────────────────────────────────────────────────────
def _ffmpeg_missing() -> HookLensError:
    return HookLensError(
        "FFmpeg is not installed on this machine.",
        hint="Install it (brew install ffmpeg · choco install ffmpeg · apt install ffmpeg). "
             "On Streamlit Cloud it is installed from packages.txt.",
    )


def probe_media(path: str) -> dict:
    """ffprobe (via ffmpeg-python): duration and whether audio / video streams exist."""
    try:
        import ffmpeg  # ffmpeg-python
    except ImportError:  # can't probe → let the decoder decide
        return {"duration": None, "has_audio": True, "has_video": False}
    try:
        info = ffmpeg.probe(path)
    except FileNotFoundError as exc:
        raise _ffmpeg_missing() from exc
    except ffmpeg.Error as exc:
        detail = (exc.stderr or b"").decode("utf-8", "ignore").strip()[-300:]
        print(f"ffprobe failed for {path}: {detail}")
        raise HookLensError(
            "This file couldn't be read as audio or video.",
            hint="It may be corrupted or use an unusual codec — try exporting it again as .mp4 or .mp3.",
        ) from exc
    streams = info.get("streams", [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    # Cover art inside an .mp3/.m4a shows up as a "video" stream with attached_pic=1.
    has_video = any(
        s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic", 0)
        for s in streams
    )
    try:
        duration = float(info.get("format", {}).get("duration") or 0) or None
    except (TypeError, ValueError):
        duration = None
    return {"duration": duration, "has_audio": has_audio, "has_video": has_video}


# ── YouTube ───────────────────────────────────────────────────────────────────
def _friendly_youtube_error(exc: Exception) -> HookLensError:
    message = _ANSI.sub("", str(exc))
    low = message.lower()
    if "sign in to confirm" in low or "not a bot" in low or "http error 429" in low:
        return HookLensError(
            "YouTube is blocking downloads from this server (bot check).",
            hint="Download the episode and use “Upload file” instead — or run HookLens locally.",
        )
    if "javascript runtime" in low or "js runtime" in low or "challenge" in low:
        return HookLensError(
            "yt-dlp needs a JavaScript runtime (Deno) to read YouTube right now.",
            hint="Install Deno (https://deno.com) and restart, or use “Upload file”.",
        )
    if "private video" in low:
        return HookLensError("This YouTube video is private.", hint="Make it unlisted, or upload the file.")
    if "age" in low and ("confirm your age" in low or "age-restricted" in low or "inappropriate" in low):
        return HookLensError("This video is age-restricted, so it can't be downloaded without signing in.",
                             hint="Upload the file instead.")
    if "requested format is not available" in low:  # must be checked before "not available"
        return HookLensError("YouTube didn't offer an audio stream for this video.",
                             hint="Update yt-dlp (pip install -U \"yt-dlp[default]\"), install Deno, "
                                  "or upload the file.")
    if "video unavailable" in low or "not available" in low or "removed" in low:
        return HookLensError("This video is unavailable (removed, region-locked or restricted).")
    if "unsupported url" in low or "not a valid url" in low:
        return HookLensError("That link isn't a supported YouTube video URL.")
    if "ffmpeg" in low or "ffprobe" in low:
        return _ffmpeg_missing()
    first_line = message.splitlines()[0][:200] if message else type(exc).__name__
    return HookLensError("Couldn't download this YouTube video.", hint=f"Details: {first_line}")


def download_youtube_audio(url: str) -> tuple[str, dict]:
    """Download the audio track of a YouTube video as WAV → (wav_path, metadata)."""
    try:
        import yt_dlp
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise HookLensError("yt-dlp is not installed.", hint="Run: pip install -r requirements.txt") from exc

    video_id = youtube_video_id(url)
    if not video_id:
        raise HookLensError(
            "That doesn't look like a YouTube video link.",
            hint="Paste a link like https://youtube.com/watch?v=… , https://youtu.be/… or a /shorts/ link.",
        )
    if shutil.which("ffmpeg") is None:
        raise _ffmpeg_missing()

    def reject_long_or_live(info: dict, *, incomplete: bool = False):
        """yt-dlp match_filter: returning a string skips the download."""
        if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
            return "live streams are not supported"
        duration = info.get("duration")
        if duration and duration > cfg.MAX_EPISODE_SECONDS:
            return f"video is {int(duration)} s long"
        return None

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(DOWNLOAD_DIR / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "match_filter": reject_long_or_live,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }
    cookies = cfg.env("YTDLP_COOKIES_FILE")
    if cookies and Path(cookies).is_file():
        ydl_opts["cookiefile"] = cookies

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadError as exc:
        raise _friendly_youtube_error(exc) from exc

    if not info:
        raise HookLensError("YouTube returned no information for this link.")
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        raise HookLensError("Live streams can't be analysed.", hint="Use a finished episode.")
    duration = float(info.get("duration") or 0)
    if duration > cfg.MAX_EPISODE_SECONDS:
        raise HookLensError(
            f"This video is {duration / 60:.0f} minutes long — HookLens is built for 1–3 minute episodes.",
            hint=f"Limit: {cfg.MAX_EPISODE_SECONDS // 60} minutes.",
        )

    # After FFmpegExtractAudio, requested_downloads[*].filepath points at the .wav.
    wav_path = None
    for item in info.get("requested_downloads") or []:
        if item.get("filepath"):
            wav_path = item["filepath"]
    if not wav_path or not Path(wav_path).exists():
        candidate = DOWNLOAD_DIR / f"{info.get('id') or video_id}.wav"
        wav_path = str(candidate) if candidate.exists() else None
    if not wav_path:
        raise HookLensError("The download finished but no audio file was produced.",
                            hint="Check that FFmpeg is installed.")

    vid = info.get("id") or video_id
    meta = {
        "id": vid,
        "title": info.get("title") or f"YouTube video {vid}",
        "duration": duration or None,
        "uploader": info.get("uploader") or info.get("channel"),
        "webpage_url": f"https://www.youtube.com/watch?v={vid}",   # canonical (works for Shorts too)
        "thumbnail": info.get("thumbnail"),
    }
    return wav_path, meta


# ── Conversion & chunking (reused) ────────────────────────────────────────────
def convert_to_wav(input_path: str, output_dir: Path | None = None) -> str:
    """Convert any audio/video file to 16 kHz mono WAV with pydub."""
    from pydub import AudioSegment
    from pydub.exceptions import CouldntDecodeError

    src = Path(input_path)
    out_dir = Path(output_dir) if output_dir else DOWNLOAD_DIR
    output_path = out_dir / f"{src.stem}_16k.wav"
    try:
        audio = AudioSegment.from_file(str(src))
    except FileNotFoundError as exc:
        if not src.exists():
            raise HookLensError(f"File not found: {src.name}") from exc
        raise _ffmpeg_missing() from exc
    except (CouldntDecodeError, IndexError, OSError) as exc:
        raise HookLensError(
            "Couldn't decode the audio in this file.",
            hint="It may have no audio track or be corrupted — try re-exporting it as .mp4 or .mp3.",
        ) from exc
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16 kHz mono
    audio.export(str(output_path), format="wav")
    return str(output_path)


def chunk_audio(wav_path: str, chunk_minutes: int = cfg.CHUNK_MINUTES) -> list:
    """Split long audio into chunks (reused). 1–3 min episodes stay a single chunk."""
    from pydub import AudioSegment

    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    if len(audio) <= chunk_ms:
        return [wav_path]  # no re-export needed for micro-drama episodes

    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks


def audio_stats(wav_path: str) -> tuple[float, float]:
    """(duration in seconds, average loudness in dBFS)."""
    from pydub import AudioSegment

    audio = AudioSegment.from_wav(wav_path)
    return len(audio) / 1000.0, audio.dBFS


def _pretty_title(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    if stem and stem == stem.lower():
        stem = stem.title()
    return stem or "Untitled episode"


# ── One-call entry point ──────────────────────────────────────────────────────
def prepare_media(source: str, *, display_name: str | None = None) -> dict:
    """
    Normalise a YouTube URL or local/uploaded file for analysis.

    Returns a dict with: source, source_type, title, media_id, wav_path, chunks,
    chunk_seconds, duration, youtube_url, video_path, is_video, uploader.
    """
    source = (source or "").strip()
    if not source:
        raise HookLensError("No input given.", hint="Paste a YouTube link or upload an episode file.")

    if is_url(source):
        print("Detected YouTube URL. Downloading audio...")
        raw_wav, meta = download_youtube_audio(source)
        wav_path = convert_to_wav(raw_wav)  # yt-dlp WAV is 44.1/48 kHz stereo → 16 kHz mono
        if Path(raw_wav).resolve() != Path(wav_path).resolve():
            Path(raw_wav).unlink(missing_ok=True)
        media = {
            "source_type": "youtube",
            "title": meta["title"],
            "media_id": f"yt-{meta['id']}",
            "youtube_url": meta["webpage_url"],
            "video_path": None,
            "is_video": True,
            "uploader": meta.get("uploader"),
        }
    else:
        path = Path(source).expanduser()
        if not path.exists():
            raise HookLensError(f"File not found: {path}")
        ext = path.suffix.lower().lstrip(".")
        if ext not in cfg.ALLOWED_UPLOAD_TYPES:
            raise HookLensError(
                f"“.{ext or '?'}” files aren't supported.",
                hint="Use one of: " + ", ".join(f".{e}" for e in cfg.ALLOWED_UPLOAD_TYPES),
            )
        probe = probe_media(str(path))
        if not probe["has_audio"]:
            raise HookLensError(
                "This file has no audio track.",
                hint="HookLens analyses dialogue, so the episode needs sound.",
            )
        if probe["duration"] and probe["duration"] > cfg.MAX_EPISODE_SECONDS:
            raise HookLensError(
                f"This file is {probe['duration'] / 60:.0f} minutes long — HookLens is built for 1–3 minute episodes.",
                hint=f"Limit: {cfg.MAX_EPISODE_SECONDS // 60} minutes.",
            )
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(str(path))
        media = {
            "source_type": "upload" if display_name else "file",
            "title": _pretty_title(display_name or path.name),
            "media_id": f"file-{file_fingerprint(path)}",
            "youtube_url": None,
            "video_path": str(path),
            "is_video": bool(probe["has_video"]) or ext in cfg.VIDEO_EXTENSIONS,
            "uploader": None,
        }

    duration, loudness = audio_stats(wav_path)
    if duration < cfg.MIN_EPISODE_SECONDS:
        raise HookLensError(f"The audio is only {duration:.1f} s long — too short to analyse.")
    if duration > cfg.MAX_EPISODE_SECONDS:
        raise HookLensError(f"The audio is {duration / 60:.0f} minutes long — too long for a micro-drama episode.")
    if loudness == float("-inf") or loudness < -60:
        raise HookLensError("The audio track is silent.", hint="Check that the file actually contains dialogue.")

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    media.update({
        "source": source,
        "wav_path": wav_path,
        "chunks": chunks,
        "chunk_seconds": cfg.CHUNK_MINUTES * 60,
        "duration": round(duration, 2),
    })
    return media


def process_input(source: str) -> list:
    """Backward-compatible API from AI Video Assistant: returns the WAV chunk paths."""
    return prepare_media(source)["chunks"]
