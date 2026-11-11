"""Input handling: URL parsing (pure) and a real pydub/FFmpeg round-trip on a generated tone."""
import tempfile
from pathlib import Path

from tests._runner import SkipTest, require, run

from utils.audio_processor import is_youtube_url, youtube_video_id
from utils.errors import HookLensError


def test_youtube_url_parsing():
    assert youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s") == "dQw4w9WgXcQ"
    assert youtube_video_id("https://youtu.be/dQw4w9WgXcQ?si=abc") == "dQw4w9WgXcQ"
    assert youtube_video_id("https://youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert youtube_video_id("https://m.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert not is_youtube_url("https://vimeo.com/123456")
    assert not is_youtube_url("not a url")


def test_prepare_media_on_generated_audio():
    require("pydub")
    import shutil

    if shutil.which("ffmpeg") is None:
        raise SkipTest("FFmpeg binary not installed")
    from pydub.generators import Sine

    from utils.audio_processor import prepare_media

    folder = Path(tempfile.mkdtemp())
    tone = folder / "tone_episode.mp3"
    Sine(440).to_audio_segment(duration=6000).apply_gain(-6).export(tone, format="mp3")
    media = prepare_media(str(tone))
    assert 5.5 < media["duration"] < 6.5
    assert media["chunks"] == [media["wav_path"]] and media["media_id"].startswith("file-")
    assert not media["is_video"]


def test_friendly_errors():
    from utils.audio_processor import prepare_media

    for bad in ("", "/definitely/not/here.mp4"):
        try:
            prepare_media(bad)
            raise AssertionError("should have raised")
        except HookLensError as exc:
            assert exc.message


if __name__ == "__main__":
    raise SystemExit(run(globals()))
