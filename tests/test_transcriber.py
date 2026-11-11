"""
Manual transcription check (needs Whisper, or Sarvam for Hindi/Hinglish).

    python tests/test_transcriber.py path/to/episode.mp4 [english|hindi|hinglish]

Prints the timestamped segments — useful to sanity-check a new episode format.
Without arguments it only runs the offline Sarvam-response parsing test.
"""
import sys

from dotenv import load_dotenv

load_dotenv()

from tests._runner import run  # noqa: E402


def test_sarvam_timestamp_parsing():
    from core.transcriber import _parse_sarvam

    body = {"transcript": "a b", "timestamps": {"words": ["Ruko!", "Ye shaadi nahi ho sakti."],
                                                "start_time_seconds": [0.2, 1.4],
                                                "end_time_seconds": [1.1, 3.9]}}
    segments = _parse_sarvam(body, offset_s=25.0, speech_start_s=25.0, speech_end_s=29.0)
    assert [s["start"] for s in segments] == [25.2, 26.4] and segments[1]["text"] == "Ye shaadi nahi ho sakti."
    fallback = _parse_sarvam({"transcript": "poora piece"}, offset_s=0, speech_start_s=0.5, speech_end_s=9.0)
    assert fallback == [{"start": 0.5, "end": 9.0, "text": "poora piece"}]


def manual(path: str, language: str = "english") -> None:
    from core.transcriber import transcribe_all
    from core.common import fmt_range
    from utils.audio_processor import prepare_media

    media = prepare_media(path)
    out = transcribe_all(media["chunks"], language)
    print(f"\nEngine: {out['engine']} · {len(out['segments'])} segments · warnings: {out['warnings']}")
    for seg in out["segments"]:
        print(f"[{fmt_range(seg['start'], seg['end'])}] {seg['text']}")
        if seg.get("text_en") and seg["text_en"] != seg["text"]:
            print(f"{'':>17}EN: {seg['text_en']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        manual(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "english")
    else:
        raise SystemExit(run(globals()))
