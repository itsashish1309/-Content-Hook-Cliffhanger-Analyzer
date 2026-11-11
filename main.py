"""
main.py — HookLens command-line entry point.

Runs the exact same pipeline as the Streamlit app (core/pipeline.py), prints a
scorecard, optionally writes the PDF / JSON, and can drop into "chat with this
episode" — like AI Video Assistant's CLI did for meetings.

Examples
    python main.py "https://youtube.com/shorts/XXXXXXXXXXX" -l hinglish --pdf
    python main.py episode_03.mp4 --json out.json --chat
    python main.py            # interactive prompts, as before
"""
from dotenv import load_dotenv

load_dotenv()  # MUST run before any core/ imports (config reads env at import time)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import config as cfg  # noqa: E402
from core.common import fmt_ts  # noqa: E402
from core.pipeline import STEPS, run_pipeline  # noqa: E402
from utils.errors import HookLensError  # noqa: E402

_LABELS = {key: (icon, label) for key, icon, label in STEPS}


def _print_step(key: str, state: str) -> None:
    icon, label = _LABELS.get(key, ("•", key))
    marks = {"active": "⏳", "done": "✅", "error": "⚠️ "}
    if state in marks:
        print(f"{marks[state]} {icon} {label}{'…' if state == 'active' else ''}", flush=True)


def _score(value) -> str:
    return "  n/a" if value is None else f"{value:4.1f}"


def print_report(result: dict) -> None:
    hook, cliff, verdict = result["hook"], result["cliffhanger"], result["verdict"]
    pacing, emotion = result["pacing"], result["emotion"]
    line = "=" * 72
    print(f"\n{line}\n🎬 {result['title']}   ({result['language']}, {fmt_ts(result['duration'])})\n{line}")
    print(f"🎣 Hook          {_score(hook['score'])}/10")
    print(f"🧗 Cliffhanger   {_score(cliff['score'])}/10")
    print(f"⚖️  Retention     {verdict['risk'].upper()}  (weighted {_score(verdict['overall'])}/10)")
    print(f"\n{verdict['summary']}")

    print("\n— Hook sub-scores —")
    for key, label in cfg.HOOK_DIMENSIONS.items():
        print(f"  {label:<28}{_score(hook['sub_scores'].get(key))}")
    print(f"  Why: {hook.get('reason', '')}")
    for i, rw in enumerate(hook.get("rewrites", []), 1):
        print(f"  ✏️  Rewrite {i} [{rw.get('technique', '')}]: {rw['line']}")
        if rw.get("english") and rw["english"] != rw["line"]:
            print(f"      EN: {rw['english']}")

    print("\n— Cliffhanger sub-scores —")
    for key, label in cfg.CLIFF_DIMENSIONS.items():
        print(f"  {label:<28}{_score(cliff['sub_scores'].get(key))}")
    print(f"  Why: {cliff.get('reason', '')}")
    print(f"  ✂️  Best cut: {fmt_ts(cliff.get('best_cut_s'), True)} (trim {cliff.get('trim_s', 0):.1f}s) "
          f"— {cliff.get('cut_reason', '')}")
    for i, end in enumerate(cliff.get("endings", []), 1):
        print(f"  ✏️  Ending {i} [{end.get('technique', '')}]: {end['ending']}")
        if end.get("english") and end["english"] != end["ending"]:
            print(f"      EN: {end['english']}")

    print("\n— Pacing & emotion —")
    for note in pacing.get("notes", []):
        print(f"  • {note}")
    stats = emotion.get("stats") or {}
    if stats:
        print(f"  • Emotion mean {stats['mean']}/10, final 15 s {stats['end_mean']}/10, mostly {stats['dominant']}")
    for peak in emotion.get("peaks", []):
        print(f"  ▲ {fmt_ts(peak['time'])} {peak['emotion']} {peak['intensity']:.0f}/10 — {peak['text']}")

    print("\n— Top fixes (by impact) —")
    for i, fix in enumerate(verdict.get("fixes", []), 1):
        print(f"  {i}. {fix['title']}  (+{fix['impact']:.2f} pts)\n     {fix['action']}")

    bench = result.get("benchmark") or {}
    if bench.get("similar_hooks"):
        print("\n— Most similar past hooks —")
        for item in bench["similar_hooks"]:
            print(f"  {item['similarity']:.0%} similar · {item['score']}/10 · {item['title']}")
    if result.get("warnings"):
        print("\n— Notes —")
        for warning in result["warnings"]:
            print(f"  ⚠️  {warning}")
    print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="HookLens — score a micro-drama episode's hook and cliffhanger.")
    parser.add_argument("source", nargs="?", help="YouTube URL or path to a video/audio file")
    parser.add_argument("-l", "--language", choices=list(cfg.LANGUAGES), help="episode language")
    parser.add_argument("--pdf", nargs="?", const="", help="write the PDF scorecard (optional path)")
    parser.add_argument("--json", help="write the full analysis JSON to this path")
    parser.add_argument("--no-translate", action="store_true", help="skip English line translations")
    parser.add_argument("--no-benchmark", action="store_true", help="don't save this episode to the benchmark")
    parser.add_argument("--chat", action="store_true", help="chat with the episode after the analysis")
    args = parser.parse_args()

    # Interactive prompts when run without arguments (same feel as the old CLI).
    source = args.source or input("Enter YouTube URL or local file path: ").strip()
    language = args.language or (input("Language (english/hindi/hinglish) [english]: ").strip().lower() or "english")
    if language not in cfg.LANGUAGES:
        print(f"Unknown language '{language}', using english.")
        language = "english"

    print("🎬 Starting HookLens")
    try:
        result, chat_chain = run_pipeline(source, language, translate=not args.no_translate,
                                          save_to_benchmark=not args.no_benchmark,
                                          build_chat=args.chat, on_step=_print_step)
    except HookLensError as exc:
        print(f"\n❌ {exc}")
        return 1

    print_report(result)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"💾 JSON written to {args.json}")
    if args.pdf is not None:
        from core.report import save_pdf_report

        path = save_pdf_report(result, args.pdf or None)
        print(f"📄 PDF written to {path}")

    if args.chat and chat_chain is not None:
        from core.rag_engine import ask_question

        print("\n💬 Chat with this episode (type 'exit' to quit)\n")
        while True:
            question = input("You: ").strip()
            if question.lower() in ("exit", "quit", "q"):
                print("👋 Goodbye!")
                break
            if question:
                print(f"\n🤖 HookLens: {ask_question(chat_chain, question)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
