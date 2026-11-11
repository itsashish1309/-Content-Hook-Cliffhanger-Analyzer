"""
Run every offline test module:   python -m tests.run_all
(Live / manual checks: tests/test_live_llm.py and tests/test_transcriber.py.)
"""
from __future__ import annotations

import importlib

from tests._runner import run

MODULES = [
    "tests.test_common",
    "tests.test_pacing",
    "tests.test_prompts",
    "tests.test_llm_json",
    "tests.test_analyzers",
    "tests.test_feedback",
    "tests.test_audio_processor",
    "tests.test_rag",
    "tests.test_pipeline_offline",
    "tests.test_report",
    "tests.test_transcriber",
]


def main() -> int:
    failures = 0
    for name in MODULES:
        module = importlib.import_module(name)
        failures += run(vars(module))
    print("\n" + ("✅ all offline test modules passed" if not failures else f"❌ {failures} module(s) failed"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
