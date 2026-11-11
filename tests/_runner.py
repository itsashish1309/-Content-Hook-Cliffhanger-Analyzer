"""
tests/_runner.py — a tiny test runner so every test file works with plain
`python tests/test_x.py` (no pytest needed — it isn't part of the stack).
The test_* functions are also pytest-compatible if you prefer pytest.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class SkipTest(Exception):
    """Raise inside a test to skip it (missing optional dependency, no API key, …)."""


def run(namespace: dict) -> int:
    tests = [(name, fn) for name, fn in namespace.items() if name.startswith("test_") and callable(fn)]
    failed = skipped = 0
    print(f"\n▶ {Path(namespace.get('__file__', 'tests')).name}")
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
        except SkipTest as exc:
            skipped += 1
            print(f"  ⏭️  {name} — skipped: {exc}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"  ❌ {name}")
            traceback.print_exc()
    passed = len(tests) - failed - skipped
    print(f"  → {passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


def require(module_name: str):
    """Import an optional dependency or skip the test."""
    try:
        return __import__(module_name)
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"{module_name} not available ({exc.__class__.__name__})") from exc
