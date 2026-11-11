"""
utils/errors.py — one exception type for problems a writer can act on.

Anything raised as HookLensError is shown verbatim in the UI (bad link, no
audio, missing API key…). Every other exception is treated as a bug: the UI
shows a generic message and keeps the traceback in an expander for developers.
"""
from __future__ import annotations


class HookLensError(Exception):
    """An error whose message is safe and friendly to show to a writer."""

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message} {self.hint}" if self.hint else self.message
