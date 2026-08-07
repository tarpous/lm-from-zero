"""Dependency-backed live progress reporting for long-running commands."""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Mapping
from typing import TextIO

from tqdm import tqdm  # type: ignore[import-untyped]


def progress_enabled(stream: TextIO | None = None) -> bool:
    """Return whether live progress should be rendered for ``stream``.

    Progress is terminal-only by default so JSON-producing commands remain
    quiet when redirected. ``LM_FROM_ZERO_PROGRESS=1`` forces it on, while
    ``LM_FROM_ZERO_PROGRESS=0`` disables it even in an interactive terminal.
    """

    value = os.environ.get("LM_FROM_ZERO_PROGRESS", "").strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    target = sys.stderr if stream is None else stream
    return bool(getattr(target, "isatty", lambda: False)())


def _postfix(fields: Mapping[str, object]) -> dict[str, object]:
    """Keep live metrics compact while letting tqdm format the line."""

    compact: dict[str, object] = {}
    for key, value in fields.items():
        if isinstance(value, float):
            compact[key] = f"{value:.4g}"
        elif isinstance(value, int):
            compact[key] = f"{value:,}"
        else:
            compact[key] = value
    return compact


class ProgressReporter:
    """Add phases and live metrics to one standard tqdm progress bar."""

    def __init__(
        self,
        label: str,
        *,
        enabled: bool | None = None,
        stream: TextIO | None = None,
        refresh_seconds: float = 0.5,
    ) -> None:
        if refresh_seconds <= 0:
            raise ValueError("refresh_seconds must be positive")
        self.label = label
        self.stream = sys.stderr if stream is None else stream
        self.enabled = progress_enabled(self.stream) if enabled is None else enabled
        self.refresh_seconds = refresh_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._current = 0
        self._bar = tqdm(
            total=None,
            desc=label,
            unit="it",
            file=self.stream,
            dynamic_ncols=True,
            disable=not self.enabled,
            mininterval=refresh_seconds,
            miniters=1,
        )
        self._finished = False
        if self.enabled:
            self._thread = threading.Thread(
                target=self._ticker,
                name="lm-from-zero-progress",
                daemon=True,
            )
            self._thread.start()

    def phase(
        self,
        name: str,
        *,
        total: int | None = None,
        current: int = 0,
        fields: Mapping[str, object] | None = None,
    ) -> None:
        """Start or switch to a named phase."""

        if total is not None and total <= 0:
            raise ValueError("progress total must be positive")
        if current < 0 or (total is not None and current > total):
            raise ValueError("progress current value is outside the total")
        with self._lock:
            self._bar.set_description(f"{self.label} | {name}", refresh=False)
            self._bar.reset(total=total)
            self._current = current
            if fields:
                self._bar.set_postfix(_postfix(fields), refresh=False)
            if current:
                self._bar.update(current)
            else:
                self._bar.refresh()

    def update(
        self,
        current: int,
        *,
        fields: Mapping[str, object] | None = None,
    ) -> None:
        """Set the current phase position and optional live metrics."""

        if current < 0:
            raise ValueError("progress current value must not be negative")
        with self._lock:
            delta = current - self._current
            if delta < 0:
                raise ValueError("progress cannot move backwards")
            self._current = current
            if fields:
                self._bar.set_postfix(_postfix(fields), refresh=False)
            if delta:
                self._bar.update(delta)
            else:
                self._bar.refresh()

    def advance(
        self,
        amount: int = 1,
        *,
        fields: Mapping[str, object] | None = None,
    ) -> None:
        """Advance the current phase by ``amount``."""

        if amount < 0:
            raise ValueError("progress advance must not be negative")
        self.update(self._current + amount, fields=fields)

    def finish(self, message: str = "complete") -> None:
        """Stop refreshing and leave a completed line in the terminal."""

        if self._finished:
            return
        with self._lock:
            self._finished = True
            self._bar.set_description(f"{self.label} | {message}", refresh=False)
            self._bar.refresh()
            self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(1.0, self.refresh_seconds * 2))
        with self._lock:
            self._bar.close()

    def _ticker(self) -> None:
        while not self._stop.wait(self.refresh_seconds):
            with self._lock:
                if self._finished:
                    return
                self._bar.refresh()
