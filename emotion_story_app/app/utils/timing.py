"""Timing helpers."""

from __future__ import annotations

import time


class PeriodicTimer:
    """Simple periodic timer that keeps a steady interval across loop iterations."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = max(interval_s, 0.001)
        self._next_tick = time.monotonic()

    def wait_next(self) -> None:
        """Sleep until the next scheduled tick."""
        self._next_tick += self.interval_s
        now = time.monotonic()
        sleep_for = self._next_tick - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            self._next_tick = now
