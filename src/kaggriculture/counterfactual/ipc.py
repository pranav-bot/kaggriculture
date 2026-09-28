"""Subprocess IPC manager for kagg serve."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from kaggsim.serve import Serve, ServeError


class KaggServeProcess:
    """Manages lifecycle and pipe communication with the native Rust kagg serve process."""

    def __init__(self, binary_path: Path | str | None = None):
        self.binary_path = str(binary_path) if binary_path is not None else None
        self._serve: Serve | None = None
        self._ensure()

    def _ensure(self):
        if self._serve is None:
            self._serve = Serve(kagg=self.binary_path)
        self._serve._ensure()

    @property
    def serve(self) -> Serve:
        self._ensure()
        return self._serve

    def is_healthy(self) -> bool:
        """Check whether the underlying subprocess is responsive."""
        try:
            self._ensure()
            return self._serve.proc is not None and self._serve.proc.poll() is None
        except Exception:
            return False

    def ping(self) -> bool:
        """Send a lightweight reset probe to verify protocol responsiveness."""
        try:
            res = self.serve.reset(seed=42)
            return isinstance(res, dict) and "step" in res
        except Exception:
            return False

    def close(self):
        """Terminate the server process and release pipes."""
        if self._serve is not None:
            self._serve.close()
            self._serve = None

    def __enter__(self):
        self._ensure()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
