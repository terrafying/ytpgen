"""Locate and configure the espeak-ng backend for phonemizer.

macOS/homebrew installs espeak-ng outside the default dylib search
path, so phonemizer's auto-discovery fails with 'RuntimeError: espeak
not installed'. This module finds the library + data dir explicitly
and returns a ready backend. All pipeline code must get its backend
through here instead of constructing EspeakBackend directly.
"""

from __future__ import annotations

import ctypes.util
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_HOMEBREW_PREFIXES = ("/opt/homebrew", "/usr/local")

_BACKEND: object | None = None
_INIT_FAILED = False


class _EspeakCompat:
    """Adapter: exposes upstream phonemizer's .process(list, strip=) API
    over the phonemizer-fork backend, which only has .phonemize()."""

    def __init__(self, backend) -> None:
        self._backend = backend

    def process(self, texts, strip: bool = False):
        return self._backend.phonemize(list(texts), strip=strip)

    def __getattr__(self, name):  # delegate the rest
        return getattr(self._backend, name)


def _find_library() -> str | None:
    import os

    env = os.environ.get("YTP_ESPEAK_LIBRARY")
    if env and Path(env).exists():
        return env
    for prefix in _HOMEBREW_PREFIXES:
        cand = Path(prefix) / "lib" / "libespeak-ng.dylib"
        if cand.exists():
            return str(cand)
    found = ctypes.util.find_library("espeak-ng")
    return found


def _find_data_path() -> str | None:
    import os

    env = os.environ.get("YTP_ESPEAK_DATA_PATH")
    if env and Path(env).is_dir():
        return env
    for prefix in _HOMEBREW_PREFIXES:
        cand = Path(prefix) / "share" / "espeak-ng-data"
        if cand.is_dir():
            return str(cand)
    return None


def get_espeak_backend(language: str = "en-us"):
    """Return a configured EspeakBackend, or None if espeak is unavailable.

    Result is cached; failures are remembered so we only warn once.
    """
    global _BACKEND, _INIT_FAILED
    if _BACKEND is not None or _INIT_FAILED:
        return _BACKEND
    try:
        from phonemizer.backend import EspeakBackend  # type: ignore[import]
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        lib = _find_library()
        data = _find_data_path()
        if lib:
            EspeakWrapper.set_library(lib)
        if data:
            EspeakWrapper.set_data_path(data)
        _BACKEND = _EspeakCompat(EspeakBackend(language))
        logger.info("espeak backend ready (lib=%s data=%s)", lib, data)
    except Exception as exc:  # pragma: no cover - depends on host
        logger.warning("espeak unavailable, IPA phonemization disabled: %s", exc)
        _INIT_FAILED = True
        _BACKEND = None
    return _BACKEND
