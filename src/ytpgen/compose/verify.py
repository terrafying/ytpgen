"""Comprehension check: re-transcribe a render and compare with what it should say."""

from __future__ import annotations

import re
from pathlib import Path

_model = None


def wer(ref: str, hyp: str) -> float:
    """Word error rate of *hyp* against *ref* (Levenshtein over words)."""
    norm = lambda t: re.sub(r"[^a-z0-9 ]", " ", re.sub(r"['\u2019]", "", t.lower())).split()  # apostrophes dropped, like the index
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    row = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, row[0] = row[0], i
        for j, hw in enumerate(h, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (rw != hw))
    return row[-1] / len(r)


def transcribe(cfg: dict, path: Path) -> str:
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(cfg.get("model_size", "base"), compute_type=cfg.get("compute_type", "int8"))
    segments, _ = _model.transcribe(str(path), language=cfg.get("language", "en"))
    return " ".join(s.text for s in segments)
