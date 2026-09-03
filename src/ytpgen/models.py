"""Shared data models for the ytpgen pipeline.

All modules build against these types. Keep them dependency-light
(pydantic only) so every stage can serialize them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceVideo(BaseModel):
    """A downloaded source video on disk."""

    video_id: str
    url: str = ""
    title: str = ""
    local_path: str
    duration_s: float = 0.0


class Scene(BaseModel):
    """A detected scene segment within a source video."""

    start_s: float
    end_s: float
    label: str = "shot"
    score: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)


class PhoneSpan(BaseModel):
    """One phone with its IPA symbol, optional ARPABET equivalent, timing."""

    ipa: str
    arpabet: str = ""
    start_s: float = 0.0
    end_s: float = 0.0


class Word(BaseModel):
    """A transcribed word with timing and phone-level detail."""

    word: str
    start_s: float
    end_s: float
    phonemes: str = ""  # space-separated ARPABET phones for the whole word
    phones: list[PhoneSpan] = Field(default_factory=list)
    source_id: int = 0

    @property
    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)


class Clip(BaseModel):
    """A resolvable unit (word or scene chunk) handed to the composer."""

    local_path: str
    start_s: float
    end_s: float
    source_id: int = 0
    text: str = ""
    kind: str = "word"  # word | scene | auto


class EDLEvent(BaseModel):
    """One cut in the edit decision list."""

    index: int = 0
    local_path: str
    start_s: float
    end_s: float
    text: str = ""
    reverse: bool = False
    speed: float = 1.0
    gain_db: float = 0.0
    effects: list[str] = Field(default_factory=list)


class EDL(BaseModel):
    """Edit decision list — the hand-editable artifact between auto and render."""

    version: int = 1
    fps: float = 30.0
    width: int = 1280
    height: int = 720
    seed: int | None = None
    events: list[EDLEvent] = Field(default_factory=list)
    notes: str = ""
