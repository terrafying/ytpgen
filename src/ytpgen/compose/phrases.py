"""Phrase inventory: runs of words a speaker actually said in one breath.

Derived on the fly from the words + scenes tables (no schema change):
consecutive words in one source form a phrase until a pause longer than
GAP_S, a scene cut, or MAX_S of speech. Whisper strips punctuation at index
time, so pauses stand in for sentence boundaries.
"""

from __future__ import annotations

import bisect
import sqlite3
from dataclasses import dataclass, field

GAP_S = 0.45
MAX_S = 9.0


@dataclass
class Phrase:
    id: int
    source_id: int
    local_path: str
    words: list[str]
    starts: list[float]
    ends: list[float]
    prev_end_s: float | None = None  # end of the word before the phrase (padding clamp)
    next_start_s: float | None = None
    scene_idx: int = -1
    extra: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(self.words)

    @property
    def start_s(self) -> float:
        return self.starts[0]

    @property
    def end_s(self) -> float:
        return self.ends[-1]

    def find(self, words: list[str]) -> tuple[int, int] | None:
        """[i, j) where *words* occur contiguously in this phrase, else None."""
        n = len(words)
        for i in range(len(self.words) - n + 1):
            if self.words[i : i + n] == words:
                return i, i + n
        return None


def derive_phrases(
    conn: sqlite3.Connection,
    source_ids: set[int] | None = None,
    gap_s: float = GAP_S,
    max_s: float = MAX_S,
) -> list[Phrase]:
    rows = conn.execute(
        "SELECT w.source_id, s.local_path, w.word, w.start_s, w.end_s"
        " FROM words w JOIN sources s ON w.source_id = s.id"
        " ORDER BY w.source_id, w.start_s"
    ).fetchall()
    cuts: dict[int, list[float]] = {}
    for sid, start in conn.execute("SELECT source_id, start_s FROM scenes ORDER BY source_id, start_s"):
        cuts.setdefault(sid, []).append(float(start))

    phrases: list[Phrase] = []
    cur: Phrase | None = None
    prev_end: float | None = None
    prev_sid = None
    for sid, path, word, start, end in rows:
        if source_ids and sid not in source_ids:
            continue
        if sid != prev_sid:
            prev_end = None
        scene = bisect.bisect_right(cuts.get(sid, []), start) - 1
        new = (
            cur is None
            or sid != cur.source_id
            or start - cur.end_s > gap_s
            or scene != cur.scene_idx
            or end - cur.start_s > max_s
        )
        if new:
            if cur is not None and cur.source_id == sid:
                cur.next_start_s = start
            cur = Phrase(len(phrases), sid, path, [], [], [], prev_end_s=prev_end, scene_idx=scene)
            phrases.append(cur)
        cur.words.append(word)
        cur.starts.append(float(start))
        cur.ends.append(float(end))
        prev_end, prev_sid = float(end), sid
    return phrases
