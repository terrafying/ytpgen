"""SQLite index over sources, scenes, words, and phones.

Single canonical schema. All pipeline modules receive the raw
sqlite3 connection from Index.conn() and may also defensively
CREATE TABLE IF NOT EXISTS — the schema here is authoritative.
"""

from __future__ import annotations

import logging
import random
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT UNIQUE NOT NULL,
    url TEXT DEFAULT '',
    title TEXT DEFAULT '',
    local_path TEXT NOT NULL,
    duration_s REAL DEFAULT 0.0,
    added_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS scenes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES sources(id),
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    label TEXT DEFAULT 'shot',
    score REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_scenes_source ON scenes(source_id);
CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES sources(id),
    word TEXT NOT NULL,
    word_lower TEXT NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    phonemes TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_words_lower ON words(word_lower);
CREATE INDEX IF NOT EXISTS idx_words_phonemes ON words(phonemes);
CREATE TABLE IF NOT EXISTS words_phones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES sources(id),
    word_rowid INTEGER NOT NULL REFERENCES words(id),
    ipa TEXT NOT NULL,
    arpabet TEXT DEFAULT '',
    start_s REAL DEFAULT 0.0,
    end_s REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_phones_arpabet ON words_phones(arpabet);
CREATE INDEX IF NOT EXISTS idx_phones_source ON words_phones(source_id);
"""


class Index:
    """Thin data-access layer over the ytpgen library database."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    def add_source(
        self,
        video_id: str,
        url: str,
        title: str,
        local_path: str,
        duration_s: float = 0.0,
    ) -> int:
        """Insert (or update) a source row and return its id."""
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO sources (video_id, url, title, local_path, duration_s)"
            " VALUES (?, ?, ?, ?, ?)",
            (video_id, url, title, local_path, duration_s),
        )
        self._conn.commit()
        if cur.lastrowid and cur.lastrowid > 0:
            return int(cur.lastrowid)
        row = self._conn.execute(
            "SELECT id FROM sources WHERE video_id = ?", (video_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"source insert failed for {video_id!r}")
        return int(row["id"])

    def has_source(self, video_id: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM sources WHERE video_id = ?", (video_id,)
            ).fetchone()
            is not None
        )

    def get_source(self, source_id: int) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM sources WHERE id = ?", (source_id,)
        ).fetchone()

    def all_sources(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM sources ORDER BY added_at DESC"
        ).fetchall()

    def update_source_path(self, video_id: str, local_path: str) -> None:
        self._conn.execute(
            "UPDATE sources SET local_path = ? WHERE video_id = ?",
            (local_path, video_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Words / phones
    # ------------------------------------------------------------------

    def add_word(
        self,
        source_id: int,
        word: str,
        start_s: float,
        end_s: float,
        phonemes: str = "",
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO words (source_id, word, word_lower, start_s, end_s, phonemes)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (source_id, word, word.lower(), start_s, end_s, phonemes),
        )
        return int(cur.lastrowid)

    def add_phone(
        self,
        source_id: int,
        word_rowid: int,
        ipa: str,
        arpabet: str,
        start_s: float,
        end_s: float,
    ) -> None:
        self._conn.execute(
            "INSERT INTO words_phones (source_id, word_rowid, ipa, arpabet, start_s, end_s)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (source_id, word_rowid, ipa, arpabet, start_s, end_s),
        )

    def commit(self) -> None:
        self._conn.commit()

    def find_word(self, word: str, limit: int = 20) -> list[dict]:
        """All indexed occurrences of *word* (case-insensitive)."""
        rows = self._conn.execute(
            "SELECT w.id, w.source_id, s.local_path, w.word, w.start_s, w.end_s, w.phonemes"
            " FROM words w JOIN sources s ON w.source_id = s.id"
            " WHERE w.word_lower = ?",
            (word.lower().strip(".,!?;:\"'"),),
        ).fetchall()
        out = [dict(r) for r in rows]
        random.shuffle(out)
        return out[:limit]

    def find_phones(self, arpabet_seq: str, limit: int = 20) -> list[dict]:
        """Rows whose ARPABET sequence contains the given space-separated phones."""
        rows = self._conn.execute(
            "SELECT wp.source_id, s.local_path, wp.ipa, wp.arpabet,"
            " wp.start_s, wp.end_s, w.word"
            " FROM words_phones wp"
            " JOIN words w ON wp.word_rowid = w.id"
            " JOIN sources s ON wp.source_id = s.id"
            " WHERE wp.arpabet LIKE ?",
            (f"%{arpabet_seq}%",),
        ).fetchall()
        out = [dict(r) for r in rows]
        random.shuffle(out)
        return out[:limit]

    def all_words(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT word_lower FROM words ORDER BY word_lower"
        ).fetchall()
        return [r["word_lower"] for r in rows]

    # ------------------------------------------------------------------
    # Scenes
    # ------------------------------------------------------------------

    def add_scene(
        self, source_id: int, start_s: float, end_s: float, label: str, score: float
    ) -> None:
        self._conn.execute(
            "INSERT INTO scenes (source_id, start_s, end_s, label, score)"
            " VALUES (?, ?, ?, ?, ?)",
            (source_id, start_s, end_s, label, score),
        )

    def scenes_for(self, source_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT start_s, end_s, label, score FROM scenes"
            " WHERE source_id = ? ORDER BY start_s",
            (source_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        c = self._conn.execute
        return {
            "total_sources": c("SELECT COUNT(*) FROM sources").fetchone()[0],
            "total_scenes": c("SELECT COUNT(*) FROM scenes").fetchone()[0],
            "total_words": c("SELECT COUNT(*) FROM words").fetchone()[0],
            "unique_words": c(
                "SELECT COUNT(DISTINCT word_lower) FROM words"
            ).fetchone()[0],
            "total_phones": c("SELECT COUNT(*) FROM words_phones").fetchone()[0],
        }

    def close(self) -> None:
        self._conn.close()