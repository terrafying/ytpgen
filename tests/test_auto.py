"""Tests for ytpgen.compose.auto (AutoPooper — auto-YTP planner).

Builds a real Index on tmp_path sqlite and inserts fake sources, scenes,
and words rows via raw SQL (tables come from Index init). No media files,
no ffmpeg: the planner only reads the db and produces an EDL.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

# Run against the working tree, not the venv's installed snapshot.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ytpgen.compose.auto import AutoPooper, _stable_seed  # noqa: E402
from ytpgen.index import Index  # noqa: E402
from ytpgen.models import EDL, EDLEvent  # noqa: E402

VID = "autotest1"
LOCAL = "/media/auto.mp4"
CFG = {"compose": {"fps": 30, "width": 1280, "height": 720}}


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _insert_source(conn, video_id: str, local_path: str, duration: float) -> int:
    cur = conn.execute(
        "INSERT INTO sources (video_id, url, title, local_path, duration_s)"
        " VALUES (?, ?, ?, ?, ?)",
        (video_id, "http://example.com/x", video_id, local_path, duration),
    )
    return int(cur.lastrowid)


@pytest.fixture()
def db(tmp_path):
    """Index with one source: 10 scenes (2-5s) + ~30 words spread across."""
    idx = Index(tmp_path / "db.sqlite")
    conn = idx.conn
    sid = _insert_source(conn, VID, LOCAL, 120.0)
    t = 0.0
    for i in range(10):
        dur = 2.0 + (i % 4)  # 2,3,4,5,2,3,4,5,2,3 -> 33s total
        conn.execute(
            "INSERT INTO scenes (source_id, start_s, end_s, label, score)"
            " VALUES (?, ?, ?, ?, ?)",
            (sid, t, t + dur, "shot", 1.0 + (i % 5) * 0.5),
        )
        t += dur
    word_list = ["yeah", "what", "the", "heck", "wow", "dude", "no", "way", "boom", "lol"] * 3
    wt = 0.5
    for w in word_list:
        conn.execute(
            "INSERT INTO words (source_id, word, word_lower, start_s, end_s, phonemes)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (sid, w, w, wt, wt + 0.4, "AA"),
        )
        wt += 1.0
    idx.commit()
    yield idx
    idx.close()


# ----------------------------------------------------------------------
# plan(): shape and content
# ----------------------------------------------------------------------


def test_plan_returns_edl_with_events(db):
    edl = AutoPooper(CFG, db.conn).plan(VID, target_s=60)
    assert isinstance(edl, EDL)
    assert len(edl.events) >= 5


def test_duration_bounds(db):
    edl = AutoPooper(CFG, db.conn).plan(VID, target_s=60)
    total = sum(e.end_s - e.start_s for e in edl.events)
    assert 60 * 0.5 <= total <= 60 * 2


def test_determinism(db):
    a = AutoPooper(CFG, db.conn).plan(VID, target_s=60)
    b = AutoPooper(CFG, db.conn).plan(VID, target_s=60)
    assert a.seed is not None and a.seed == b.seed
    assert a.events == b.events
    assert a.notes == b.notes


def test_events_reference_source(db):
    edl = AutoPooper(CFG, db.conn).plan(VID, target_s=60)
    assert edl.events, "expected events"
    assert all(e.local_path == LOCAL for e in edl.events)


def test_stutter_groups_when_present(db):
    edl = AutoPooper(CFG, db.conn).plan(VID, target_s=60)
    for i, ev in enumerate(edl.events):
        if "stutter" not in ev.effects:
            continue
        group = edl.events[i : i + 3]
        assert len(group) == 3, f"stutter at {i} truncated"
        assert all(g.local_path == ev.local_path for g in group)
        mid = group[1]
        assert 0.0 < mid.end_s - mid.start_s <= 0.2


def test_edl_metadata(db):
    edl = AutoPooper(CFG, db.conn).plan(VID, target_s=60)
    assert edl.version == 1
    assert edl.fps == 30
    assert edl.width == 1280
    assert edl.height == 720
    assert edl.seed == _stable_seed(VID)
    assert edl.notes == f"auto: {VID} target={60}s"


# ----------------------------------------------------------------------
# Fallbacks and errors
# ----------------------------------------------------------------------


def test_scene_only_fallback(db):
    """No words -> scene-based remix still plans."""
    conn = db.conn
    sid = _insert_source(conn, "scenonly", "/media/scen.mp4", 40.0)
    for i in range(8):
        conn.execute(
            "INSERT INTO scenes (source_id, start_s, end_s, label, score)"
            " VALUES (?, ?, ?, ?, ?)",
            (sid, i * 4.0, i * 4.0 + 4.0, "shot", float(i)),
        )
    db.commit()
    edl = AutoPooper(CFG, db.conn).plan("scenonly", target_s=12)
    assert len(edl.events) >= 3
    assert all(e.local_path == "/media/scen.mp4" for e in edl.events)


def test_word_only_fallback(db):
    """No scenes -> word groups become remixable segments."""
    conn = db.conn
    sid = _insert_source(conn, "wordsonly", "/media/words.mp4", 40.0)
    for i in range(12):
        conn.execute(
            "INSERT INTO words (source_id, word, word_lower, start_s, end_s, phonemes)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (sid, f"w{i}", f"w{i}", i * 1.0, i * 1.0 + 0.5, "AA"),
        )
    db.commit()
    edl = AutoPooper(CFG, db.conn).plan("wordsonly", target_s=10)
    assert len(edl.events) >= 3
    assert all(e.local_path == "/media/words.mp4" for e in edl.events)


def test_source_without_analysis_raises(db):
    _insert_source(db.conn, "empty1", "/media/empty.mp4", 10.0)
    db.commit()
    with pytest.raises(ValueError, match="video not analyzed"):
        AutoPooper(CFG, db.conn).plan("empty1", target_s=10)


def test_missing_source_raises(db):
    with pytest.raises(ValueError, match="video not analyzed"):
        AutoPooper(CFG, db.conn).plan("ghost-video", target_s=10)


# ----------------------------------------------------------------------
# Helper methods (unit-testable)
# ----------------------------------------------------------------------


def test_mangle_event_value_ranges(db):
    ap = AutoPooper(CFG, db.conn)
    rng = random.Random(3)
    base = {"local_path": LOCAL, "start_s": 0.0, "end_s": 2.0, "text": "shot"}
    evs = [ap._mangle_event(base, rng) for _ in range(60)]
    assert all(isinstance(e, EDLEvent) for e in evs)
    assert all(e.speed in (0.75, 1.0, 1.25, 1.5) for e in evs)
    assert all(e.gain_db in (-3.0, 0.0, 2.0) for e in evs)
    assert all(e.local_path == LOCAL for e in evs)
    plain = [
        e
        for e in evs
        if e.speed == 1.0 and not e.reverse and e.gain_db == 0.0 and not e.effects
    ]
    assert plain, "expected some plain (unmangled) events"


def test_split_stutter_helper(db):
    ap = AutoPooper(CFG, db.conn)
    ev = EDLEvent(local_path=LOCAL, start_s=1.0, end_s=3.0, effects=["stutter"])
    parts = ap._split_stutter(ev, random.Random(7))
    assert len(parts) == 3
    assert all(p.local_path == LOCAL for p in parts)
    assert parts[1].end_s - parts[1].start_s <= 0.2  # middle is the blip
    assert "stutter" in parts[0].effects
    assert all("stutter" not in p.effects for p in parts[1:])
    # source coverage is preserved: first and last span the original range
    assert parts[0].start_s == pytest.approx(1.0)
    assert parts[2].end_s == pytest.approx(3.0)


def test_split_stutter_noop_without_flag(db):
    ap = AutoPooper(CFG, db.conn)
    ev = EDLEvent(local_path=LOCAL, start_s=1.0, end_s=3.0, effects=["zoom"])
    assert ap._split_stutter(ev, random.Random(7)) == [ev]
