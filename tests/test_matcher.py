"""Tests for ytpgen.compose.matcher (text-to-clips resolution).

Builds a real Index on tmp_path sqlite and inserts fake sources,
words, and phones rows directly. No network, no whisper, no espeak:
IPA inference is monkeypatched to return known ARPABET-able strings.
"""

from __future__ import annotations

import pytest

from ytpgen.compose.matcher import Matcher
from ytpgen.index import Index


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    """Index with one fake source and a few exact words."""
    idx = Index(tmp_path / "lib.db")
    sid = idx.add_source("vid1", "http://example.com/1", "One", "/media/one.mp4", 60.0)
    idx.add_word(sid, "hello", 1.0, 1.4, "HH AH L OW")
    idx.add_word(sid, "world", 2.0, 2.5, "W ER L D")
    idx.add_word(sid, "base", 3.0, 3.3, "B EY S")
    idx.add_word(sid, "line", 4.0, 4.4, "L AY N")
    idx.commit()
    yield idx
    idx.close()


def _no_ipa(token: str) -> None:
    """Sentinel IPA-inference stub: phonemizer 'unavailable'."""
    return None


@pytest.fixture()
def matcher_no_ipa(monkeypatch):
    """Matcher whose phone strategy always degrades (no IPA source)."""
    monkeypatch.setattr(Matcher, "_ipa_infer", staticmethod(_no_ipa))
    return Matcher({"compose": {"fps": 30, "width": 1280, "height": 720}})


# ----------------------------------------------------------------------
# Exact word resolution
# ----------------------------------------------------------------------


class TestExact:
    def test_exact_match_fields(self, db, matcher_no_ipa):
        clips = matcher_no_ipa.resolve_sentence("hello", db, mode="hybrid")
        assert len(clips) == 1
        clip = clips[0]
        assert clip.local_path == "/media/one.mp4"
        assert clip.start_s == pytest.approx(1.0)
        assert clip.end_s == pytest.approx(1.4)
        assert clip.kind == "word"
        assert clip.text == "hello"
        assert clip.source_id == db.get_source(1)["id"] or clip.source_id >= 1

    def test_exact_case_and_punctuation_normalized(self, db, matcher_no_ipa):
        clips = matcher_no_ipa.resolve_sentence("Hello, world!", db)
        assert [c.text for c in clips] == ["hello", "world"]
        assert all(c.kind == "word" for c in clips)

    def test_unresolvable_token_skipped(self, db, matcher_no_ipa, caplog):
        clips = matcher_no_ipa.resolve_sentence("hello xyzzy", db, mode="hybrid")
        assert [c.text for c in clips] == ["hello"]
        assert any("xyzzy" in rec.message for rec in caplog.records)


# ----------------------------------------------------------------------
# Mode filtering
# ----------------------------------------------------------------------


class TestModes:
    def test_mode_word_skips_phone_strategy(self, db, monkeypatch):
        calls: list[str] = []

        def spy(token):
            calls.append(token)
            return None

        monkeypatch.setattr(Matcher, "_ipa_infer", staticmethod(spy))
        m = Matcher({})
        clips = m.resolve_sentence("hello", db, mode="word")
        assert len(clips) == 1 and clips[0].kind == "word"
        assert calls == []  # phone strategy never invoked

    def test_mode_phone_ignores_exact_index(self, db, monkeypatch):
        # 'hello' IS indexed as a whole word, but mode='phone' must not
        # use the exact strategy — clips must come from phone spans.
        sid = 1
        word_rowid = db.conn.execute(
            "SELECT id FROM words WHERE word_lower = 'hello'"
        ).fetchone()["id"]
        for ipa, arpa, t0, t1 in [
            ("h", "HH", 1.0, 1.1), ("ə", "AH", 1.1, 1.2),
            ("l", "L", 1.2, 1.3), ("oʊ", "OW", 1.3, 1.4),
        ]:
            db.add_phone(sid, word_rowid, ipa, arpa, t0, t1)
        db.commit()
        monkeypatch.setattr(
            Matcher, "_ipa_infer", staticmethod(lambda token: "həˈloʊ")
        )
        m = Matcher({})
        clips = m.resolve_sentence("hello", db, mode="phone")
        assert clips
        assert all(c.kind == "phone" for c in clips)
        assert sum(c.end_s - c.start_s for c in clips) == pytest.approx(0.4)

    def test_unknown_mode_raises(self, db, matcher_no_ipa):
        with pytest.raises(ValueError):
            matcher_no_ipa.resolve_sentence("hello", db, mode="bogus")


# ----------------------------------------------------------------------
# Phone resolution
# ----------------------------------------------------------------------


class TestPhones:
    @pytest.fixture()
    def phone_db(self, tmp_path):
        """Index whose words have per-phone rows; no whole-word rows used."""
        idx = Index(tmp_path / "phones.db")
        sid = idx.add_source("vid2", "http://example.com/2", "Two", "/media/two.mp4", 30.0)
        rowid = idx.add_word(sid, "hello", 5.0, 5.4, "HH AH L OW")
        idx.add_phone(sid, rowid, "h", "HH", 5.0, 5.1)
        idx.add_phone(sid, rowid, "ə", "AH", 5.1, 5.2)
        idx.add_phone(sid, rowid, "l", "L", 5.2, 5.3)
        idx.add_phone(sid, rowid, "oʊ", "OW", 5.3, 5.4)
        idx.commit()
        yield idx
        idx.close()

    def test_whole_word_single_segment_cover(self, db, monkeypatch):
        # 'helo' is NOT exact-indexed; its IPA maps to ['HH','AH','L','OW'].
        # A single words_phones row carries the whole arpabet string, so
        # the greedy cover succeeds in ONE segment.
        monkeypatch.setattr(
            Matcher, "_ipa_infer", staticmethod(lambda token: "həˈloʊ")
        )
        word_rowid = db.conn.execute(
            "SELECT id FROM words WHERE word_lower = 'hello'"
        ).fetchone()["id"]
        db.add_phone(1, word_rowid, "həloʊ", "HH AH L OW", 1.0, 1.4)
        db.commit()
        m = Matcher({})
        clips = m.resolve_sentence("helo", db, mode="hybrid")
        assert len(clips) == 1
        assert clips[0].kind == "phone"
        assert clips[0].text == "HH AH L OW"

    def test_multi_segment_cover(self, phone_db, monkeypatch):
        # Same IPA; db only has 1-phone rows, so the greedy cover needs
        # four segments in order.
        monkeypatch.setattr(
            Matcher, "_ipa_infer", staticmethod(lambda token: "həˈloʊ")
        )
        m = Matcher({})
        clips = m.resolve_sentence("helo", phone_db, mode="hybrid")
        assert [c.text for c in clips] == ["HH", "AH", "L", "OW"]
        assert all(c.kind == "phone" for c in clips)
        assert all(c.local_path == "/media/two.mp4" for c in clips)
        assert clips[0].start_s == pytest.approx(5.0)
        assert clips[-1].end_s == pytest.approx(5.4)

    def test_phone_unavailable_falls_through_to_syllables(
        self, db, monkeypatch
    ):
        # 'baseline' is not indexed; phonemizer 'unavailable' -> None so
        # the phone strategy yields nothing and syllables fire.
        monkeypatch.setattr(Matcher, "_ipa_infer", staticmethod(_no_ipa))
        m = Matcher({})
        clips = m.resolve_sentence("baseline", db, mode="hybrid")
        assert [c.text for c in clips] == ["base", "line"]
        assert all(c.kind == "word" for c in clips)


# ----------------------------------------------------------------------
# Syllable fallback
# ----------------------------------------------------------------------


class TestSyllables:
    def test_baseline_splits_into_indexed_syllables(self, db, matcher_no_ipa):
        # Only 'base' and 'line' are indexed (not 'baseline').
        clips = matcher_no_ipa.resolve_sentence("baseline", db, mode="hybrid")
        assert [c.text for c in clips] == ["base", "line"]
        paths = [c.local_path for c in clips]
        assert paths == ["/media/one.mp4", "/media/one.mp4"]
        times = [(c.start_s, c.end_s) for c in clips]
        assert times == [(3.0, 3.3), (4.0, 4.4)]

    def test_short_word_never_splits(self, db, matcher_no_ipa):
        # 'an' has no vowel-cuts; resolves nowhere and is skipped.
        clips = matcher_no_ipa.resolve_sentence("an", db, mode="hybrid")
        assert clips == []

    def test_split_syllables_unit(self):
        assert Matcher._split_syllables("baseline") == ["base", "line"]
        assert Matcher._split_syllables("water") == ["wa", "ter"]
        assert Matcher._split_syllables("an") == ["an"]
        assert Matcher._split_syllables("xyzzy") == ["xy", "zzy"]


# ----------------------------------------------------------------------
# EDL assembly
# ----------------------------------------------------------------------


class TestEdl:
    def test_edl_from_clips_ordered_with_defaults(self, db, matcher_no_ipa):
        clips = matcher_no_ipa.resolve_sentence("hello world", db)
        edl = Matcher.edl_from_clips(clips, fps=29.97, width=1920, height=1080, seed=42)
        assert edl.fps == pytest.approx(29.97)
        assert edl.width == 1920
        assert edl.height == 1080
        assert edl.seed == 42
        assert len(edl.events) == len(clips)
        for i, (ev, clip) in enumerate(zip(edl.events, clips)):
            assert ev.index == i
            assert ev.local_path == clip.local_path
            assert ev.start_s == clip.start_s
            assert ev.end_s == clip.end_s
            assert ev.text == clip.text
            assert ev.reverse is False
            assert ev.speed == 1.0
            assert ev.gain_db == 0.0
            assert ev.effects == []

    def test_edl_from_empty_clips(self):
        edl = Matcher.edl_from_clips([], fps=30.0, width=1280, height=720)
        assert edl.events == []
        assert edl.fps == pytest.approx(30.0)
