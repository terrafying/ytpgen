"""Tests for ytpgen.analyze.segment — all offline.

Scene detection runs REAL ffmpeg on a lavfi-generated clip (no network);
whisper is faked; phonemizer is faked or forced into its degrade path.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

import ytpgen.analyze.segment as segment_mod
from ytpgen.analyze.phonemes.ipa_arpabet import IPA_TO_arpabet, ipa_to_arpabet
from ytpgen.analyze.segment import Segmenter, merge_boundaries
from ytpgen.config import load_config
from ytpgen.models import PhoneSpan, Scene, SourceVideo, Word

# ---------------------------------------------------------------- boundaries


def test_merge_drops_cuts_closer_than_min_len():
    assert merge_boundaries([0.5, 0.8, 5.0, 5.4], 1.0, 10.0) == [0.0, 5.0, 10.0]


def test_merge_always_bookends():
    assert merge_boundaries([], 1.0, 7.5) == [0.0, 7.5]


def test_merge_dedupes_and_sorts():
    assert merge_boundaries([9.0, 2.0, 2.0, 1.0], 0.5, 10.0) == [
        0.0, 1.0, 2.0, 9.0, 10.0,
    ]


def test_merge_drops_cut_too_close_to_duration():
    assert merge_boundaries([9.5], 1.0, 10.0) == [0.0, 10.0]


def test_merge_ignores_out_of_range_times():
    assert merge_boundaries([-3.0, 11.0], 1.0, 10.0) == [0.0, 10.0]


def test_merge_zero_min_len_keeps_all():
    assert merge_boundaries([1.0, 1.5, 2.0], 0.0, 3.0) == [0.0, 1.0, 1.5, 2.0, 3.0]


# ---------------------------------------------------------------- IPA table

_ALL_ARPABET = (
    "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW "
    "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z"
).split()

_VOWELS = {"ɑ": "AA", "æ": "AE", "ʌ": "AH", "ɔ": "AO", "ʊ": "UH", "u": "UW",
           "ɪ": "IH", "i": "IY", "ɛ": "EH", "ɝ": "ER", "ə": "AH"}
_DIPHONGS = {"aʊ": "AW", "aɪ": "AY", "eɪ": "EY", "oʊ": "OW", "ɔɪ": "OY"}
_CONSONANTS = {"p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "ɡ": "G",
               "f": "F", "v": "V", "θ": "TH", "ð": "DH", "s": "S", "z": "Z",
               "ʃ": "SH", "ʒ": "ZH", "tʃ": "CH", "dʒ": "JH", "h": "HH",
               "m": "M", "n": "N", "ŋ": "NG", "l": "L", "r": "R",
               "w": "W", "j": "Y"}


def test_vowel_mappings():
    for ipa, arp in _VOWELS.items():
        assert IPA_TO_arpabet[ipa] == arp, ipa


def test_diphthong_mappings():
    for ipa, arp in _DIPHONGS.items():
        assert IPA_TO_arpabet[ipa] == arp, ipa


def test_consonant_mappings():
    for ipa, arp in _CONSONANTS.items():
        assert IPA_TO_arpabet[ipa] == arp, ipa


def test_glottal_stop_maps_to_skip():
    assert IPA_TO_arpabet["ʔ"] == ""


def test_full_arpabet_coverage():
    missing = set(_ALL_ARPABET) - set(IPA_TO_arpabet.values())
    assert not missing, f"ARPABET phones missing from table: {sorted(missing)}"


def test_greedy_multichar_units_win():
    assert ipa_to_arpabet("aɪ") == ["AY"]
    assert ipa_to_arpabet("oʊ") == ["OW"]
    assert ipa_to_arpabet("tʃ") == ["CH"]
    assert ipa_to_arpabet("dʒ") == ["JH"]


def test_hello_ipa():
    assert ipa_to_arpabet("həloʊ") == ["HH", "AH", "L", "OW"]


def test_stress_and_length_marks_stripped():
    assert ipa_to_arpabet("ˈwɜːld") == ["W", "ER", "L", "D"]


def test_unknown_chars_and_glottals_skipped():
    assert ipa_to_arpabet("hə?loʊ") == ["HH", "AH", "L", "OW"]
    assert ipa_to_arpabet("bʌʔɾɚ") == ["B", "AH", "T", "ER"]


# ---------------------------------------------------------------- Segmenter


@pytest.fixture()
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(tmp_path / "test.db")
    conn.execute(
        "CREATE TABLE sources (id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT UNIQUE)"
    )
    yield conn
    conn.close()


def make_segmenter(conn: sqlite3.Connection) -> Segmenter:
    return Segmenter(load_config(), conn)


def _fake_model(segments_words: list[list[tuple[str, float, float]]]) -> SimpleNamespace:
    segs = [
        SimpleNamespace(
            words=[SimpleNamespace(word=w, start=s, end=e) for w, s, e in ws]
        )
        for ws in segments_words
    ]

    def transcribe(path, word_timestamps=False, language=None, **_):
        assert word_timestamps is True
        assert language == "en"
        return iter(segs), SimpleNamespace(language="en")

    return SimpleNamespace(transcribe=transcribe)


def test_transcribe_builds_clean_words_with_phones(db, monkeypatch):
    seg = make_segmenter(db)
    fake = _fake_model(
        [[("Hello,", 0.0, 0.5), ("world!", 0.5, 1.0)], [("it's", 1.0, 1.4)]]
    )
    monkeypatch.setattr(seg, "_load_model", lambda: fake)
    ipa = {"hello": "həloʊ", "world": "ˈwɜːld", "its": "ɪts"}
    monkeypatch.setattr(seg, "_phonemize", lambda tokens: [ipa.get(t, "") for t in tokens])

    words = seg.transcribe(Path("/nonexistent.mp4"), source_id=7)

    assert [w.word for w in words] == ["hello", "world", "its"]
    assert all(w.source_id == 7 for w in words)
    assert words[0].phonemes == "HH AH L OW"
    assert [p.arpabet for p in words[0].phones] == ["HH", "AH", "L", "OW"]
    assert [p.ipa for p in words[0].phones] == ["h", "ə", "l", "oʊ"]
    # even split of [0.0, 0.5] across 4 phones
    assert words[0].phones[0].start_s == pytest.approx(0.0)
    assert words[0].phones[1].start_s == pytest.approx(0.125)
    assert words[0].phones[-1].end_s == pytest.approx(0.5)
    assert words[1].phonemes == "W ER L D"


def test_transcribe_degrades_without_phonemizer(db, monkeypatch):
    seg = make_segmenter(db)
    seg._phonemizer_checked = True
    seg._phonemize_fn = None
    monkeypatch.setattr(seg, "_load_model", lambda: _fake_model([[("Hey", 0.0, 0.4)]]))

    words = seg.transcribe(Path("/nonexistent.mp4"), source_id=1)

    assert words[0].word == "hey"
    assert words[0].phonemes == ""
    assert words[0].phones == []


def test_phonemize_live_backend_or_graceful_degrade(db):
    seg = make_segmenter(db)
    out = seg._phonemize(["hello"])
    if out and out[0]:
        assert out[0].startswith("h")  # real espeak reached via library rescue
    else:
        assert out == [""]  # degrade path: empty IPA, no crash


def test_detect_scenes_missing_ffmpeg_raises(db, monkeypatch):
    seg = make_segmenter(db)
    monkeypatch.setattr(segment_mod.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        seg.detect_scenes(Path("/nonexistent.mp4"))


def test_detect_scenes_real_ffmpeg(db, tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg, "ffmpeg expected on this machine"
    clip = tmp_path / "cut.mp4"
    subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-y",
         "-f", "lavfi", "-i", "color=c=red:size=320x240:rate=10:duration=2",
         "-f", "lavfi", "-i", "color=c=blue:size=320x240:rate=10:duration=2",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True, capture_output=True, text=True,
    )
    scenes = make_segmenter(db).detect_scenes(clip)
    # red [0,2) then blue [2,4): one real cut at t=2 -> two scenes.
    assert len(scenes) == 2
    assert scenes[0].start_s == pytest.approx(0.0)
    assert scenes[0].end_s == pytest.approx(2.0, abs=0.1)
    assert scenes[1].end_s == pytest.approx(4.0, abs=0.1)
    # The scene-change score lands on the last frame of the outgoing segment.
    assert scenes[0].score > 0.2
    assert scenes[1].score == 0.0


def test_analyze_writes_db(db, monkeypatch):
    conn = db
    conn.execute("INSERT INTO sources (video_id) VALUES ('abc')")
    conn.commit()
    seg = make_segmenter(conn)
    monkeypatch.setattr(
        seg, "detect_scenes",
        lambda path: [Scene(start_s=0.0, end_s=2.0, score=0.4),
                      Scene(start_s=2.0, end_s=4.0, score=0.3)],
    )
    word = Word(
        word="hello", start_s=0.0, end_s=0.5, source_id=1, phonemes="HH AH L OW",
        phones=[PhoneSpan(ipa="h", arpabet="HH", start_s=0.0, end_s=0.125),
                PhoneSpan(ipa="ə", arpabet="AH", start_s=0.125, end_s=0.25)],
    )
    punct = Word(word=",", start_s=0.5, end_s=0.6, source_id=1)
    monkeypatch.setattr(seg, "transcribe", lambda path, source_id: [word, punct])

    result = seg.analyze(SourceVideo(video_id="abc", local_path="/x.mp4"), conn)

    assert result == {"scenes": 2, "words": 2}
    assert conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0] == 2
    assert conn.execute(
        "SELECT score, label FROM scenes ORDER BY start_s LIMIT 1"
    ).fetchone() == (0.4, "shot")
    word_row = conn.execute(
        "SELECT rowid, word, phonemes FROM words"
    ).fetchall()
    assert [(r[1], r[2]) for r in word_row] == [("hello", "HH AH L OW"), (",", "")]
    phones = conn.execute(
        "SELECT wp.word_rowid, wp.ipa, wp.arpabet, wp.start_s FROM words_phones wp"
        " JOIN words w ON w.rowid = wp.word_rowid WHERE w.word = 'hello'"
    ).fetchall()
    assert [(p[1], p[2]) for p in phones] == [("h", "HH"), ("ə", "AH")]
    assert phones[0][3] == pytest.approx(0.0)
    assert conn.execute(
        "SELECT COUNT(*) FROM words_phones wp JOIN words w"
        " ON w.rowid = wp.word_rowid WHERE w.word = ','"
    ).fetchone()[0] == 0


def test_analyze_swallows_bad_video(db, monkeypatch):
    seg = make_segmenter(db)

    def boom(path):
        raise RuntimeError("bad file")

    monkeypatch.setattr(seg, "detect_scenes", boom)
    result = seg.analyze(SourceVideo(video_id="nope", local_path="/x.mp4"), db)
    assert result["scenes"] == 0 and result["words"] == 0
    assert "bad file" in result["error"]
