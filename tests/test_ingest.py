"""Tests for ytpgen.acquire.ingest (bulk acquisition) — all offline, yt-dlp faked."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import ytpgen.acquire.ingest as ingest_mod
from ytpgen.acquire.ingest import Ingestor

V1, V2, V3 = "aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"
URL1 = f"https://www.youtube.com/watch?v={V1}"
URL2 = f"https://www.youtube.com/watch?v={V2}"
URL3 = f"https://youtu.be/{V3}"
NON_YT = "https://example.com/clip.mp4"
PLAYLIST = "https://www.youtube.com/playlist?list=PLfake"
BAD_URL = "https://www.youtube.com/watch?v=missing0000"

DEFAULT_EXPANSION: dict[str, list[str]] = {PLAYLIST: [URL1, URL2, URL1, URL2, URL3]}
FAKE_EXPANSION: dict[str, list[str]] = {}
FAKE_IDS: dict[str, str] = {URL1: V1, URL2: V2, URL3: V3, NON_YT: "extvid00000"}
FAKE_INFO_OVERRIDES: dict[str, dict] = {}


def _write_media(outtmpl: str, vid: str, ext: str = "mp4") -> None:
    path = Path(outtmpl.replace("%(id)s", vid).replace("%(ext)s", ext))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-media-bytes")


class FakeYoutubeDL:
    """Drop-in for yt_dlp.YoutubeDL: fakes flat expansion, metadata, download."""

    instances: list["FakeYoutubeDL"] = []

    def __init__(self, opts: dict | None = None) -> None:
        self.opts = opts or {}
        self.calls: list[tuple[str, bool]] = []
        FakeYoutubeDL.instances.append(self)

    def __enter__(self) -> "FakeYoutubeDL":
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def extract_info(self, url: str, download: bool = True) -> dict:
        self.calls.append((url, download))
        if self.opts.get("extract_flat"):
            entries = [{"url": u} for u in FAKE_EXPANSION.get(url, [url])]
            return {"_type": "playlist", "entries": entries}
        if url not in FAKE_IDS:
            raise RuntimeError(f"no such video: {url}")
        vid = FAKE_IDS[url]
        if download and "outtmpl" in self.opts:
            _write_media(self.opts["outtmpl"], vid)
        info = {"id": vid, "title": f"Title {vid}", "duration": 12.5, "ext": "mp4"}
        info.update(FAKE_INFO_OVERRIDES.get(url, {}))
        return info

    def prepare_filename(self, info: dict) -> str:
        outtmpl = self.opts.get("outtmpl", "%(id)s.%(ext)s")
        return outtmpl.replace("%(id)s", info["id"]).replace("%(ext)s", info.get("ext", "mp4"))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    cfg = {
        "workspace": str(tmp_path / "ws"),
        "download": {"format": "bestvideo[height<=720]+bestaudio/best[height<=720]"},
    }
    db = sqlite3.connect(str(tmp_path / "index.db"))
    db.execute(
        "CREATE TABLE sources ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT UNIQUE, url TEXT, title TEXT,"
        "local_path TEXT, duration_s REAL, added_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    FakeYoutubeDL.instances.clear()
    FAKE_EXPANSION.clear()
    FAKE_EXPANSION.update(DEFAULT_EXPANSION)
    FAKE_INFO_OVERRIDES.clear()
    monkeypatch.setattr(ingest_mod.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    yield SimpleNamespace(cfg=cfg, db=db, tmp=tmp_path, media=tmp_path / "ws" / "media")
    db.close()


def _all_downloads() -> list[bool]:
    return [d for inst in FakeYoutubeDL.instances for _, d in inst.calls]


def test_resolve_single_urls_pass_through_without_ytdlp(env):
    ing = Ingestor(env.cfg, env.db)
    urls = [URL1, URL3, NON_YT]
    assert ing.resolve_targets(urls) == urls
    assert FakeYoutubeDL.instances == []  # yt-dlp never touched


def test_resolve_dedupes_repeated_urls(env):
    ing = Ingestor(env.cfg, env.db)
    assert ing.resolve_targets([URL1, URL1, URL2, URL1]) == [URL1, URL2]


def test_resolve_expands_playlist_dedupes_and_caps(env):
    FAKE_EXPANSION[PLAYLIST] = [f"https://www.youtube.com/watch?v=v{i:05d}" for i in range(150)]
    FAKE_EXPANSION[PLAYLIST] += FAKE_EXPANSION[PLAYLIST][:5]  # duplicates
    ing = Ingestor(env.cfg, env.db)
    out = ing.resolve_targets([PLAYLIST])
    assert out == FAKE_EXPANSION[PLAYLIST][:100]
    assert len(out) == 100 and len(set(out)) == 100
    assert all(not d for d in _all_downloads())  # resolve never downloads


def test_download_skips_existing_file(env):
    target = env.media / f"{V1}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"already-here")
    ing = Ingestor(env.cfg, env.db)
    video = ing.download(URL1)
    assert video.video_id == V1
    assert video.local_path == str(target)
    assert video.title == f"Title {V1}"
    assert video.duration_s == pytest.approx(12.5)
    assert target.read_bytes() == b"already-here"  # untouched
    assert _all_downloads() == [False]  # metadata probe only, no download


def test_download_downloads_when_missing(env):
    ing = Ingestor(env.cfg, env.db)
    video = ing.download(URL2)
    assert video.video_id == V2
    assert video.local_path == str(env.media / f"{V2}.mp4")
    assert Path(video.local_path).exists()
    assert _all_downloads() == [False, True]  # probe, then download


def test_download_duration_falls_back_to_ffprobe(env, monkeypatch):
    FAKE_INFO_OVERRIDES[URL3] = {"duration": None}
    monkeypatch.setattr(ingest_mod, "_ffprobe_duration", lambda p: 42.0)
    ing = Ingestor(env.cfg, env.db)
    video = ing.download(URL3)
    assert video.duration_s == pytest.approx(42.0)


def test_ingest_counts_and_db_rows(env):
    ing = Ingestor(env.cfg, env.db)
    result = ing.ingest(PLAYLIST)
    assert result == {"added": 3, "skipped": 0, "downloads_failed": 0, "video_ids": [V1, V2, V3]}
    rows = env.db.execute(
        "SELECT video_id, url, title, local_path, duration_s, added_at"
        " FROM sources ORDER BY video_id"
    ).fetchall()
    expected_urls = {V1: URL1, V2: URL2, V3: URL3}
    assert len(rows) == 3
    for vid, url, title, local_path, duration, added_at in rows:
        assert url == expected_urls[vid]
        assert title == f"Title {vid}"
        assert local_path == str(env.media / f"{vid}.mp4")
        assert duration == pytest.approx(12.5)
        assert added_at is not None
    again = ing.ingest(PLAYLIST)
    assert again == {"added": 0, "skipped": 3, "downloads_failed": 0, "video_ids": []}


def test_ingest_skips_indexed_and_counts_mixed(env):
    ing = Ingestor(env.cfg, env.db)
    first = ing.ingest(URL2)
    assert first["added"] == 1
    second = ing.ingest(PLAYLIST)
    assert second["added"] == 2
    assert second["skipped"] == 1
    assert second["video_ids"] == [V1, V3]


def test_ingest_force_reindexes_without_duplicating_rows(env):
    ing = Ingestor(env.cfg, env.db)
    ing.ingest(PLAYLIST)
    forced = ing.ingest(PLAYLIST, force=True)
    assert forced["added"] == 3
    assert forced["skipped"] == 0
    assert env.db.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 3


def test_ingest_counts_failures_without_crashing(env):
    ing = Ingestor(env.cfg, env.db)
    result = ing.ingest(BAD_URL)
    assert result == {"added": 0, "skipped": 0, "downloads_failed": 1, "video_ids": []}


def test_ingest_creates_missing_sources_table(tmp_path, monkeypatch):
    db = sqlite3.connect(str(tmp_path / "fresh.db"))  # no sources table
    FakeYoutubeDL.instances.clear()
    monkeypatch.setattr(ingest_mod.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    cfg = {"workspace": str(tmp_path / "ws"), "download": {"format": "best"}}
    try:
        ing = Ingestor(cfg, db)
        assert ing.ingest(URL1)["added"] == 1
        assert db.execute("SELECT video_id FROM sources").fetchall() == [(V1,)]
    finally:
        db.close()
