"""Tests for the local Flask web UI (ytpgen.web.app).

All heavy operations (ingest / analyze / render / auto-plan) are
monkeypatched with fakes so tests run offline and instantly. Media
serving uses a real tiny clip via make_test_clip.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ytpgen.compose.edl import load_edl, save_edl
from ytpgen.index import Index
from ytpgen.models import EDL, EDLEvent

sys.path.insert(0, str(Path(__file__).parent))
from media_helper import make_test_clip  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def env(tmp_path):
    """Fresh workspace + app + db with one source row, no analysis."""
    ws = tmp_path / "ws"
    cfg = {
        "workspace": str(ws),
        "compose": {"fps": 30, "width": 640, "height": 360},
    }
    from ytpgen.web.app import create_app

    app = create_app(cfg)
    app.config["TESTING"] = True
    db = Index(ws / "index.db")
    sid = db.add_source(
        "v1", "https://youtu.be/v1", "Test Source One",
        str(ws / "media" / "v1.mp4"), 120.0,
    )
    db.commit()
    return SimpleNamespace(
        app=app, client=app.test_client(), db=db, ws=ws, cfg=cfg, sid=sid,
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def test_dashboard_200_stats_and_sources(env):
    env.db.add_scene(env.sid, 0.0, 5.0, "shot", 0.5)
    env.db.add_word(env.sid, "poop", 1.0, 2.0)
    env.db.commit()
    r = env.client.get("/")
    assert r.status_code == 200
    body = r.data.decode()
    assert "total_sources" in body
    assert "unique_words" in body
    assert "Test Source One" in body
    assert "v1" in body
    assert "Ingest" in body  # ingest form present


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def _wait_job(env: SimpleNamespace, job_id: str, timeout: float = 10.0) -> dict:
    """Poll the app's JobRunner until the job settles; return its status dict."""
    import time

    runner = env.app.config["JOBS"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = runner.status(job_id)
        assert st is not None, f"unknown job {job_id}"
        if st["status"] in ("done", "error"):
            return st
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not settle within {timeout}s")


def test_ingest_post_mocked_redirects_with_flash(env, monkeypatch):
    calls = []

    class FakeIngestor:
        def __init__(self, cfg, conn):
            pass

        def ingest(self, target, force=False):
            calls.append((target, force))
            return {"added": 2, "skipped": 1, "downloads_failed": 0, "video_ids": []}

    monkeypatch.setattr("ytpgen.acquire.ingest.Ingestor", FakeIngestor)
    r = env.client.post(
        "/ingest",
        data={"urls": "https://youtu.be/aaaaaaaaaaa\nhttps://youtu.be/bbbbbbbbbbb"},
    )
    assert r.status_code == 302
    loc = r.headers["Location"]
    assert "/jobs/" in loc
    st = _wait_job(env, loc.rsplit("/", 1)[-1])
    assert st["status"] == "done"
    assert st["result"] == {"added": 4, "skipped": 2, "failed": 0}
    # one Ingestor call per URL line
    assert [c[0] for c in calls] == [
        "https://youtu.be/aaaaaaaaaaa",
        "https://youtu.be/bbbbbbbbbbb",
    ]


def test_ingest_error_flashes_message(env, monkeypatch):
    class BoomIngestor:
        def __init__(self, cfg, conn):
            pass

        def ingest(self, target, force=False):
            raise RuntimeError("boom")

    monkeypatch.setattr("ytpgen.acquire.ingest.Ingestor", BoomIngestor)
    r = env.client.post("/ingest", data={"urls": "https://youtu.be/aaaaaaaaaaa"})
    st = _wait_job(env, r.headers["Location"].rsplit("/", 1)[-1])
    assert st["status"] == "error"
    assert "boom" in st["error"]


def test_ingest_empty_urls_flashes(env):
    r = env.client.post("/ingest", data={"urls": "  \n "}, follow_redirects=True)
    assert r.status_code == 200
    assert b"No URLs" in r.data


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------


def test_analyze_runs_for_unanalyzed_sources(env, monkeypatch):
    calls = []

    class FakeSegmenter:
        def __init__(self, cfg, conn):
            pass

        def analyze(self, video, db):
            calls.append(video.video_id)
            return {"scenes": 3, "words": 7}

    monkeypatch.setattr("ytpgen.analyze.segment.Segmenter", FakeSegmenter)
    r = env.client.post("/analyze")
    assert r.status_code == 302
    st = _wait_job(env, r.headers["Location"].rsplit("/", 1)[-1])
    assert st["status"] == "done"
    assert st["result"] == {"analyzed": 1, "errors": 0}
    assert calls == ["v1"]


def test_analyze_source_selection_filters(env, monkeypatch):
    calls = []
    env.db.add_source("v2", "https://youtu.be/v2", "Two",
                      str(env.ws / "media" / "v2.mp4"), 60.0)
    env.db.commit()

    class FakeSegmenter:
        def __init__(self, cfg, conn):
            pass

        def analyze(self, video, db):
            calls.append(video.video_id)
            return {"scenes": 3, "words": 7}

    monkeypatch.setattr("ytpgen.analyze.segment.Segmenter", FakeSegmenter)
    other = env.db.conn.execute(
        "SELECT id FROM sources WHERE video_id = 'v2'").fetchone()["id"]
    r = env.client.post("/analyze", data={"source_ids": str(other)})
    st = _wait_job(env, r.headers["Location"].rsplit("/", 1)[-1])
    assert st["status"] == "done"
    assert calls == ["v2"]  # only the checked source analyzed


def test_analyze_missing_module_friendly_error(env, monkeypatch):
    monkeypatch.setitem(sys.modules, "ytpgen.analyze.segment", None)
    r = env.client.post("/analyze")
    st = _wait_job(env, r.headers["Location"].rsplit("/", 1)[-1])
    assert st["status"] == "error"


# ---------------------------------------------------------------------------
# Words
# ---------------------------------------------------------------------------


def test_words_page_lists_words(env):
    env.db.add_word(env.sid, "Poop", 1.0, 2.0)
    env.db.add_word(env.sid, "hello", 3.0, 4.0)
    env.db.commit()
    r = env.client.get("/words")
    assert r.status_code == 200
    assert b"poop" in r.data
    assert b"hello" in r.data


def test_words_pagination(env):
    for i in range(150):
        env.db.add_word(env.sid, f"w{i:03d}", 0.0, 1.0)
    env.db.commit()
    page1 = env.client.get("/words").data.decode()
    page2 = env.client.get("/words?page=2").data.decode()
    assert "w000" in page1 and "w149" not in page1
    assert "w149" in page2


def test_word_detail_shows_occurrences_and_preview(env):
    make_test_clip(env.ws / "media" / "v1.mp4", 1.0)
    env.db.add_word(env.sid, "Poop", 1.0, 2.0)
    env.db.commit()
    r = env.client.get("/words/poop")
    assert r.status_code == 200
    body = r.data.decode()
    assert "Test Source One" in body
    assert "#t=1.00,2.00" in body
    assert "<video" in body


def test_word_detail_empty(env):
    r = env.client.get("/words/zzz")
    assert r.status_code == 200
    assert b"No occurrences" in r.data


# ---------------------------------------------------------------------------
# EDL list / edit / render
# ---------------------------------------------------------------------------


def _make_edl(env: SimpleNamespace, name: str = "demo") -> EDL:
    clip = env.ws / "media" / "v1.mp4"
    make_test_clip(clip, 1.0)
    edl = EDL(
        events=[EDLEvent(index=0, local_path=str(clip), start_s=0.0,
                         end_s=0.5, text="original")]
    )
    save_edl(edl, env.ws / "outputs" / f"{name}.edl.json")
    return edl


def test_edl_list_shows_files(env):
    _make_edl(env)
    r = env.client.get("/edl")
    assert r.status_code == 200
    assert b"demo.edl.json" in r.data
    assert b"1" in r.data  # event count


def test_edl_roundtrip_edit_and_save(env):
    edl = _make_edl(env)
    r = env.client.get("/edl/demo.edl.json")
    assert r.status_code == 200
    assert b"original" in r.data

    dumped = edl.model_dump()
    dumped["events"][0]["text"] = "hello"
    r = env.client.post("/edl/demo.edl.json", data={"content": json.dumps(dumped)})
    assert r.status_code == 302

    r = env.client.get("/edl/demo.edl.json")
    assert b"hello" in r.data
    assert load_edl(env.ws / "outputs" / "demo.edl.json").events[0].text == "hello"


def test_edl_save_invalid_json_rejected(env):
    _make_edl(env)
    r = env.client.post("/edl/demo.edl.json",
                        data={"content": "{not valid json"},
                        follow_redirects=True)
    assert r.status_code == 200
    assert b"Invalid EDL" in r.data
    # original file untouched
    assert load_edl(env.ws / "outputs" / "demo.edl.json").events[0].text == "original"


def test_edl_render_mocked(env, monkeypatch):
    _make_edl(env)

    class FakeRenderer:
        def __init__(self, cfg):
            pass

        def render(self, edl, output_path):
            Path(output_path).write_bytes(b"\x00\x00")
            return Path(output_path)

    monkeypatch.setattr("ytpgen.compose.renderer.Renderer", FakeRenderer)
    r = env.client.post("/edl/demo.edl.json/render")
    assert r.status_code == 302
    st = _wait_job(env, r.headers["Location"].rsplit("/", 1)[-1])
    assert st["status"] == "done"
    assert (env.ws / "outputs" / "demo.mp4").exists()


def test_edl_detail_missing_404(env):
    r = env.client.get("/edl/nope.edl.json")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Generate / Auto
# ---------------------------------------------------------------------------


def test_generate_creates_and_renders(env, monkeypatch):
    env.db.add_word(env.sid, "poop", 1.0, 2.0)
    env.db.commit()

    class FakeRenderer:
        def __init__(self, cfg):
            pass

        def render(self, edl, output_path):
            Path(output_path).write_bytes(b"\x00\x00")
            return Path(output_path)

    monkeypatch.setattr("ytpgen.compose.renderer.Renderer", FakeRenderer)
    r = env.client.post("/generate", data={"text": "poop", "mode": "word"})
    assert r.status_code == 302
    st = _wait_job(env, r.headers["Location"].rsplit("/", 1)[-1])
    assert st["status"] == "done"
    assert st["result"]["edl"] == "gen_poop.edl.json"
    edl_path = env.ws / "outputs" / "gen_poop.edl.json"
    assert edl_path.exists()
    saved = load_edl(edl_path)
    assert saved.events[0].text == "poop"
    assert (env.ws / "outputs" / "gen_poop.mp4").exists()
    r = env.client.get("/edl/gen_poop.edl.json")
    assert r.status_code == 200


def test_generate_no_clips_job_errors(env):
    r = env.client.post("/generate", data={"text": "qqqzzz", "mode": "word"})
    st = _wait_job(env, r.headers["Location"].rsplit("/", 1)[-1])
    assert st["status"] == "error"
    assert "no clips matched" in st["error"].lower()


def test_auto_plans_and_renders(env, monkeypatch):
    calls = []

    class FakeAutoPooper:
        def __init__(self, cfg, conn):
            pass

        def plan(self, video_id, target_s=60):
            calls.append((video_id, target_s))
            return EDL(events=[EDLEvent(index=0, local_path="x.mp4",
                                        start_s=0.0, end_s=1.0, text="auto")])

    class FakeRenderer:
        def __init__(self, cfg):
            pass

        def render(self, edl, output_path):
            Path(output_path).write_bytes(b"\x00\x00")
            return Path(output_path)

    monkeypatch.setattr("ytpgen.compose.auto.AutoPooper", FakeAutoPooper)
    monkeypatch.setattr("ytpgen.compose.renderer.Renderer", FakeRenderer)
    r = env.client.post("/auto", data={"video_id": "v1", "target": "30"})
    assert r.status_code == 302
    st = _wait_job(env, r.headers["Location"].rsplit("/", 1)[-1])
    assert st["status"] == "done"
    assert calls == [("v1", 30)]
    assert (env.ws / "outputs" / "auto_v1.edl.json").exists()
    assert (env.ws / "outputs" / "auto_v1.mp4").exists()


# ---------------------------------------------------------------------------
# Media serving
# ---------------------------------------------------------------------------


def test_media_serves_file_from_workspace(env):
    make_test_clip(env.ws / "media" / "v1.mp4", 1.0)
    r = env.client.get("/media/media/v1.mp4")
    assert r.status_code == 200
    assert r.content_type.startswith("video/")
    assert b"ftyp" in r.data[:32]


def test_media_blocks_traversal(env):
    r1 = env.client.get("/media/../index.db")
    r2 = env.client.get("/media/%2e%2e/index.db")
    assert r1.status_code == 404
    assert r2.status_code == 404


# ---------------------------------------------------------------------------
# Word preferences + reroll
# ---------------------------------------------------------------------------


def test_word_pref_set_and_clear(env, monkeypatch):
    env.db.add_word(env.sid, "poop", 1.0, 2.0)
    env.db.commit()
    # dashboard lists prefs after set
    r = env.client.post("/words/poop/pref", data={"source_id": str(env.sid),
                                                  "action": "set"},
                        follow_redirects=True)
    assert r.status_code == 200
    assert env.db.word_prefs().get("poop", {}).get("source_id") == env.sid

    # find_word ranks pref source first when multiple sources have the word
    other = env.db.add_source("v2", "u", "Two", "/tmp/v2.mp4", 10.0)
    env.db.add_word(other, "poop", 5.0, 6.0)
    env.db.commit()
    hits = env.db.find_word("poop")
    assert hits[0]["source_id"] == env.sid

    r = env.client.post("/words/poop/pref", data={"action": "clear"},
                        follow_redirects=True)
    assert r.status_code == 200
    assert "poop" not in env.db.word_prefs()


def test_generate_uses_pref_source(env, monkeypatch):
    env.db.add_word(env.sid, "poop", 1.0, 2.0)
    other = env.db.add_source("v2", "u", "Two", "/tmp/v2.mp4", 10.0)
    env.db.add_word(other, "poop", 5.0, 6.0)
    env.db.set_word_pref("poop", other)
    env.db.commit()
    from ytpgen.compose.matcher import Matcher

    clips = Matcher(env.cfg).resolve_sentence("poop", env.db, mode="word")
    assert clips[0].source_id == other


def test_generate_source_selection(env, monkeypatch):
    env.db.add_word(env.sid, "poop", 1.0, 2.0)
    other = env.db.add_source("v2", "u", "Two", "/tmp/v2.mp4", 10.0)
    env.db.add_word(other, "poop", 5.0, 6.0)
    env.db.commit()
    from ytpgen.compose.matcher import Matcher

    clips = Matcher(env.cfg).resolve_sentence("poop", env.db, mode="word",
                                              source_ids={other})
    assert len(clips) == 1 and clips[0].source_id == other
