"""Generated bridges: which cuts get one, the J-cut shift, and rendering a silent clip with borrowed audio."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ytpgen.compose import bridge
from ytpgen.compose.renderer import Renderer
from ytpgen.models import EDL, EDLEvent

sys.path.insert(0, str(Path(__file__).parent))
from media_helper import ffprobe_info, make_test_clip  # noqa: E402

D = bridge.FRAMES / bridge.WAN_FPS


def _edl(a: str, b: str) -> EDL:
    return EDL(fps=30, width=320, height=240, events=[
        EDLEvent(local_path=a, start_s=0.0, end_s=1.0),
        EDLEvent(local_path=a, start_s=1.0, end_s=3.0),           # contiguous with 0: no bridge
        EDLEvent(local_path=b, start_s=0.2, end_s=2.8),           # jump: bridge
        EDLEvent(local_path=a, start_s=0.0, end_s=0.4, effects=["hold"]),  # stinger: never
        EDLEvent(local_path=b, start_s=0.0, end_s=1.2),           # too short to give its head away
    ])


def test_candidates_skip_contiguous_stingers_and_short():
    assert bridge.candidates(_edl("a.mp4", "b.mp4")) == [1]


@pytest.fixture
def media(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    make_test_clip(a, seconds=3.0)
    make_test_clip(b, seconds=3.0, pattern="smptebars")
    return a, b


def test_add_bridges_jcut(tmp_path, media, monkeypatch):
    a, b = map(str, media)
    calls = []

    def fake_generate(cfg, a_png, b_png, out, w, h, **kw):
        calls.append((len(a_png) > 0, len(b_png) > 0))
        make_test_clip(out, seconds=D, pattern="rgbtestsrc", with_audio=False)
        return out

    monkeypatch.setattr(bridge, "generate", fake_generate)
    cfg = {"workspace": str(tmp_path / "ws")}
    edl = _edl(a, b)
    out = bridge.add_bridges(cfg, edl, n=3)
    assert calls == [(True, True)]
    assert [e.index for e in out.events] == list(range(6))
    br, nxt = out.events[2], out.events[3]
    assert br.effects == ["bridge"] and br.audio_path == b and br.audio_start_s == pytest.approx(0.2)
    assert nxt.start_s == pytest.approx(0.2 + D)  # B's picture resumes where the bridge's audio stops
    assert edl.events[2].start_s == pytest.approx(0.2)  # input EDL untouched

    bridge.add_bridges(cfg, edl, n=3)  # same inputs: served from cache
    assert len(calls) == 1

    video = Renderer({"compose": {"fps": 30, "width": 320, "height": 240}}).render(out, tmp_path / "out.mp4")
    streams = {s["codec_type"] for s in ffprobe_info(video)["streams"]}
    assert streams == {"video", "audio"}
