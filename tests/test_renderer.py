"""Tests for the pure-ffmpeg renderer and EDL save/load."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ytpgen.compose.edl import load_edl, save_edl
from ytpgen.compose.renderer import Renderer, _atempo_chain
from ytpgen.models import EDL, EDLEvent

sys.path.insert(0, str(Path(__file__).parent))
from media_helper import ffprobe_info, make_test_clip  # noqa: E402

CFG = {
    "compose": {"fps": 30, "width": 1280, "height": 720},
}


def _probe_duration(path: Path) -> float:
    return float(ffprobe_info(path)["format"]["duration"])


def _has_video_stream(path: Path) -> bool:
    return any(s.get("codec_type") == "video" for s in ffprobe_info(path)["streams"])


def _build_edl(clip_a: Path, clip_b: Path) -> EDL:
    events = [
        EDLEvent(index=0, local_path=str(clip_a), start_s=0.0, end_s=1.0, text="normal"),
        EDLEvent(index=1, local_path=str(clip_b), start_s=0.0, end_s=1.0,
                 text="reversed", reverse=True),
        EDLEvent(index=2, local_path=str(clip_a), start_s=0.0, end_s=1.0,
                 text="fast+loud", speed=1.5, gain_db=3.0),
    ]
    return EDL(version=1, fps=30, width=1280, height=720, seed=42, events=events)


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("clips")
    a = d / "a.mp4"
    b = d / "b.mp4"
    make_test_clip(a, seconds=1.0)
    make_test_clip(b, seconds=1.0, pattern="smptebars")
    return a, b


class TestAtempoChain:
    def test_speed_3(self) -> None:
        # 3.0 -> 2.0 * 1.5
        assert _atempo_chain(3.0) == "atempo=2.0,atempo=1.5"

    def test_speed_0_4(self) -> None:
        # 0.4 -> 0.5 * 0.8
        assert _atempo_chain(0.4) == "atempo=0.5,atempo=0.8"

    def test_speed_in_range(self) -> None:
        assert _atempo_chain(1.5) == "atempo=1.5"
        assert _atempo_chain(0.75) == "atempo=0.75"
        assert _atempo_chain(1.0) == "atempo=1.0"

    def test_speed_extreme(self) -> None:
        # 8.0 -> 2*2*2
        assert _atempo_chain(8.0) == "atempo=2.0,atempo=2.0,atempo=2.0"

    def test_bad_speed(self) -> None:
        with pytest.raises(ValueError):
            _atempo_chain(0.0)
        with pytest.raises(ValueError):
            _atempo_chain(-1.0)


class TestEDLio:
    def test_roundtrip(self, tmp_path: Path, clips: tuple[Path, Path]) -> None:
        edl = _build_edl(*clips)
        p = tmp_path / "edit.json"
        out = save_edl(edl, p)
        assert out == p
        assert p.exists()
        loaded = load_edl(p)
        assert loaded == edl
        assert loaded.events[2].speed == 1.5
        assert loaded.seed == 42

    def test_load_creates_validation_error(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text('{"version": 1, "events": [{"nope": true}]}')
        with pytest.raises(Exception):
            load_edl(p)


class TestRenderer:
    def test_render_end_to_end(self, tmp_path: Path, clips: tuple[Path, Path]) -> None:
        edl = _build_edl(*clips)
        out = tmp_path / "out.mp4"
        result = Renderer(CFG).render(edl, out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 10 * 1024
        assert _has_video_stream(out)
        dur = _probe_duration(out)
        # events: 1.0 + 1.0 + (1.0 / 1.5) = 2.667s; allow generous slack
        assert dur == pytest.approx(8 / 3, abs=1.3)

    def test_render_skips_bad_events(self, tmp_path: Path, clips: tuple[Path, Path]) -> None:
        edl = _build_edl(*clips)
        edl.events.append(
            EDLEvent(index=3, local_path=str(tmp_path / "missing.mp4"),
                     start_s=0.0, end_s=1.0)
        )
        out = tmp_path / "out_partial.mp4"
        result = Renderer(CFG).render(edl, out)
        assert result == out
        assert out.exists()

    def test_render_all_bad_raises(self, tmp_path: Path) -> None:
        edl = EDL(events=[EDLEvent(index=0, local_path="/nonexistent/x.mp4",
                                   start_s=0.0, end_s=1.0)])
        with pytest.raises(RuntimeError, match="no events rendered"):
            Renderer(CFG).render(edl, tmp_path / "out.mp4")

    def test_render_respects_edl_fps(self, tmp_path: Path, clips: tuple[Path, Path]) -> None:
        edl = _build_edl(*clips)
        edl.fps = 24
        out = tmp_path / "out24.mp4"
        Renderer(CFG).render(edl, out)
        vstream = next(s for s in ffprobe_info(out)["streams"]
                       if s.get("codec_type") == "video")
        assert abs(float(vstream["avg_frame_rate"].split("/")[0]) - 24.0) < 1.0

    def test_geometry_matches_cfg(self, tmp_path: Path, clips: tuple[Path, Path]) -> None:
        edl = _build_edl(*clips)
        out = tmp_path / "geo.mp4"
        Renderer(CFG).render(edl, out)
        vstream = next(s for s in ffprobe_info(out)["streams"]
                       if s.get("codec_type") == "video")
        assert vstream["width"] == 1280
        assert vstream["height"] == 720
