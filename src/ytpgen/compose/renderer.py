"""Pure-ffmpeg/subprocess renderer: turn an EDL into one mp4.

Pipeline per event (single ffmpeg pass):
  extract (-ss/-t) -> scale/pad to WxH -> fps -> [reverse] -> speed -> gain
  -> libx264/aac segment file. Segments are then concatenated with the
  concat demuxer (-c copy); on failure (e.g. mixed stream parameters) we
  fall back to a concat-filter re-encode of all segments.

Determinism: the render is a pure function of the EDL — no randomness.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ytpgen.models import EDL, EDLEvent

logger = logging.getLogger(__name__)

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"

CRF = "18"
PRESET = "veryfast"
AUDIO_BITRATE = "128k"
FADE_S = 0.008
HOLD_S = 0.35
LEVEL_TARGET_DB = -24.0  # mean (RMS) level every clip is brought to before the mix
LEVEL_MAX_DB = 12.0  # cap so a near-silent clip isn't boosted into hiss


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run an ffmpeg subprocess, raising with stderr on failure."""
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"] + cmd
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {tail}\ncmd: {' '.join(cmd)}")
    return proc


def _has_audio_stream(path: Path) -> bool:
    """True if the media file has at least one audio stream."""
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return bool(proc.stdout.strip())


def _fmt_stage(value: float) -> str:
    """Format an atempo stage: integral values as '2.0', else full precision."""
    if abs(value - round(value)) < 1e-9:
        return f"{round(value):.1f}"
    return f"{value:.10g}"


def _atempo_chain(speed: float) -> str:
    """Build an atempo filter chain clamping each stage to [0.5, 2.0].

    speed=3.0 -> "atempo=2.0,atempo=1.5"; speed=0.4 -> "atempo=0.5,atempo=0.8".
    """
    if speed <= 0:
        raise ValueError(f"speed must be > 0, got {speed}")
    stages: list[float] = []
    remaining = float(speed)
    while remaining > 2.0:
        stages.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        stages.append(0.5)
        remaining /= 0.5
    stages.append(remaining)
    return ",".join(f"atempo={_fmt_stage(s)}" for s in stages)


class Renderer:
    """Renders an EDL to a single video using only ffmpeg subprocesses."""

    def __init__(self, cfg: dict):
        compose = cfg.get("compose", {})
        self.width: int = int(compose.get("width", 1280))
        self.height: int = int(compose.get("height", 720))
        self.fps: float = float(compose.get("fps", 30.0))
        self.ffmpeg: str = FFMPEG

    # ------------------------------------------------------------------ #
    # per-event single-pass graph: extract + geometry + reverse + speed +
    # gain all in one -filter_complex
    # ------------------------------------------------------------------ #
    @staticmethod
    def _audio_start(ev: EDLEvent) -> float:
        return ev.audio_start_s if ev.audio_path else ev.start_s

    def _level_db(self, ev: EDLEvent) -> float:
        """Gain that brings this clip's mean level to LEVEL_TARGET_DB (films differ by ~6 dB)."""
        proc = subprocess.run(
            [self.ffmpeg, "-nostdin", "-hide_banner", "-ss", f"{max(0.0, self._audio_start(ev)):.6f}",
             "-t", f"{ev.end_s - ev.start_s:.6f}", "-i", str(ev.audio_path or ev.local_path),
             "-vn", "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True,
        )
        m = re.search(r"mean_volume: (-?[\d.]+) dB", proc.stderr)
        if not m:
            return 0.0
        return max(-LEVEL_MAX_DB, min(LEVEL_MAX_DB, LEVEL_TARGET_DB - float(m.group(1))))

    def _event_filter_complex(self, ev: EDLEvent, level_db: float = 0.0) -> tuple[str, str]:
        """Return (filter_complex, vout_label) for one event."""
        vparts: list[str] = []
        apart: list[str] = []
        a_in = "[1:a]" if ev.audio_path else "[0:a]"

        if ev.reverse:
            vparts.append("[0:v]reverse")
            apart.append(f"{a_in}areverse")
        else:
            vparts.append("[0:v]null")
            apart.append(f"{a_in}anull")

        # audio speed before video setpts; both before geometry so pad/fps
        # operate on the final timeline
        if abs(ev.speed - 1.0) > 1e-9:
            apart.append(_atempo_chain(ev.speed))
            vparts.append(f"setpts=PTS/{ev.speed:g}")

        gain = ev.gain_db + level_db  # gain_db stays the hand-edited extra on top of levelling
        if abs(gain) > 1e-9:
            apart.append(f"volume={gain:g}dB")

        # 8 ms fades at both ends: cuts land mid-waveform, and a hard edge clicks
        dur = (ev.end_s - ev.start_s) / ev.speed
        if dur > 4 * FADE_S:
            apart.append(f"afade=t=in:d={FADE_S},afade=t=out:st={dur - FADE_S:.6f}:d={FADE_S}")

        if "hold" in ev.effects:  # freeze the last frame in silence: comic timing for stingers
            hold = round(HOLD_S * self.fps) / self.fps  # whole frames, or audio drifts ahead of video
            vparts.append(f"tpad=stop_mode=clone:stop_duration={hold:.6f}")
            apart.append(f"apad=pad_dur={hold:.6f}")

        vparts.append(
            f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
            f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,"
            f"fps={self.fps:g}"
        )
        # uniform audio layout across events: concat demuxer needs identical
        # stream parameters or -c copy drifts/clicks between segments
        apart.append("aresample=44100")
        apart.append("aformat=sample_fmts=fltp:channel_layouts=stereo")
        apart.append("apad")  # + -shortest: audio cut to exactly the video's length, so cuts don't drift

        fc = ",".join(vparts) + "[vout];" + ",".join(apart) + "[aout]"
        return fc, "[vout]"

    def _render_event(self, ev: EDLEvent, out_path: Path) -> None:
        duration = max(0.0, ev.end_s - ev.start_s)
        if duration <= 0:
            raise ValueError(
                f"event {ev.index} has non-positive duration "
                f"({ev.start_s}..{ev.end_s})"
            )
        fc, vout = self._event_filter_complex(ev, self._level_db(ev))
        cmd = [
            "-ss", f"{max(0.0, ev.start_s):.6f}",
            "-t", f"{duration:.6f}",
            "-i", str(ev.local_path),
            *(["-ss", f"{max(0.0, ev.audio_start_s):.6f}", "-t", f"{duration:.6f}", "-i", str(ev.audio_path)]
              if ev.audio_path else []),
            "-map", f"{vout}",
            "-map", "[aout]",
            "-filter_complex", fc,
            "-c:v", "libx264", "-crf", CRF, "-preset", PRESET,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", AUDIO_BITRATE,
            "-r", f"{self.fps:g}",
            "-shortest",
            "-movflags", "+faststart",
            str(out_path),
        ]
        _run(cmd)

    # ------------------------------------------------------------------ #
    # concat
    # ------------------------------------------------------------------ #
    def _concat_copy(self, segments: list[Path], list_file: Path, output_path: Path) -> None:
        lines = "".join(f"file '{s}'\n" for s in segments)
        list_file.write_text(lines, encoding="utf-8")
        _run([
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ])

    def _concat_reencode(self, segments: list[Path], output_path: Path) -> None:
        first = segments[0] if segments else None
        has_audio = bool(first and _has_audio_stream(first))
        n = len(segments)
        inputs: list[str] = []
        for s in segments:
            inputs += ["-i", str(s)]
        if has_audio:
            fc = (
                "".join(f"[{i}:v]" for i in range(n))
                + f"concat=n={n}:v=1:a=0[v];"
                + "".join(f"[{i}:a]" for i in range(n))
                + f"concat=n={n}:v=0:a=1,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=44100[a]"  # clips come from different films and mics; loudnorm outputs 192 kHz
            )
            maps = ["-map", "[v]", "-map", "[a]"]
        else:
            fc = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
            maps = ["-map", "[v]"]
        _run([
            *inputs,
            "-filter_complex", fc,
            *maps,
            "-c:v", "libx264", "-crf", CRF, "-preset", PRESET,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", AUDIO_BITRATE,
            "-movflags", "+faststart",
            str(output_path),
        ])

    # ------------------------------------------------------------------ #
    # main entry
    # ------------------------------------------------------------------ #
    def render(self, edl: EDL, output_path: Path) -> Path:
        """Render every event in the EDL; failed events are skipped."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = edl.fps if edl.fps else self.fps
        self.fps = float(fps)  # EDL fps wins over cfg compose fps

        tmpdir = Path(tempfile.mkdtemp(prefix="ytpgen-render-"))
        segments: list[Path] = []
        try:
            rendered = 0
            failed = 0
            for ev in edl.events:
                seg = tmpdir / f"segment_{ev.index:04d}.mp4"
                try:
                    logger.info(
                        "rendering event %d: %s [%.2f-%.2f] reverse=%s speed=%s gain=%sdB",
                        ev.index, Path(ev.local_path).name,
                        ev.start_s, ev.end_s, ev.reverse, ev.speed, ev.gain_db,
                    )
                    self._render_event(ev, seg)
                    segments.append(seg)
                    rendered += 1
                except Exception as exc:  # noqa: BLE001 — skip bad events, keep rendering
                    failed += 1
                    logger.warning(
                        "skipping event %d (render failed): %s", ev.index, exc
                    )
            if not segments:
                raise RuntimeError(
                    f"no events rendered successfully "
                    f"({failed} failed of {len(edl.events)})"
                )
            if failed:
                logger.warning("%d/%d events skipped", failed, len(edl.events))

            list_file = tmpdir / "concat.txt"
            # Always re-encode the final concat: -c copy on independently
            # encoded AAC segments accumulates priming-sample drift and
            # produces clicks at every boundary. list_file kept for the
            # fallback concat-demuxer path in _concat_reencode.
            self._concat_reencode(segments, output_path)
            logger.info(
                "concat re-encode OK: %s (%d segments)", output_path, len(segments)
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return output_path
