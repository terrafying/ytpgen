"""Shared ffmpeg helpers for tests and the web preview.

Not a test module (no test_ prefix) so it can be imported by both.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"


def make_test_clip(
    path: Path,
    seconds: float = 1.0,
    pattern: str = "testsrc2",
    with_audio: bool = True,
) -> None:
    """Generate a tiny h264+aac clip from a lavfi source (real ffmpeg).

    Used by tests and the web UI preview to synthesize disposable media.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    cmd += ["-f", "lavfi", "-i", f"{pattern}=size=320x240:duration={seconds:g}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds:g}"]
    cmd += [
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p",
    ]
    if with_audio:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
    else:
        cmd += ["-an"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def ffprobe_info(path: Path) -> dict:
    """Return full ffprobe JSON (format + streams) for a media file."""
    cmd = [
        FFPROBE, "-v", "error",
        "-show_format", "-show_streams",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)
