"""Generated bridges between cuts: Wan 2.2 first-last-frame video via ComfyUI (cfg video.base_url).

A bridge starts on the last frame of clip A and ends on the frame where clip B's picture resumes.
B's audio starts under the bridge (a J-cut), so speech keeps its pace and the generated motion
reads as continuity (a head turn, a pan) rather than a pause. Bridges are cached by their inputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from ytpgen.config import workspace_dir
from ytpgen.models import EDL, EDLEvent

logger = logging.getLogger(__name__)

WORKFLOW = Path(__file__).with_name("wan_flf2v.json")
WAN_FPS = 16
FRAMES = 17  # ~1 s; Wan wants 4k+1
MIN_REST_S = 0.6  # B must still have this much left after the bridge eats its head
# shortcut: one generic prompt; per-scene tags (analyze/tag.py) could steer it once bridges prove out
PROMPT = ("black and white mid-century educational film, the scene continues naturally, "
          "small natural movements, one continuous take, film grain")


def _base(cfg: dict) -> str:
    return cfg.get("video", {}).get("base_url", "http://127.0.0.1:8188")


def _upload(cfg: dict, png: bytes) -> str:
    b = "----" + uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{uuid.uuid4().hex}.png\"\r\n"
            "Content-Type: image/png\r\n\r\n").encode() + png + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(_base(cfg) + "/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["name"]


def frame_png(path: str, at_s: float, width: int, height: int) -> bytes:
    proc = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{max(0.0, at_s):.3f}", "-i", path,
                           "-frames:v", "1", "-vf", f"scale={width}:{height}", "-f", "image2", "-c:v", "png", "-"],
                          capture_output=True, check=True)
    return proc.stdout


def generate(cfg: dict, a_png: bytes, b_png: bytes, out: Path, width: int, height: int,
             prompt: str = PROMPT, frames: int = FRAMES, seed: int = 0, timeout_s: int = 900) -> Path:
    """Render one silent bridge clip to *out* (16 fps)."""
    wf = json.loads(WORKFLOW.read_text())
    wf["11"]["inputs"]["image"], wf["12"]["inputs"]["image"] = _upload(cfg, a_png), _upload(cfg, b_png)
    wf["8"]["inputs"]["text"] = prompt
    wf["13"]["inputs"].update(width=width, height=height, length=frames)
    wf["14"]["inputs"]["noise_seed"] = seed
    req = urllib.request.Request(_base(cfg) + "/prompt", data=json.dumps({"prompt": wf}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        pid = json.loads(r.read())["prompt_id"]
    deadline = time.time() + timeout_s
    while True:
        with urllib.request.urlopen(f"{_base(cfg)}/history/{pid}", timeout=60) as r:
            h = json.loads(r.read()).get(pid)
        if h and h.get("status", {}).get("completed") is not None:
            break
        if time.time() > deadline:
            raise TimeoutError(f"bridge {pid} not done after {timeout_s}s")
        time.sleep(3)
    if h["status"].get("status_str") != "success":
        raise RuntimeError(f"bridge failed: {h['status']}")
    frames_dir = out.with_suffix("")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(h["outputs"]["17"]["images"]):
        q = urllib.parse.urlencode({k: im[k] for k in ("filename", "subfolder", "type")})
        with urllib.request.urlopen(f"{_base(cfg)}/view?{q}", timeout=60) as r:
            (frames_dir / f"{i:04d}.png").write_bytes(r.read())
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-framerate", str(WAN_FPS),
                    "-i", str(frames_dir / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
                   check=True)
    return out


def _size(cfg: dict, path: str) -> tuple[int, int]:
    """Wan size at 480p keeping the source aspect (multiples of 16)."""
    proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        w, h = map(int, proc.stdout.strip().split(",")[:2])
    except ValueError:
        w, h = 640, 480
    return max(16, round(480 * w / h / 16) * 16), 480


def candidates(edl: EDL, min_rest_s: float = MIN_REST_S) -> list[int]:
    """Indices i where a bridge can go between events i and i+1: a real jump, no stinger on either side."""
    d = FRAMES / WAN_FPS
    out = []
    for i, (a, b) in enumerate(zip(edl.events, edl.events[1:])):
        contiguous = a.local_path == b.local_path and abs(b.start_s - a.end_s) < 0.1
        stinger = "hold" in a.effects or "hold" in b.effects
        if not contiguous and not stinger and not b.reverse and (b.end_s - b.start_s) >= d + min_rest_s:
            out.append(i)
    return out


def add_bridges(cfg: dict, edl: EDL, n: int, seed: int = 0, log=logger.info) -> EDL:
    """Insert up to *n* generated bridges (earliest eligible cuts first). Returns a new EDL."""
    picks = set(candidates(edl)[:n])  # shortcut: earliest cuts; rank by judge look score once it's per-cut
    if not picks:
        return edl
    cache = workspace_dir(cfg) / "bridges"
    cache.mkdir(parents=True, exist_ok=True)
    d = FRAMES / WAN_FPS
    events: list[EDLEvent] = []
    for i, ev in enumerate(edl.events):
        if i - 1 in picks:  # this event's head plays under the bridge before it
            ev = ev.model_copy(update={"start_s": ev.start_s + d})
        events.append(ev)
        if i in picks:
            b = edl.events[i + 1]
            w, h = _size(cfg, b.local_path)
            a_png = frame_png(ev.local_path, ev.end_s - 1 / WAN_FPS, w, h)
            b_png = frame_png(b.local_path, b.start_s + d, w, h)
            key = hashlib.sha256(a_png + b_png + f"{PROMPT}|{FRAMES}|{seed}|{WORKFLOW.read_text()}".encode()).hexdigest()[:16]
            clip = cache / f"{key}.mp4"
            if not clip.exists():
                t0 = time.time()
                generate(cfg, a_png, b_png, clip, w, h, seed=seed)
                log(f"bridge after line {i} ({time.time() - t0:.0f}s): {clip.name}")
            events.append(EDLEvent(local_path=str(clip), start_s=0.0, end_s=d, effects=["bridge"],
                                   audio_path=b.local_path, audio_start_s=b.start_s))
    for k, ev in enumerate(events):
        ev.index = k
    return edl.model_copy(update={"events": events})
