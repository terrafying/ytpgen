"""Who is on screen: tag each scene's middle frame with a vision-language model.

Gives the planner visual continuity to work with (same character across a
conversation, reaction shots, narrator-over-face jokes on purpose) instead of
text alone. VLM is any OpenAI-compatible server with image input, e.g.
llama-server with Qwen3-VL + mmproj (cfg vision.base_url).
"""

from __future__ import annotations

import base64
import json
import logging
import re
import sqlite3
import subprocess
import urllib.request

logger = logging.getLogger(__name__)

PROMPT = """You are helping a video editor. Describe this film frame as JSON only:
{"people": ["<short visual description of each visible person or character, max 4, e.g. 'man in suit with hat'>"],
 "speaking": true/false (someone visibly talking),
 "setting": "<3-6 words>",
 "shot": "close-up|medium|wide|insert|title card|animation"}"""


def frame_jpeg(path: str, at_s: float, width: int = 512) -> bytes:
    proc = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{at_s:.2f}", "-i", path,
                           "-frames:v", "1", "-vf", f"scale={width}:-2", "-f", "image2", "-c:v", "mjpeg", "-"],
                          capture_output=True)
    return proc.stdout


def vlm_chat(cfg: dict, image: bytes, prompt: str = PROMPT) -> str:
    vc = cfg.get("vision", {})
    body = {"model": vc.get("model", "local"), "temperature": 0.0, "max_tokens": 300,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}},
                {"type": "text", "text": prompt}]}]}
    req = urllib.request.Request(vc.get("base_url", "http://127.0.0.1:8098/v1") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=vc.get("timeout_s", 120)) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def parse_tags(reply: str) -> dict:
    m = re.search(r"\{.*\}", reply, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    people = [str(p)[:60] for p in d.get("people", []) if p][:4] if isinstance(d.get("people"), list) else []
    return {"people": people, "speaking": bool(d.get("speaking")),
            "setting": str(d.get("setting", ""))[:60], "shot": str(d.get("shot", ""))[:20]}


def tag_scenes(cfg: dict, conn: sqlite3.Connection, source_ids: set[int] | None = None,
               llm=vlm_chat, grab=frame_jpeg) -> int:
    """Tag every untagged scene (middle frame). Returns how many were tagged."""
    rows = conn.execute(
        "SELECT sc.id, sc.source_id, sc.start_s, sc.end_s, s.local_path FROM scenes sc"
        " JOIN sources s ON s.id = sc.source_id"
        " WHERE sc.id NOT IN (SELECT scene_id FROM scene_tags) ORDER BY sc.source_id, sc.start_s"
    ).fetchall()
    n = 0
    for sid, src, start, end, path in rows:
        if source_ids and src not in source_ids:
            continue
        img = grab(path, (start + end) / 2)
        if not img:
            continue
        try:
            tags = parse_tags(llm(cfg, img))
        except Exception as exc:  # noqa: BLE001 - one bad frame shouldn't stop the library
            logger.warning("tagging scene %d failed: %s", sid, exc)
            continue
        conn.execute("INSERT OR REPLACE INTO scene_tags (scene_id, tags) VALUES (?, ?)", (sid, json.dumps(tags)))
        n += 1
        if n % 25 == 0:
            conn.commit()
            logger.info("tagged %d scenes", n)
    conn.commit()
    return n


def describe(tags: dict | None) -> str:
    """One-line human/LLM-readable scene description."""
    if not tags:
        return ""
    who = "; ".join(tags.get("people") or []) or "no people"
    talk = ", someone talking" if tags.get("speaking") else ""
    return f"{tags.get('setting', '')}, {tags.get('shot', '')}: {who}{talk}".strip(", ")
