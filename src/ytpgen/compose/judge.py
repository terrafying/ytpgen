"""Unexpected congruence: a first, crude stand-in for "is it funny / does it land".

A line scores *unexpected* when its meaning in the new piece is far from what it
meant in its original scene, and a transition scores *congruent* when the next
line follows as if it had been written for it. The product rewards pieces that
are both surprising and coherent; either alone is just noise or just the film.
Owner may later fine-tune a judge on their own ratings (2026-11-22).
"""

from __future__ import annotations

import json
import re

from ytpgen.compose.phrases import Phrase
from ytpgen.compose.planner import chat

PROMPT = """These lines were cut out of old films and re-assembled into a new piece. For each line
you get its ORIGINAL context (the lines around it in the film) and its position in the NEW piece.

Score, as integers 0-10:
- "shift" per line: how different its meaning or tone is in the new piece compared to the original
  scene (0 = means exactly the same, 10 = completely recontextualised).
- "fit" per transition: how naturally line i+1 follows line i in the new piece, as if written to
  follow it (0 = non sequitur, 10 = seamless).

Reply with JSON only: {"shift": [..one per line..], "fit": [..one per transition..], "why": "<one sentence>"}

"""


def _context(ph: Phrase | None, by_id: dict[int, Phrase]) -> str:
    if ph is None:
        return "(spoken glue, no original scene)"
    near = [by_id.get(ph.id + d) for d in (-1, 1)]
    before, after = (n.text if n and (n.source_id, n.scene_idx) == (ph.source_id, ph.scene_idx) else "" for n in near)
    return f"...{before} [{ph.text}] {after}..."


def judge(cfg: dict, lines: list[dict], by_id: dict[int, Phrase], llm=chat, seed: int = 0) -> dict:
    """Returns {"shift", "fit", "unexpected", "congruent", "score", "why"}; score in [0, 1]."""
    items = []
    for line in lines:
        ph = by_id.get(int(line["p"])) if "p" in line and str(line["p"]).isdigit() else None
        text = line.get("words") or (ph.text if ph else line.get("say", ""))
        items.append({"new_line": text, "original_context": _context(ph, by_id)})
    if len(items) < 2:
        return {"shift": [], "fit": [], "unexpected": 0.0, "congruent": 0.0, "score": 0.0, "why": "too short"}
    msg = [{"role": "user", "content": PROMPT + json.dumps(items, indent=1)}]
    m = re.search(r"\{.*\}", llm(cfg, msg, temperature=0.0, seed=seed), re.S)
    data = json.loads(m.group(0)) if m else {}
    shift = [min(10, max(0, int(x))) for x in data.get("shift", [])][: len(items)]
    fit = [min(10, max(0, int(x))) for x in data.get("fit", [])][: len(items) - 1]
    unexpected = sum(shift) / (10 * len(items))  # missing scores count as 0
    congruent = sum(fit) / (10 * (len(items) - 1))
    return {"shift": shift, "fit": fit, "unexpected": round(unexpected, 3),
            "congruent": round(congruent, 3), "score": round(unexpected * congruent, 3),
            "why": str(data.get("why", ""))}
