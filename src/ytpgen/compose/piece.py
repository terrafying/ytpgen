"""One call from "these films, this mood" to a finished, checked video (shared by CLI and web)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ytpgen.compose.bridge import add_bridges
from ytpgen.compose.edl import save_edl
from ytpgen.compose.judge import judge
from ytpgen.compose.matcher import Matcher
from ytpgen.compose.phrases import derive_phrases
from ytpgen.compose.planner import make_plans
from ytpgen.compose.renderer import Renderer
from ytpgen.compose.verify import transcribe, wer
from ytpgen.index import Index

logger = logging.getLogger(__name__)


def make_piece(cfg: dict, db: Index, out: Path, tone: str = "funny", theme: str | None = None,
               source_ids: set[int] | None = None, seconds: int = 30, drafts: int = 4,
               render: int = 2, max_wer: float = 0.35, seed: int = 0, bridges: int = 0, found_only: bool = False,
               log=logger.info) -> list[dict]:
    """Draft, judge, render the top `render` drafts, verify; returns results best first (+ .plan.json)."""
    phrases = derive_phrases(db.conn, source_ids)
    if not phrases:
        raise RuntimeError("no phrases: ingest and analyze some films first")
    plans = make_plans(cfg, db, phrases, tone, theme, seconds=seconds, n=drafts, seed=seed,
                       source_ids=source_ids, found_only=found_only)
    if not plans:
        raise RuntimeError("the planner produced no usable drafts (is the LLM server up?)")
    by_id = {p.id: p for p in phrases}
    for k, plan in enumerate(plans):
        try:
            plan.judged = judge(cfg, plan.lines, by_id, seed=seed + k)
        except Exception as exc:  # noqa: BLE001 - an unjudged draft just ranks last
            plan.judged = {"score": 0.0, "why": f"judge failed: {exc}"}
        log(f"draft {plan.title!r}: unexpected congruence {plan.judged['score']:.2f}, cost {plan.cost:.2f}")
    plans.sort(key=lambda p: (-p.judged["score"], p.cost))
    results = []
    for k, plan in enumerate(plans[:render]):
        edl = Matcher.edl_from_clips(plan.clips, fps=cfg["compose"]["fps"],
                                     width=cfg["compose"]["width"], height=cfg["compose"]["height"])
        if bridges:  # needs ComfyUI with Wan 2.2 and the GPU free of the planner LLM
            edl = add_bridges(cfg, edl, bridges, seed=seed, log=log)
        path = out.with_name(f"{out.stem}.{k}{out.suffix}")
        save_edl(edl, path.with_suffix(".edl.json"))
        Renderer(cfg).render(edl, path)
        heard = transcribe(cfg, path)
        score = wer(plan.intended, heard)
        results.append({
            "file": str(path), "title": plan.title, "cost": round(plan.cost, 3),
            "unexpected_congruence": plan.judged, "wer": round(score, 3), "missing": plan.missing,
            "intended": plan.intended, "heard": heard.strip(), "lines": plan.lines,
            "chips": [{"text": c.text, "source_id": c.source_id, "kind": c.kind} for c in plan.clips],
        })
        log(f"[{k}] {plan.title!r} uc={plan.judged['score']:.2f} cost={plan.cost:.2f} wer={score:.2f} {path}")
    results.sort(key=lambda r: (r["wer"] > max_wer, -r["unexpected_congruence"]["score"], r["cost"]))
    out.with_suffix(".plan.json").write_text(json.dumps(results, indent=1))
    return results
