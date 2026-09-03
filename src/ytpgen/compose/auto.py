"""AutoPooper: automatic YouTube-Poop style edits from an analyzed source.

Reads one analyzed video's scenes and words straight from the sqlite index
and plans an :class:`~ytpgen.models.EDL` of YTP-style mangled cuts.
Deterministic: the seed is a stable hash of the video_id and every choice
is made through ``random.Random(seed)``, so the same video_id always plans
the identical event list (reproducible renders / re-renders).

Selection strategy:
- Prefer high-scoring scenes; prefer word-dense regions (speech is the
  raw material of YTP).
- Avoid reusing a segment until every candidate has been used once.
- ~50% of events stay plain; the rest get reverse/speed/gain/effects.
- Occasional ``stutter`` expands one segment into 3 events whose middle
  is a ~0.15s blip; ``zoom``/``reverse`` are appended as renderer hints
  (the renderer may ignore unknown effects).
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Any

from ytpgen.models import EDL, EDLEvent

logger = logging.getLogger(__name__)

# Speed / gain choice tables — duplicated entries raise their odds.
SPEEDS: list[float] = [0.75, 1.0, 1.0, 1.25, 1.5]
GAINS: list[float] = [-3.0, 0.0, 0.0, 2.0]

# Mangling probabilities.
P_REVERSE = 0.25
P_PLAIN = 0.50  # share of events left untouched
P_STUTTER = 0.15
P_ZOOM = 0.15

STUTTER_BLIP_S = 0.15


def _stable_seed(video_id: str) -> int:
    """Stable, process-independent seed derived from the video id."""
    return int(hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:8], 16)


class AutoPooper:
    """Plan an auto-YTP EDL from one analyzed source video."""

    def __init__(self, cfg: dict, db_conn: Any) -> None:
        self.cfg = cfg
        self.conn = db_conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, video_id: str, target_s: int = 60) -> EDL:
        """Build an auto-YTP edit for *video_id* totalling ~target_s seconds."""
        compose = self.cfg.get("compose", {})
        fps = float(compose.get("fps", 30))
        width = int(compose.get("width", 1280))
        height = int(compose.get("height", 720))

        source = self.conn.execute(
            "SELECT id, local_path, duration_s FROM sources WHERE video_id = ?",
            (video_id,),
        ).fetchone()
        if source is None:
            raise ValueError(f"video not analyzed: {video_id!r}")
        local_path = source["local_path"]

        scenes = [
            dict(r)
            for r in self.conn.execute(
                "SELECT start_s, end_s, label, score FROM scenes"
                " WHERE source_id = ? ORDER BY start_s",
                (source["id"],),
            ).fetchall()
        ]
        words = [
            dict(r)
            for r in self.conn.execute(
                "SELECT word, start_s, end_s FROM words"
                " WHERE source_id = ? ORDER BY start_s",
                (source["id"],),
            ).fetchall()
        ]
        if not words and not scenes:
            raise ValueError(f"video not analyzed: {video_id!r}")

        seed = _stable_seed(video_id)
        rng = random.Random(seed)

        segments = self._pick_segments(scenes, words, rng, target_s)
        events: list[EDLEvent] = []
        for seg in segments:
            base = {
                "local_path": local_path,
                "start_s": float(seg["start_s"]),
                "end_s": float(seg["end_s"]),
                "text": seg.get("text", ""),
            }
            ev = self._mangle_event(base, rng)
            events.extend(self._split_stutter(ev, rng))

        notes = f"auto: {video_id} target={target_s}s"
        total = sum(e.end_s - e.start_s for e in events)
        logger.info("auto: %s -> %d events, %.1fs (target %ds, seed %d)",
                    video_id, len(events), total, target_s, seed)
        return EDL(
            version=1,
            fps=fps,
            width=width,
            height=height,
            seed=seed,
            events=events,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Segment selection
    # ------------------------------------------------------------------

    def _pick_segments(
        self,
        scenes: list[dict],
        words: list[dict],
        rng: random.Random,
        target_s: int,
    ) -> list[dict]:
        """Pick interesting segments until total duration >= target_s.

        Ranks candidates by scene score + word density (or word-group
        density in the scene-less fallback), then picks in seeded-random
        rounds biased toward the best, never reusing a segment until
        every candidate has been used once.
        """

        def _density(start: float, end: float) -> float:
            if end <= start:
                return 0.0
            count = sum(1 for w in words if w["start_s"] < end and w["end_s"] > start)
            return count / (end - start)

        if scenes:
            candidates = [
                {
                    "start_s": float(s["start_s"]),
                    "end_s": float(s["end_s"]),
                    "text": s.get("label", "shot"),
                    "rank": float(s.get("score", 0.0))
                    + _density(s["start_s"], s["end_s"]),
                }
                for s in scenes
            ]
        else:
            candidates = self._word_groups(words)
        if not candidates:
            raise ValueError("video not analyzed")

        # Sort by rank descending, deal into rotating tiers so selection
        # cycles hot -> mid -> cold while staying seeded-random within a
        # tier. No reuse until every candidate is used once. Each
        # candidate carries its ranked index so reuse tracking is
        # consistent across tier and fallback pools.
        ranked = sorted(candidates, key=lambda c: (-c["rank"], c["start_s"]))
        for i, c in enumerate(ranked):
            c["_idx"] = i
        tier_size = max(1, len(ranked) // 3)
        tiers = [ranked[:tier_size], ranked[tier_size : 2 * tier_size], ranked[2 * tier_size :]]  # noqa: E501

        picked: list[dict] = []
        used: set[int] = set()
        total = 0.0
        tier_idx = 0
        # Cap per-event length so one long static scene can't blow the target.
        max_chunk = max(2.0, min(6.0, target_s / 4.0))
        while total < target_s and len(used) < len(ranked):
            pool = tiers[tier_idx % len(tiers)]
            avail = [c for c in pool if c["_idx"] not in used]
            if not avail:
                # Tier exhausted; fall back to any unused candidate.
                avail = [c for c in ranked if c["_idx"] not in used]
            if not avail:
                break
            choice = rng.choice(avail)
            used.add(choice["_idx"])
            dur = max(0.0, choice["end_s"] - choice["start_s"])
            if dur > max_chunk:
                win_start = rng.uniform(choice["start_s"], choice["end_s"] - max_chunk)
                choice = {
                    **choice,
                    "start_s": win_start,
                    "end_s": win_start + max_chunk,
                }
                dur = max_chunk
            picked.append(choice)
            total += dur
            tier_idx += 1
        return picked

    def _word_groups(self, words: list[dict]) -> list[dict]:
        """Group words into ~3s candidate segments (scene-less fallback)."""
        groups: list[dict] = []
        chunk: list[dict] = []
        for w in words:
            chunk.append(w)
            span = w["end_s"] - chunk[0]["start_s"]
            if span >= 3.0:
                groups.append(
                    {
                        "start_s": chunk[0]["start_s"],
                        "end_s": w["end_s"],
                        "text": " ".join(c["word"] for c in chunk),
                        "rank": len(chunk) / max(0.1, span),
                    }
                )
                chunk = []
        if chunk:
            span = chunk[-1]["end_s"] - chunk[0]["start_s"]
            groups.append(
                {
                    "start_s": chunk[0]["start_s"],
                    "end_s": chunk[-1]["end_s"],
                    "text": " ".join(c["word"] for c in chunk),
                    "rank": len(chunk) / max(0.1, span),
                }
            )
        return groups

    # ------------------------------------------------------------------
    # Mangling
    # ------------------------------------------------------------------

    def _mangle_event(self, base: dict, rng: random.Random) -> EDLEvent:
        """Apply seeded-random YTP mangling to one base segment dict."""
        if rng.random() < P_PLAIN:
            return EDLEvent(
                local_path=base["local_path"],
                start_s=base["start_s"],
                end_s=base["end_s"],
                text=base.get("text", ""),
            )
        effects: list[str] = []
        reverse = rng.random() < P_REVERSE
        if reverse:
            effects.append("reverse")
        if rng.random() < P_ZOOM:
            effects.append("zoom")
        if rng.random() < P_STUTTER:
            effects.append("stutter")
        return EDLEvent(
            local_path=base["local_path"],
            start_s=base["start_s"],
            end_s=base["end_s"],
            text=base.get("text", ""),
            reverse=reverse,
            speed=rng.choice(SPEEDS),
            gain_db=rng.choice(GAINS),
            effects=effects,
        )

    def _split_stutter(self, ev: EDLEvent, rng: random.Random) -> list[EDLEvent]:
        """Stutter events expand into 3 events, middle a ~0.15s blip."""
        if "stutter" not in ev.effects:
            return [ev]
        dur = ev.end_s - ev.start_s
        blip = min(STUTTER_BLIP_S, dur / 3.0)
        head = max(0.0, (dur - blip) / 2.0)
        # Keep the `stutter` marker on the head event so the 3-event group
        # is identifiable downstream; renderers may ignore it there.
        parts = [
            EDLEvent(
                local_path=ev.local_path,
                start_s=ev.start_s,
                end_s=ev.start_s + head,
                text=ev.text,
                reverse=ev.reverse,
                speed=ev.speed,
                gain_db=ev.gain_db,
                effects=list(ev.effects),
            ),
            EDLEvent(
                local_path=ev.local_path,
                start_s=ev.start_s + head,
                end_s=ev.start_s + head + blip,
                text=ev.text,
                reverse=ev.reverse,
                speed=ev.speed,
                gain_db=ev.gain_db,
                effects=[e for e in ev.effects if e != "stutter"],
            ),
            EDLEvent(
                local_path=ev.local_path,
                start_s=ev.start_s + head + blip,
                end_s=ev.end_s,
                text=ev.text,
                reverse=ev.reverse,
                speed=ev.speed,
                gain_db=ev.gain_db,
                effects=[e for e in ev.effects if e != "stutter"],
            ),
        ]
        return parts
