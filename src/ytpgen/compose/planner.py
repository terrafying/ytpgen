"""v0.2 planner: let an LLM write *with* the footage instead of forcing a script through it.

The LLM sees every phrase the library actually contains and composes a short
piece (funny or deep) mostly from whole phrases, with a few glue words that the
v0.1 Matcher fills from words/phones as a last resort. Candidate plans are
scored by how much they cut (fewer, longer clips win); verify.py then checks
that the render is still comprehensible.
"""

from __future__ import annotations

import json
import logging
import random
import re
import urllib.request
from dataclasses import dataclass, field

from ytpgen.compose.matcher import Matcher
from ytpgen.compose.phrases import Phrase
from ytpgen.index import Index
from ytpgen.models import Clip

logger = logging.getLogger(__name__)

PAD_HEAD_S, PAD_TAIL_S, GUARD_S = 0.04, 0.08, 0.02
MIN_CLIP_S = 0.15
KIND_COST = {"phrase": 0.0, "word": 1.0, "phone": 3.0}
SPLICE_COST = 0.5  # every cut, on top of the clip's kind cost
SOURCE_CHANGE_COST = 0.25
RETELL_COST = 1.5  # next clip is just what came next in the film: retelling, not recontextualising
RETELL_GAP_S = 2.0
DOMINANCE_COST = 2.0  # per word, scaled by how far one film's share of clips exceeds MAX_SHARE
MAX_SHARE = 0.5  # the LLM gravitates to the most famous film (Duck and Cover) and retells it
MISSING_COST = 5.0  # a glue word nothing could say
MAX_PHRASES = 1500  # shortcut: random sample above this; upgrade to retrieval for multi-hour libraries

TONES = {
    "funny": "absurd and comic: recontextualise innocent lines into non sequiturs, escalation, "
             "callbacks and deadpan punchlines, the way a great YouTube Poop or sentence-mix does",
    "deep": "sober and unexpectedly profound: find a reading of these lines that says something true "
            "and a little unsettling about people, fear, love, technology or time",
    "deadpan": "deadpan: earnest mid-century advice played completely straight until it lands as brutal "
               "or absurd about the listener personally (e.g. 'people don't like you'); escalate the "
               "personal judgement, never wink",
}

SYSTEM = """You compose new short spoken pieces by re-cutting old film footage. You may ONLY use
lines that exist in the footage, listed below as `id: words`. The audience hears the original
voices, so the message must emerge from what was really said.

Rules:
- Build the piece mostly from whole lines: {{"p": <id>}}.
- You may use a contiguous part of a line: {{"p": <id>, "words": "<exact consecutive words from it>"}}.
{glue}
- Do NOT retell the original films. Pull lines out of their context and set them against lines
  from elsewhere (other scenes, other films) so they mean something new. Never use two lines
  that follow each other in the original. Draw on at least three different films, and no
  film should supply more than half the lines.
- When scenes are described (## headers say who is on screen), make the cuts make visual sense
  too: keep the same character on screen through a back-and-forth, cut to a matching or reaction
  shot, and only put a line over a mismatched face when that mismatch is the joke.
- Use 2-4 STINGERS: a single word or two cut out of a real line ({{"p": <id>, "words": "but"}}) and
  dropped in as an interjection: "But!", "Don't.", "No.", "Cover!", a name. They punctuate, interrupt
  and reverse what was just said. Repeating a stinger or an earlier line later as a callback is good.
- Aim for {n_lines} lines, about {seconds} seconds spoken. Order matters: setup, turn, payoff.
- Tone: {tone}.
{theme}
Reply with JSON only: {{"title": "...", "lines": [ ... ]}}"""


@dataclass
class Plan:
    title: str
    lines: list[dict]
    clips: list[Clip]
    intended: str
    missing: int
    cost: float
    judged: dict = field(default_factory=dict)  # judge.py unexpected-congruence result


def phrase_clip(ph: Phrase, i: int = 0, j: int | None = None) -> Clip:
    """Clip for words [i, j) of a phrase, padded into silence only."""
    j = len(ph.words) if j is None else j
    lo = ph.ends[i - 1] if i > 0 else ph.prev_end_s
    hi = ph.starts[j] if j < len(ph.words) else ph.next_start_s
    start = ph.starts[i] - PAD_HEAD_S
    end = ph.ends[j - 1] + PAD_TAIL_S
    if lo is not None:
        start = max(start, lo + GUARD_S)
    if hi is not None:
        end = min(end, hi - GUARD_S)
    start, end = min(start, ph.starts[i]), max(end, ph.ends[j - 1])
    end = max(end, start + MIN_CLIP_S)  # whisper sometimes times a word at zero length
    return Clip(local_path=ph.local_path, start_s=max(0.0, start), end_s=end,
                source_id=ph.source_id, text=" ".join(ph.words[i:j]), kind="phrase")


def plan_cost(clips: list[Clip], missing: int) -> float:
    """Cutting cost per spoken word: lower = fewer, longer, same-source clips."""
    if not clips:
        return float("inf")
    cost = sum(KIND_COST.get(c.kind, 1.0) for c in clips)
    cost += SPLICE_COST * (len(clips) - 1)
    cost += SOURCE_CHANGE_COST * sum(a.source_id != b.source_id for a, b in zip(clips, clips[1:]))
    cost += RETELL_COST * sum(a.source_id == b.source_id and 0 <= b.start_s - a.end_s < RETELL_GAP_S
                              for a, b in zip(clips, clips[1:]))
    cost += MISSING_COST * missing
    words = sum(len(c.text.split()) for c in clips if c.kind != "phone") or 1
    top = max(sum(c.source_id == s for c in clips) for s in {c.source_id for c in clips}) / len(clips)
    return cost / words + DOMINANCE_COST * max(0.0, top - MAX_SHARE)


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def resolve(lines: list[dict], phrases: dict[int, Phrase], db: Index, matcher: Matcher,
            source_ids: set[int] | None = None) -> tuple[list[Clip], str, int]:
    """Turn planner lines into clips; returns (clips, intended text, missing glue words)."""
    clips: list[Clip] = []
    said: list[str] = []
    missing = 0
    for line in lines:
        if "p" in line:
            ph = phrases.get(int(line["p"])) if str(line["p"]).lstrip("-").isdigit() else None
            if ph is None:
                logger.warning("planner cited unknown phrase %r", line["p"])
                continue
            span = ph.find(_words(line["words"])) if line.get("words") else None
            i, j = span or (0, len(ph.words))
            clips.append(phrase_clip(ph, i, j))
            said.extend(ph.words[i:j])
        elif line.get("say"):
            for w in _words(line["say"]):
                got = matcher.resolve_sentence(w, db, mode="hybrid", source_ids=source_ids)
                if got:
                    clips.extend(got)
                    said.append(w)
                else:
                    missing += 1
    return clips, " ".join(said), missing


def _listing(pool: list[Phrase]) -> str:
    """Lines grouped under a scene header saying who is on screen (when scenes are tagged)."""
    from ytpgen.analyze.tag import describe

    out, last = [], None
    for p in pool:
        key = (p.source_id, p.scene_idx)
        if key != last and p.extra.get("visual"):
            out.append(f"## film {p.source_id}, scene {p.scene_idx}: {describe(p.extra['visual'])}")
        last = key
        out.append(f"{p.id}: {p.text}")
    return "\n".join(out)


GLUE = ('- Only when no line can carry the idea, add 1-3 glue words: {"say": "<words>"}. These are\n'
        '  stitched from single words or syllables and sound worse, so use them rarely.')
FOUND_ONLY = "- Never add words of your own: every line must be a real line or part of one."


def build_prompt(phrases: list[Phrase], tone: str, theme: str | None, seconds: int, seed: int,
                 found_only: bool = False) -> list[dict]:
    pool = phrases
    if len(pool) > MAX_PHRASES:
        pool = sorted(random.Random(seed).sample(pool, MAX_PHRASES), key=lambda p: p.id)
    listing = _listing(pool)
    system = SYSTEM.format(
        n_lines=max(4, seconds // 3), seconds=seconds, tone=TONES.get(tone, tone),
        glue=FOUND_ONLY if found_only else GLUE,
        theme=f"- Theme to explore: {theme}." if theme else "- No theme given: find the most striking message the footage wants to say.",
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": "Footage lines:\n" + listing}]


def chat(cfg: dict, messages: list[dict], temperature: float, seed: int) -> str:
    pc = cfg.get("planner", {})
    body = {
        "model": pc.get("model", "local"), "messages": messages, "temperature": temperature,
        "seed": seed, "max_tokens": pc.get("max_tokens", 2048),
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(pc.get("base_url", "http://127.0.0.1:8099/v1") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=pc.get("timeout_s", 900)) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def parse(reply: str) -> tuple[str, list[dict]]:
    m = re.search(r"\{.*\}", reply, re.S)
    data = json.loads(m.group(0)) if m else {}
    lines = [l for l in data.get("lines", []) if isinstance(l, dict)]
    return str(data.get("title", "untitled")), lines


def make_plans(cfg: dict, db: Index, phrases: list[Phrase], tone: str, theme: str | None,
               seconds: int = 30, n: int = 4, seed: int = 0, llm=chat,
               source_ids: set[int] | None = None, found_only: bool = False) -> list[Plan]:
    """n LLM drafts, resolved and sorted by cutting cost (best first).

    found_only: no LLM-written glue words; every word is a real line from the footage.
    """
    by_id = {p.id: p for p in phrases}
    matcher = Matcher(cfg)
    messages = build_prompt(phrases, tone, theme, seconds, seed, found_only)
    plans: list[Plan] = []
    for k in range(n):
        try:
            title, lines = parse(llm(cfg, messages, temperature=0.9, seed=seed + k))
        except Exception as exc:  # noqa: BLE001 - one bad draft shouldn't sink the batch
            logger.warning("draft %d failed: %s", k, exc)
            continue
        if found_only:
            lines = [l for l in lines if "p" in l]
        clips, intended, missing = resolve(lines, by_id, db, matcher, source_ids)
        if clips:
            plans.append(Plan(title, lines, clips, intended, missing, plan_cost(clips, missing)))
    return sorted(plans, key=lambda p: p.cost)
