"""Say a fixed text (a quote) with the library: longest real runs of speech first.

Greedy cover: at each position take the longest stretch of the text that one speaker said
contiguously (a phrase span), ties broken at random so repeats vary; single words come from
phrases too; only words nobody said fall back to the word/phone matcher.
"""

from __future__ import annotations

import random
from collections import defaultdict

from ytpgen.compose.matcher import Matcher
from ytpgen.compose.phrases import Phrase
from ytpgen.compose.planner import _words, phrase_clip
from ytpgen.index import Index
from ytpgen.models import Clip


def cover(text: str, phrases: list[Phrase], db: Index, matcher: Matcher, seed: int = 0,
          source_ids: set[int] | None = None, mode: str = "hybrid") -> tuple[list[Clip], list[tuple[str, str]]]:
    """Clips that say *text*, plus a report [(span, "phrase:N words" | "matched" | "missing")].
    mode="word": a word nobody said is reported missing, never spliced from syllables/phones (those are unintelligible)."""
    rng = random.Random(seed)
    occ: dict[str, list[tuple[Phrase, int]]] = defaultdict(list)
    for ph in phrases:
        for k, w in enumerate(ph.words):
            occ[w].append((ph, k))
    words = _words(text)
    clips: list[Clip] = []
    report: list[tuple[str, str]] = []
    i = 0
    while i < len(words):
        best: list[tuple[Phrase, int, int]] = []
        for ph, k in occ.get(words[i], []):
            n = 1
            while k + n < len(ph.words) and i + n < len(words) and ph.words[k + n] == words[i + n]:
                n += 1
            if not best or n > best[0][2]:
                best = [(ph, k, n)]
            elif n == best[0][2]:
                best.append((ph, k, n))
        if best:
            ph, k, n = rng.choice(best)
            clips.append(phrase_clip(ph, k, k + n))
            report.append((" ".join(words[i:i + n]), f"phrase:{n}"))
            i += n
            continue
        got = matcher.resolve_sentence(words[i], db, mode=mode, source_ids=source_ids)
        clips.extend(got)
        report.append((words[i], "matched" if got else "missing"))
        i += 1
    return clips, report
