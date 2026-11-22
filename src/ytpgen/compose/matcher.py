"""Text-to-clips matcher: turn a sentence into source-video clips.

Resolution order per token (mode filters which strategies run):

* ``word``   — exact / normalized lookup in the words table
* ``phone``  — cover the token's ARPABET phones with indexed phone spans
* ``hybrid`` — word, then phone, then a naive syllable fallback

Tokens that resolve under no strategy are logged at WARNING and skipped;
the rest of the sentence still renders.
"""

from __future__ import annotations

import logging
import random
import re
import string
from typing import Any, Callable

from ytpgen.index import Index
from ytpgen.models import Clip, EDL, EDLEvent

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


# Vowel classes for the naive syllable splitter.
_VOWELS = "aeiouy"


def _tokenize(text: str) -> list[str]:
    """Lowercased word / number tokens; punctuation and whitespace split."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _normalize(token: str) -> str:
    """Strip surrounding punctuation and lowercase."""
    return token.strip(string.punctuation).strip().lower()


def _normalize_ipa(ipa: str) -> str:
    """Uppercase ASCII IPA letters; drop non-mapping punctuation."""
    out = []
    for ch in ipa:
        if ch.isascii() and ch.isalpha():
            out.append(ch.upper())
        elif ch.isalnum():
            out.append(ch)
        # else: punctuation/whitespace between phones is ignored
    return "".join(out)


#: Word-clip padding: whisper timestamps clip the first phoneme's attack,
#: so breathe on both sides — clamped by _pad_clip to the parent word AND
#: to the neighboring words' spans (never bleed into adjacent speech).
WORD_PAD_HEAD_S = 0.035
WORD_PAD_TAIL_S = 0.060
PHONE_PAD_HEAD_S = 0.015
PHONE_PAD_TAIL_S = 0.025
_GUARD_S = 0.020  # keep-out zone before the next / after the previous word


def _pad_clip(
    row: dict, start: float, end: float, kind: str
) -> tuple[float, float]:
    """Grow [start, end) by head/tail padding without swallowing speech.

    Three clamps, tightest wins:
      - parent word bounds (word_start_s / word_end_s)
      - previous word's end (prev_end_s) on the head side
      - next word's start (next_start_s) on the tail side
    so pads only ever extend into silence, never into adjacent words.
    """
    word_start = row.get("word_start_s")
    word_end = row.get("word_end_s")
    word_start = float(word_start) if word_start is not None else start
    word_end = float(word_end) if word_end is not None else end
    if kind == "phone":
        head, tail = PHONE_PAD_HEAD_S, PHONE_PAD_TAIL_S
    else:
        head, tail = WORD_PAD_HEAD_S, WORD_PAD_TAIL_S

    prev_end = row.get("prev_end_s")
    prev_end = float(prev_end) if prev_end is not None else None
    next_start = row.get("next_start_s")
    next_start = float(next_start) if next_start is not None else None

    lo = (prev_end + _GUARD_S) if prev_end is not None else word_start - head
    new_start = min(word_start, max(word_start - head, lo))
    hi = (next_start - _GUARD_S) if next_start is not None else word_end + tail
    new_end = max(word_end, min(word_end + tail, hi))
    return new_start, max(new_start, new_end)


def _pick_row(rows: list[dict]) -> dict:
    """Pick a row, honoring the pref-first ordering find_word returns.

    Falls back to uniform random choice when no row is flagged preferred.
    """
    if not rows:
        raise ValueError("no rows to pick from")
    pref = [r for r in rows if r.get("pref")]
    return random.choice(pref if pref else rows)


def _clip_from_row(row: dict, text: str, kind: str) -> Clip:
    """Build a padded Clip from a find_word/find_phones row."""
    start = float(row["start_s"])
    end = float(row["end_s"])
    new_start, new_end = _pad_clip(row, start, end, kind)
    return Clip(
        local_path=row["local_path"],
        start_s=new_start,
        end_s=new_end,
        source_id=int(row["source_id"]),
        text=text,
        kind=kind,
    )


class Matcher:
    """Resolve sentence tokens to clips from the ytpgen index."""

    #: IPA-inference hook, overridable (tests monkeypatch this).
    _ipa_infer: Callable[[str], str | None] | None = None

    _MODE_STRATEGIES: dict[str, list[str]] = {
        "word": ["exact"],
        "phone": ["phones"],
        # syllables before phones: whole-syllable word clips sound far
        # better than per-phoneme fragment confetti
        "hybrid": ["exact", "syllables", "phones"],
    }

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._source_ids: set[int] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_sentence(
        self,
        text: str,
        db: Index,
        mode: str = "hybrid",
        source_ids: set[int] | None = None,
    ) -> list[Clip]:
        """Resolve every token of *text* against *db*.

        source_ids restricts matching to those sources (None = all).
        Returns clips in sentence order. Tokens with no resolution at
        all are warned about and skipped.
        """
        strategies = self._MODE_STRATEGIES.get(mode)
        if strategies is None:
            raise ValueError(f"unknown mode: {mode!r}")
        self._source_ids = source_ids
        clips: list[Clip] = []
        for token in _tokenize(text):
            token_clips = self._resolve_token(token, db, strategies)
            if token_clips is None:
                logger.warning("no resolution for token %r; skipping", token)
                continue
            clips.extend(token_clips)
        return clips

    # ------------------------------------------------------------------
    # Token resolution
    # ------------------------------------------------------------------

    def _resolve_token(
        self, token: str, db: Index, strategies: list[str]
    ) -> list[Clip] | None:
        """First strategy that yields clips wins; None if all fail."""
        for name in strategies:
            resolver = getattr(self, f"_resolve_{name}")
            clips = resolver(token, db)
            if clips:
                return clips
        return None

    def _resolve_exact(self, token: str, db: Index) -> list[Clip] | None:
        """Exact / normalized word lookup."""
        candidates: list[dict] = []
        for probe in dict.fromkeys([token, _normalize(token)]):
            hits = db.find_word(probe, source_ids=self._source_ids)
            if hits:
                candidates = hits
                break
        if not candidates:
            return None
        row = _pick_row(candidates)
        return [_clip_from_row(row, token, kind="word")]

    def _resolve_phones(self, token: str, db: Index) -> list[Clip] | None:
        """Cover the token's ARPABET phones with indexed phone spans."""
        phones = self._token_phones(token)
        if not phones:
            return None
        return self._cover_phones(phones, db)

    def _resolve_syllables(self, token: str, db: Index) -> list[Clip] | None:
        """Naive syllable split; each syllable resolved as a word clip."""
        if len(token) < 4:
            return None
        syls = self._split_syllables(token)
        if len(syls) < 2:
            return None
        clips: list[Clip] = []
        for syl in syls:
            hits = db.find_word(syl, source_ids=self._source_ids)
            if not hits:
                return None  # every syllable must resolve or the fallback fails
            row = _pick_row(hits)
            clips.append(_clip_from_row(row, syl, kind="word"))
        return clips

    # ------------------------------------------------------------------
    # EDL assembly
    # ------------------------------------------------------------------

    @staticmethod
    def edl_from_clips(
        clips: list[Clip],
        fps: float,
        width: int,
        height: int,
        seed: int | None = None,
    ) -> EDL:
        """One EDLEvent per clip, same order as *clips*."""
        events = [
            EDLEvent(
                index=i,
                local_path=c.local_path,
                start_s=c.start_s,
                end_s=c.end_s,
                text=c.text,
                reverse=False,
                speed=1.0,
                gain_db=0.0,
                # stinger: 1-2 words cut out of a real line need a beat of air after them
                effects=["hold"] if c.kind == "phrase" and len(c.text.split()) <= 2 else [],
            )
            for i, c in enumerate(clips)
        ]
        return EDL(fps=fps, width=width, height=height, seed=seed, events=events)

    # ------------------------------------------------------------------
    # Phone machinery
    # ------------------------------------------------------------------

    def _token_phones(self, token: str) -> list[str]:
        """ARPABET phone sequence for *token*, or [] if unavailable."""
        ipa = self._infer_ipa(token)
        if not ipa:
            return []
        from ytpgen.analyze.phonemes.ipa_arpabet import ipa_to_arpabet

        return ipa_to_arpabet(ipa)

    #: Cached ESpeak backend (lazily created; None until first use).
    _espeak_backend: Any | None = None

    def _infer_ipa(self, token: str) -> str | None:
        """IPA for *token* via phonemizer ESpeak; None if unavailable."""
        if Matcher._ipa_infer is not None:
            return Matcher._ipa_infer(token)
        try:
            if Matcher._espeak_backend is None:
                from ytpgen.analyze.phonemes.espeak_backend import get_espeak_backend

                Matcher._espeak_backend = get_espeak_backend("en-us")
            if Matcher._espeak_backend is None:
                return None
            phones: list[str] = Matcher._espeak_backend.process([token], strip=True)
            return phones[0].strip() or None
        except Exception as exc:  # pragma: no cover - depends on espeak install
            logger.debug("phonemizer unavailable for %r: %s", token, exc)
            return None

    def _cover_phones(self, phones: list[str], db: Index) -> list[Clip] | None:
        """Greedy longest-prefix cover of *phones* with indexed spans.

        A segment may span several target phones when a single indexed
        row's ``arpabet`` contains the run (e.g. a whole-word row).
        """
        clips: list[Clip] = []
        i = 0
        while i < len(phones):
            seg = self._find_longest_segment(phones, i, db)
            if seg is None:
                return None
            row, span = seg
            clips.append(_clip_from_row(row, " ".join(span), kind="phone"))
            i += len(span)
        return clips or None

    def _find_longest_segment(
        self, phones: list[str], i: int, db: Index
    ) -> tuple[dict, list[str]] | None:
        """Longest ``phones[i:j]`` run covered by one find_phones hit.

        Among rows covering the same span, prefers the one whose parent
        word continues matching the following target phones longest —
        whole containing words beat single-phone fragments.
        """
        for j in range(len(phones), i, -1):
            span = phones[i:j]
            hits = db.find_phones(" ".join(span), source_ids=self._source_ids)
            if not hits:
                continue
            best: tuple[dict, list[str], int] | None = None
            for row in hits:
                indexed = [_normalize_ipa(p) for p in row["arpabet"].split()]
                pos = _contains_at(indexed, span)
                if pos < 0:
                    continue
                ext = 0
                if row.get("word_phonemes"):
                    wp = [_normalize_ipa(p) for p in str(row["word_phonemes"]).split()]
                    t, k = j, pos + len(span)
                    while t < len(phones) and k < len(wp) and phones[t] == wp[k]:
                        ext += 1
                        t += 1
                        k += 1
                if best is None or ext > best[2]:
                    best = (row, span, ext)
            if best is not None:
                return best[0], best[1]
        return None

    # ------------------------------------------------------------------
    # Syllable splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_syllables(token: str) -> list[str]:
        """Naive VCV syllable split (min 2 chars per syllable).

        Candidate boundaries sit at V|CV (onset) and V|CC (coda cluster)
        positions. The boundary closest to the token midpoint is chosen,
        then halves of >= 5 chars recurse, so common two-syllable words
        split near the middle (e.g. ``baseline`` -> ``base`` + ``line``).
        Returns ``[token]`` when no usable boundary exists.
        """
        chunks = Matcher._split_recursive(token)
        return chunks if len(chunks) >= 2 else [token]

    @staticmethod
    def _split_recursive(word: str) -> list[str]:
        """Midpoint binary VCV split; chunks shorter than 2 chars veto."""
        n = len(word)
        if n < 4:
            return [word]
        cuts: set[int] = set()
        for idx in range(1, n - 1):
            prev, cur, nxt = word[idx - 1], word[idx], word[idx + 1]
            if cur in _VOWELS and prev in _VOWELS:
                cuts.add(idx)  # V|V hiatus
            elif cur not in _VOWELS and prev in _VOWELS:
                cuts.add(idx)  # V|CV onset or V|CC coda
        if not cuts:
            return [word]
        mid = n / 2
        cut = min(cuts, key=lambda c: (abs(c - mid), c))
        head, tail = word[:cut], word[cut:]
        if len(head) < 2 or len(tail) < 2:
            return [word]
        if len(head) >= 5:
            head_parts = Matcher._split_recursive(head)
        else:
            head_parts = [head]
        if len(tail) >= 5:
            tail_parts = Matcher._split_recursive(tail)
        else:
            tail_parts = [tail]
        return head_parts + tail_parts


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------

def _contains_at(haystack: list[str], needle: list[str]) -> int:
    """Index of the first place *needle* appears in *haystack*, else -1."""
    if not needle:
        return -1
    for k in range(len(haystack) - len(needle) + 1):
        if haystack[k : k + len(needle)] == needle:
            return k
    return -1
