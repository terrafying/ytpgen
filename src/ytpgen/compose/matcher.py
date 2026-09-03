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


class Matcher:
    """Resolve sentence tokens to clips from the ytpgen index."""

    #: IPA-inference hook, overridable (tests monkeypatch this).
    _ipa_infer: Callable[[str], str | None] | None = None

    _MODE_STRATEGIES: dict[str, list[str]] = {
        "word": ["exact"],
        "phone": ["phones"],
        "hybrid": ["exact", "phones", "syllables"],
    }

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_sentence(
        self, text: str, db: Index, mode: str = "hybrid"
    ) -> list[Clip]:
        """Resolve every token of *text* against *db*.

        Returns clips in sentence order. Tokens with no resolution at
        all are warned about and skipped.
        """
        strategies = self._MODE_STRATEGIES.get(mode)
        if strategies is None:
            raise ValueError(f"unknown mode: {mode!r}")
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
            hits = db.find_word(probe)
            if hits:
                candidates = hits
                break
        if not candidates:
            return None
        row = random.choice(candidates)
        return [
            Clip(
                local_path=row["local_path"],
                start_s=float(row["start_s"]),
                end_s=float(row["end_s"]),
                source_id=int(row["source_id"]),
                text=token,
                kind="word",
            )
        ]

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
            hits = db.find_word(syl)
            if not hits:
                return None  # every syllable must resolve or the fallback fails
            row = random.choice(hits)
            clips.append(
                Clip(
                    local_path=row["local_path"],
                    start_s=float(row["start_s"]),
                    end_s=float(row["end_s"]),
                    source_id=int(row["source_id"]),
                    text=syl,
                    kind="word",
                )
            )
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
                effects=[],
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
            clips.append(
                Clip(
                    local_path=row["local_path"],
                    start_s=float(row["start_s"]),
                    end_s=float(row["end_s"]),
                    source_id=int(row["source_id"]),
                    text=" ".join(span),
                    kind="phone",
                )
            )
            i += len(span)
        return clips or None

    def _find_longest_segment(
        self, phones: list[str], i: int, db: Index
    ) -> tuple[dict, list[str]] | None:
        """Longest ``phones[i:j]`` run covered by one find_phones hit."""
        for j in range(len(phones), i, -1):
            span = phones[i:j]
            hits = db.find_phones(" ".join(span))
            if not hits:
                continue
            for row in hits:
                indexed = [_normalize_ipa(p) for p in row["arpabet"].split()]
                pos = _contains_at(indexed, span)
                if pos >= 0:
                    return row, span
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
