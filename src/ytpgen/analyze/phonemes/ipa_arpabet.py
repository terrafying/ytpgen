"""IPA -> ARPABET mapping for espeak 'en-us' output.

espeak-ng (via ``phonemizer``) emits IPA; downstream stages speak
ARPABET, so this module maps one onto the other:

- ``IPA_TO_arpabet`` maps IPA symbols to ARPABET. Several keys are
  multi-character (diphthongs ``aʊ aɪ eɪ oʊ ɔɪ``, affricates
  ``tʃ dʒ``); matching is greedy, longest key first, so ``aʊ`` never
  decomposes into ``a`` + ``ʊ``.
- Stress (``ˈ ˌ``) / length (``ː``) marks, separators, whitespace and
  combining diacritics are stripped before matching; the glottal stop
  ``ʔ`` maps to ``''`` (consumed, emits nothing); other unknown chars
  are skipped.
- Vowels follow CMUdict conventions (``e``->EY, ``o``->OW, ``ə``->AH,
  ``ɝ``/``ɚ``/``ɜ``->ER); the tap ``ɾ`` approximates to T.
"""

from __future__ import annotations

import unicodedata

# IPA symbol -> ARPABET phone ('' = consume but emit nothing).
IPA_TO_arpabet: dict[str, str] = {
    # Vowels (monophthongs)
    "ɑ": "AA", "a": "AA", "æ": "AE", "ʌ": "AH", "ə": "AH", "ɔ": "AO",
    "ʊ": "UH", "u": "UW", "ɪ": "IH", "i": "IY", "ɛ": "EH", "e": "EY",
    "ɝ": "ER", "ɚ": "ER", "ɜ": "ER",
    # Diphthongs (multi-char; must match before their components)
    "aʊ": "AW", "aɪ": "AY", "eɪ": "EY", "oʊ": "OW", "ɔɪ": "OY",
    # Consonants
    "p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "ɡ": "G", "g": "G",
    "f": "F", "v": "V", "θ": "TH", "ð": "DH", "s": "S", "z": "Z",
    "ʃ": "SH", "ʒ": "ZH", "tʃ": "CH", "dʒ": "JH", "h": "HH",
    "m": "M", "n": "N", "ŋ": "NG", "l": "L", "r": "R", "ɹ": "R",
    "w": "W", "j": "Y", "ɾ": "T",
    # Skipped phones
    "ʔ": "",
}

_STRIP = set("ˈˌːˑ.·|()")  # stress, length, syllable separators
_KEYS_BY_LEN = sorted(IPA_TO_arpabet, key=len, reverse=True)


def _normalize_ipa(ipa: str) -> str:
    """Drop stress/length marks, separators, whitespace and diacritics."""
    return "".join(
        ch for ch in ipa.strip()
        if ch not in _STRIP and not ch.isspace() and not unicodedata.combining(ch)
    )


def ipa_to_pairs(ipa: str) -> list[tuple[str, str]]:
    """Greedily split IPA into (symbol, ARPABET) pairs, skipping unknowns/''."""
    s = _normalize_ipa(ipa)
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(s):
        for key in _KEYS_BY_LEN:
            if s.startswith(key, i):
                arp = IPA_TO_arpabet[key]
                if arp:
                    out.append((key, arp))
                i += len(key)
                break
        else:
            i += 1  # unknown character: skip it
    return out


def ipa_to_arpabet(ipa: str) -> list[str]:
    """Convert an IPA string (one word) to a list of ARPABET phones."""
    return [arp for _, arp in ipa_to_pairs(ipa)]
