"""Scene detection + word-level transcription for the ytpgen pipeline.

``Segmenter`` runs one analysis pass over a source video: scenes via
ffmpeg ``select='gt(scene,T)'`` + ``metadata=print`` (cuts bookended with
0.0 / duration, merged to honor ``scene.min_scene_len_s``); words via
faster-whisper word timestamps, each token phonemized with espeak
'en-us' and mapped to ARPABET via
:mod:`ytpgen.analyze.phonemes.ipa_arpabet`. Phone timing is a coarse even
split of the word span — real alignment is the benchmark engine's job.
Whisper tokens are lowercased/punctuation-stripped before storage.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

from ytpgen.models import PhoneSpan, Scene, SourceVideo, Word

logger = logging.getLogger(__name__)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS scenes (
    source_id INTEGER, start_s REAL, end_s REAL, label TEXT, score REAL);
CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER, word TEXT, word_lower TEXT, start_s REAL, end_s REAL,
    phonemes TEXT);
CREATE TABLE IF NOT EXISTS words_phones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER, word_rowid INTEGER, ipa TEXT, arpabet TEXT,
    start_s REAL, end_s REAL);
"""

_POINTS = ".,!?;:\"'"  # punctuation stripped from whisper tokens
_PUNCT_TABLE = str.maketrans("", "", _POINTS)
_PTS_TIME_RE = re.compile(r"pts_time:(\d+(?:\.\d+)?)")
_SCENE_SCORE_RE = re.compile(r"lavfi\.scene_score=(\d+(?:\.\d+)?)")

_WARNED_NO_PHONEMIZER = {"flag": False}  # warn only once per process
_ESPEAK_LIB_CANDIDATES = (  # Homebrew macOS installs
    "/opt/homebrew/lib/libespeak-ng.dylib", "/usr/local/lib/libespeak-ng.dylib",
)


def merge_boundaries(
    times: list[float], min_len: float, duration: float
) -> list[float]:
    """Merge scene cuts so no segment is shorter than min_len.

    ``times`` may be unsorted, duplicated, or out of range. Returns
    sorted, deduped boundaries, always 0.0-first and ``duration``-last
    (when > 0). Cuts within ``min_len`` of a neighbor or the video end
    are dropped (earlier of a close pair kept; the tail is absorbed).
    """
    duration = max(0.0, duration)
    min_len = max(0.0, min_len)
    if duration <= 0.0:
        return [0.0]
    internal = sorted(
        {t for t in times if 0.0 < t < duration and duration - t >= min_len}
    )
    out: list[float] = [0.0]
    for t in internal:
        if t - out[-1] >= min_len:
            out.append(t)
    out.append(duration)
    return out


def _clean_token(raw: str) -> str:
    """Strip punctuation anywhere in a whisper token and lowercase it."""
    return raw.translate(_PUNCT_TABLE).strip().lower()


def _decode_audio(video_path: Path):  # noqa: ANN202 - np.ndarray | str
    """16 kHz mono float32 samples decoded by ffmpeg. faster-whisper's own PyAV decode returns garbage for some files
    (Prelinger .m4v: Whisper then hears only "." and the film gets 0 words); fall back to the path if ffmpeg fails."""
    import numpy as np

    try:
        pcm = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video_path), "-map", "0:a:0", "-ac", "1",
                              "-ar", "16000", "-f", "s16le", "-"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("ffmpeg audio decode failed for %s (%s); letting faster-whisper decode", video_path, exc)
        return str(video_path)
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0


class Segmenter:
    """Scene detection + word/phone transcription in one analysis pass."""

    def __init__(self, cfg: dict, db: sqlite3.Connection) -> None:
        self.cfg = cfg or {}
        self.db = db
        scene_cfg = self.cfg.get("scene") or {}
        self.threshold = float(scene_cfg.get("threshold", 27.0)) / 100.0
        self.min_scene_len_s = float(scene_cfg.get("min_scene_len_s", 1.0))
        self.model_size = str(self.cfg.get("model_size", "base"))
        self.compute_type = str(self.cfg.get("compute_type", "int8"))
        self.language = str(self.cfg.get("language", "en"))
        self._model = None  # cached whisper model (fake in tests)
        self._phonemize_fn = None  # espeak callable, or None = unavailable
        self._phonemizer_checked = False
        self.db.executescript(_CREATE_SQL)

    # ------------------------------------------------------------ scene detect

    def detect_scenes(self, video_path: Path) -> list[Scene]:
        """Detect scene cuts with ffmpeg; boundaries bookend [0.0, duration]."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found on PATH — e.g. `brew install ffmpeg`")
        duration = self._ffprobe_duration(video_path)
        cmd = [
            ffmpeg, "-nostdin", "-hide_banner", "-nostats",
            "-i", str(video_path), "-an", "-sn",
            "-vf", f"select='gt(scene,{self.threshold:.4f})',metadata=print",
            "-f", "null", "-",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1800, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"ffmpeg scene detection failed: {exc}") from exc
        times, scores = self._parse_scene_output(proc.stderr)
        boundaries = merge_boundaries(times, self.min_scene_len_s, duration)
        if len(boundaries) < 2:
            logger.warning("No usable duration for %s; no scenes", video_path)
            return []
        scenes: list[Scene] = []
        for start, end in zip(boundaries, boundaries[1:]):
            # Score = strongest scene-change magnitude inside the segment.
            score = max(
                (s for t, s in scores.items() if start < t <= end), default=0.0
            )
            scenes.append(Scene(start_s=start, end_s=end, label="shot", score=score))
        logger.info("Detected %d scenes in %s", len(scenes), video_path)
        return scenes

    @staticmethod
    def _parse_scene_output(stderr: str) -> tuple[list[float], dict[float, float]]:
        """Parse (cut times, {time: score}) from ffmpeg metadata=print output."""
        times: list[float] = []
        scores: dict[float, float] = {}
        pending: float | None = None
        for line in stderr.splitlines():
            m = _PTS_TIME_RE.search(line)
            if m:
                pending = float(m.group(1))
                times.append(pending)
            elif (m := _SCENE_SCORE_RE.search(line)) and pending is not None:
                scores[pending] = float(m.group(1))
        return times, scores

    @staticmethod
    def _ffprobe_duration(video_path: Path) -> float:
        """Duration in seconds via ffprobe; 0.0 if it cannot be determined."""
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            logger.warning("ffprobe not found; cannot read video duration")
            return 0.0
        cmd = [ffprobe, "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, check=False
            )
            return float(proc.stdout.strip())
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            logger.warning("ffprobe duration failed for %s: %s", video_path, exc)
            return 0.0

    # -------------------------------------------------------------- transcribe

    def _load_model(self):  # noqa: ANN202 - faster_whisper.WhisperModel
        """Load the whisper model once (patched with a fake in tests)."""
        if self._model is None:
            from faster_whisper import WhisperModel  # deferred: heavy import

            logger.info("Loading faster-whisper model %s (%s)",
                        self.model_size, self.compute_type)
            self._model = WhisperModel(
                self.model_size, device="auto", compute_type=self.compute_type
            )
        return self._model

    def _resolve_phonemize(self):  # noqa: ANN202
        """Return an espeak IPA phonemizer callable, or None if unavailable."""
        if self._phonemizer_checked:
            return self._phonemize_fn
        self._phonemizer_checked = True
        self._rescue_espeak_library()
        # cfg['language'] is a whisper code ('en'); espeak needs 'en-us'.
        lang = "en-us" if self.language.startswith("en") else self.language
        try:
            from phonemizer.backend import EspeakBackend

            backend = EspeakBackend(lang)
            # One probe call so a broken install fails here, not per word.
            backend.phonemize(["probe"], strip=True)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            self._warn_no_phonemizer(exc)
            return None
        self._phonemize_fn = lambda tokens: backend.phonemize(tokens, strip=True)
        return self._phonemize_fn

    @staticmethod
    def _rescue_espeak_library() -> None:
        """Point phonemizer at a Homebrew espeak-ng dylib if ctypes can't."""
        if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
            return
        import ctypes.util  # needed for ctypes.util to exist

        if ctypes.util.find_library("espeak-ng") or ctypes.util.find_library("espeak"):
            return
        for candidate in _ESPEAK_LIB_CANDIDATES:
            if Path(candidate).exists():
                logger.info("Using espeak-ng library at %s", candidate)
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = candidate
                return

    @staticmethod
    def _warn_no_phonemizer(exc: Exception) -> None:
        """Warn once that phonemization is disabled."""
        if not _WARNED_NO_PHONEMIZER["flag"]:
            _WARNED_NO_PHONEMIZER["flag"] = True
            logger.warning(
                "phonemizer/espeak-ng unavailable (%s) — words will have empty"
                " phonemes; install espeak-ng for IPA/ARPABET mapping", exc)

    def _phonemize(self, tokens: list[str]) -> list[str]:
        """IPA for each cleaned token ('' entries where unavailable)."""
        fn = self._resolve_phonemize()
        if fn is None or not tokens:
            return [""] * len(tokens)
        try:
            return list(fn(tokens))
        except Exception as exc:  # noqa: BLE001 — per-batch resilience
            logger.warning("phonemize failed for %d tokens: %s", len(tokens), exc)
            return [""] * len(tokens)

    def transcribe(self, video_path: Path, source_id: int) -> list[Word]:
        """Whisper-transcribe with word timestamps; attach ARPABET phones."""
        from ytpgen.analyze.phonemes.ipa_arpabet import ipa_to_pairs

        model = self._load_model()
        segments, _info = model.transcribe(
            _decode_audio(video_path), word_timestamps=True, language=self.language,
            # a music-only opening makes Whisper emit "." and, conditioned on it, keep emitting "." for the whole film
            # (Destination Earth: 0 -> 1273 words)
            condition_on_previous_text=False,
        )
        # Collect raw tokens first so phonemization is one batch call.
        raw: list[tuple[str, float, float]] = []
        for segment in segments:
            for w in segment.words or []:
                raw.append((w.word, float(w.start), float(w.end)))
        if not raw:
            logger.info("No words transcribed from %s", video_path)
            return []
        ipa_list = self._phonemize([_clean_token(t) for t, _, _ in raw])

        words: list[Word] = []
        for (token, start, end), ipa in zip(raw, ipa_list):
            cleaned = _clean_token(token)
            if not cleaned:
                continue  # pure punctuation / noise token
            word = Word(word=cleaned, start_s=start, end_s=end, source_id=source_id)
            # Even-split the word span across its phones (coarse split).
            pairs = ipa_to_pairs(ipa) if ipa else []  # [(ipa_sym, arpabet)]
            step = max(0.0, end - start) / len(pairs) if pairs else 0.0
            word.phones = [
                PhoneSpan(ipa=sym, arpabet=arp, start_s=start + i * step,
                          end_s=start + (i + 1) * step)
                for i, (sym, arp) in enumerate(pairs)
            ]
            word.phonemes = " ".join(arp for _, arp in pairs)
            words.append(word)
        logger.info("Transcribed %d words from %s", len(words), video_path)
        return words

    # ----------------------------------------------------------------- analyze

    def analyze(self, video: SourceVideo | dict, db: sqlite3.Connection) -> dict:
        """Full pass: scenes + words written to the db. Never raises per video.

        Accepts a SourceVideo or a plain mapping with at least video_id and
        local_path (title optional).
        """
        if not isinstance(video, SourceVideo):
            video = SourceVideo.model_validate(video)
        if hasattr(db, "conn"):  # accept Index or raw sqlite3.Connection
            db = db.conn
        result: dict = {"scenes": 0, "words": 0}
        source_id = self._resolve_source_id(db, video.video_id)

        try:
            scenes = self.detect_scenes(Path(video.local_path))
            db.execute("DELETE FROM scenes WHERE source_id = ?", (source_id,))
            db.executemany(
                "INSERT INTO scenes (source_id, start_s, end_s, label, score)"
                " VALUES (?, ?, ?, ?, ?)",
                [(source_id, s.start_s, s.end_s, s.label, s.score) for s in scenes],
            )
            result["scenes"] = len(scenes)
            db.commit()  # before transcription: holding this write lock through a long Whisper run blocks `ytp tag` and the web UI
        except Exception as exc:  # noqa: BLE001 — one bad file must not kill a batch
            logger.error("Scene detection failed for %s: %s", video.local_path, exc)
            result["error"] = str(exc)

        try:
            word_rows = self.transcribe(Path(video.local_path), source_id=source_id)
            db.execute("DELETE FROM words_phones WHERE source_id = ?", (source_id,))
            db.execute(
                "DELETE FROM words WHERE source_id = ?", (source_id,)
            )
            db.commit()
            phone_rows: list[tuple] = []
            for w in word_rows:
                cur = db.execute(
                    "INSERT INTO words (source_id, word, word_lower, start_s, end_s, phonemes)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (source_id, w.word, w.word.lower(), w.start_s, w.end_s, w.phonemes),
                )
                word_rowid = cur.lastrowid  # same transaction as the words insert
                for p in w.phones:
                    phone_rows.append((source_id, word_rowid, p.ipa, p.arpabet,
                                       p.start_s, p.end_s))
            if phone_rows:
                db.executemany(
                    "INSERT INTO words_phones (source_id, word_rowid, ipa,"
                    " arpabet, start_s, end_s) VALUES (?, ?, ?, ?, ?, ?)",
                    phone_rows,
                )
            result["words"] = len(word_rows)
        except Exception as exc:  # noqa: BLE001 — one bad file must not kill a batch
            logger.error("Transcription failed for %s: %s", video.local_path, exc)
            result.setdefault("error", str(exc))

        db.commit()
        logger.info(
            "Analyze %s: scenes=%d words=%d",
            video.local_path, result["scenes"], result["words"],
        )
        return result

    @staticmethod
    def _resolve_source_id(db: sqlite3.Connection, video_id: str) -> int:
        """Map video_id to the INTEGER sources.id (0 if unknown/unmapped)."""
        try:
            row = db.execute("SELECT id FROM sources WHERE video_id = ?",
                             (video_id,)).fetchone()
        except sqlite3.Error:
            return 0
        return int(row[0]) if row else 0
