#!/usr/bin/env python3
"""Benchmark phoneme-extraction engines on real indexed audio.

Engines:
  whisper_g2p  — pipeline default: faster-whisper word timestamps + espeak
                 G2P per word, phones evenly split across the word span.
  allosaurus   — CTC phone recognizer (pip install ".[bench]"), native
                 phone timing from the acoustic model.

Scoring (no forced alignment ground truth available):
  - phone-sequence Levenshtein distance between engines, normalized
    (each engine vs whisper_g2p reference)
  - timing sanity: fraction of an engine's phone spans inside whisper
    word spans (skipped when an engine emits no timestamps)
  - runtime per minute of audio

Usage:
    .venv/bin/python scripts/bench_phonemes.py <video_id|media_path> [--engine all]

Writes reports to <workspace>/bench/.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ytpgen.config import load_config, workspace_dir  # noqa: E402

logger = logging.getLogger("bench")


@dataclass
class EngineResult:
    name: str
    phones: list[dict] = field(default_factory=list)  # {ipa, arpabet, start_s, end_s}
    runtime_s: float = 0.0
    notes: str = ""
    ok: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_wav(video_path: Path, out_wav: Path) -> Path:
    """16k mono wav for phone recognition."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000", str(out_wav),
        ],
        check=True,
    )
    return out_wav


def media_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def levenshtein(a: list[str], b: list[str]) -> int:
    """Classic DP edit distance between two phone sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def strip_stress(phones: list[str]) -> list[str]:
    import re

    return [re.sub(r"\d", "", p).upper() for p in phones]


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


def engine_whisper_g2p(video_path: Path, wav: Path, cfg: dict) -> EngineResult:
    """Reference engine: whisper words -> espeak phones -> even split."""
    from faster_whisper import WhisperModel

    from ytpgen.analyze.phonemes.espeak_backend import get_espeak_backend
    from ytpgen.analyze.phonemes.ipa_arpabet import ipa_to_arpabet

    t0 = time.time()
    model = WhisperModel(cfg.get("model_size", "base"),
                         compute_type=cfg.get("compute_type", "int8"))
    segments, _ = model.transcribe(str(video_path), word_timestamps=True,
                                   language=cfg.get("language", "en"))
    backend = get_espeak_backend("en-us")
    phones: list[dict] = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            token = w.word.strip().lower().strip(".,!?;:\"'")
            if not token:
                continue
            ipa = ""
            if backend is not None:
                try:
                    ipa = backend.process([token], strip=True)[0]
                except Exception:
                    ipa = ""
            arp = ipa_to_arpabet(ipa) if ipa else []
            if not arp:
                arp = [token]  # unknown-word placeholder, flagged by being alpha
            span = (w.end - w.start) / max(1, len(arp))
            for i, p in enumerate(arp):
                phones.append({
                    "ipa": ipa, "arpabet": p,
                    "start_s": w.start + i * span,
                    "end_s": w.start + (i + 1) * span,
                })
    return EngineResult("whisper_g2p", phones, time.time() - t0)


def engine_allosaurus(video_path: Path, wav: Path, cfg: dict) -> EngineResult:
    """CTC phone recognizer; needs `pip install .[bench]` once."""
    try:
        try:
            from allosaurus.app import read_recognizer  # type: ignore[import]

            model = read_recognizer("uni2005")
        except ImportError:
            # older allosaurus
            from allosaurus.app import read_recognition_model  # type: ignore[import]

            model = read_recognition_model("uni2005")
    except Exception as exc:
        return EngineResult("allosaurus", ok=False,
                            notes=f"not installed: {exc}")
    t0 = time.time()
    out = model.recognize(str(wav), timestamp=True) if "timestamp" in (
        __import__("inspect").signature(model.recognize).parameters
    ) else model.recognize(str(wav))
    phones: list[dict] = []
    # allosaurus emits lines: "<token>" or "<start> <end> <phone>" depending
    # on version/timestamp config; parse both shapes defensively.
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) == 3:
            try:
                start, end, ph = float(parts[0]), float(parts[1]), parts[2]
            except ValueError:
                continue
            phones.append({"ipa": ph, "arpabet": ph.upper(),
                           "start_s": start, "end_s": end})
        elif len(parts) == 1:
            phones.append({"ipa": parts[0], "arpabet": parts[0].upper(),
                           "start_s": -1.0, "end_s": -1.0})
    return EngineResult("allosaurus", phones, time.time() - t0)


ENGINES = {"whisper_g2p": engine_whisper_g2p, "allosaurus": engine_allosaurus}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score(results: list[EngineResult], duration_s: float) -> list[dict]:
    ref = next((r for r in results if r.name == "whisper_g2p" and r.ok), None)
    rows = []
    for r in results:
        row: dict = {
            "engine": r.name, "ok": r.ok, "phones": len(r.phones),
            "runtime_s": round(r.runtime_s, 2),
            "runtime_per_min": round(r.runtime_s / max(duration_s / 60, 1e-6), 2),
            "notes": r.notes,
        }
        if ref and r.ok and r.name != ref.name:
            dist = levenshtein(strip_stress([p["arpabet"] for p in ref.phones]),
                               strip_stress([p["arpabet"] for p in r.phones]))
            n = max(len(ref.phones), 1)
            row["lev_vs_whisper_g2p"] = dist
            row["phone_acc_vs_whisper_g2p"] = round(1.0 - dist / n, 3)
        timed = [p for p in r.phones if p["start_s"] >= 0]
        if ref and timed and r.ok:
            inside = sum(
                1 for p in timed
                if any(ws["start_s"] - 0.05 <= p["start_s"]
                       and p["end_s"] <= ws["end_s"] + 0.05
                       for ws in ref.phones)
            )
            row["pct_inside_word_spans"] = round(100.0 * inside / len(timed), 1)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", help="video_id (from index) or media file path")
    ap.add_argument("--engine", default="all",
                    help="comma list: whisper_g2p,allosaurus or 'all'")
    ap.add_argument("--config", default="config/ytp.yaml")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config if Path(args.config).exists() else None)
    ws = workspace_dir(cfg)

    path = Path(args.target)
    if not path.exists():
        db_path = ws / "index.db"
        if not db_path.exists():
            print("target not found and no index.db in workspace", file=sys.stderr)
            return 1
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT local_path FROM sources WHERE video_id = ?",
                           (args.target,)).fetchone()
        if row is None:
            print(f"unknown video_id: {args.target}", file=sys.stderr)
            return 1
        path = Path(row["local_path"])
    if not path.exists():
        print(f"media file missing: {path}", file=sys.stderr)
        return 1

    duration = media_duration(path)
    print(f"target: {path} ({duration:.1f}s)")

    wanted = list(ENGINES) if args.engine == "all" else args.engine.split(",")
    out_dir = ws / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[EngineResult] = []
    with tempfile.TemporaryDirectory() as td:
        wav = extract_wav(path, Path(td) / "audio.wav")
        for name in wanted:
            if name not in ENGINES:
                print(f"unknown engine: {name}", file=sys.stderr)
                continue
            logger.info("running engine %s ...", name)
            try:
                results.append(ENGINES[name](path, wav, cfg))
            except Exception as exc:
                results.append(EngineResult(name, ok=False, notes=f"{type(exc).__name__}: {exc}"))

    rows = score(results, duration)
    report = {"target": str(path), "duration_s": duration, "results": rows}
    out_file = out_dir / f"bench_{path.stem}_{int(duration)}.json"
    out_file.write_text(json.dumps(report, indent=2))
    print(json.dumps(rows, indent=2))
    print(f"report: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
