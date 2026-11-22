"""Bulk acquisition: expand playlists/channels, download sources, index them."""

from __future__ import annotations

import logging
import re
import sqlite3
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp

from ytpgen.config import DEFAULTS
from ytpgen.models import SourceVideo

logger = logging.getLogger(__name__)

LOCAL_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".m4v", ".webm", ".avi", ".mpg", ".mpeg"}
_MAX_PLAYLIST = 100  # hard cap on videos expanded per playlist/channel

# Single-video YouTube URL shapes: watch?v=, youtu.be/, /shorts/, /embed/, /live/
_VIDEO_ID_PATTERNS = (
    re.compile(r"[?&]v=([\w-]{6,})"),
    re.compile(r"youtu\.be/([\w-]{6,})"),
    re.compile(r"/(?:shorts|embed|live|v)/([\w-]{6,})"),
)

_COLLECTION_PATH_RE = re.compile(r"^/(?:playlist|channel|c|user|@)")

_CREATE_SOURCES_SQL = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT UNIQUE,
    url TEXT,
    title TEXT,
    local_path TEXT,
    duration_s REAL,
    added_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


def _is_youtube(url: str) -> bool:
    """True if the URL points at youtube.com or youtu.be."""
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return False
    return host in ("youtube.com", "youtu.be") or host.endswith(
        (".youtube.com", ".youtu.be")
    )


def _peek_video_id(url: str) -> str | None:
    """Extract a YouTube video id from a URL without any network access."""
    if not _is_youtube(url):
        return None
    for pattern in _VIDEO_ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None


def _is_youtube_collection(url: str) -> bool:
    """True for YouTube playlist/channel URLs (things that expand to many videos)."""
    if not _is_youtube(url) or _peek_video_id(url):
        return False
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if "list=" in (parsed.query or ""):
        return True
    path = (parsed.path or "").rstrip("/")
    return bool(_COLLECTION_PATH_RE.match(path)) or path.endswith("/videos")


def _ffprobe_duration(path: str | Path) -> float | None:
    """Duration in seconds via ffprobe, or None if it cannot be determined."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("ffprobe failed for %s: %s", path, exc)
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        logger.warning("ffprobe returned no duration for %s", path)
        return None


class Ingestor:
    """Bulk acquisition: resolve targets, download media, index into the db."""

    def __init__(self, cfg: dict, db: sqlite3.Connection) -> None:
        self.cfg = cfg or {}
        self.db = db
        workspace = Path(self.cfg.get("workspace") or DEFAULTS["workspace"]).expanduser()
        self.media_dir = workspace / "media"
        # Defensive: index.py normally creates this, but stay self-sufficient.
        self.db.execute(_CREATE_SOURCES_SQL)
        self.db.commit()

    # ------------------------------------------------------------------ resolve

    def resolve_targets(self, targets: list[str]) -> list[str]:
        """Expand playlists/channels into video URLs; pass single URLs through.

        Never downloads anything. Dedupes across all targets (order-preserving)
        and caps expansion at 100 videos per playlist/channel.
        """
        out: list[str] = []
        seen: set[str] = set()

        def push(url: str) -> None:
            if url and url not in seen:
                seen.add(url)
                out.append(url)

        for target in targets:
            target = (target or "").strip()
            if not target:
                continue
            if not _is_youtube_collection(target):
                push(target)
                continue
            try:
                for url in self._expand_collection(target):
                    push(url)
            except Exception as exc:  # noqa: BLE001 — one bad target must not kill the batch
                logger.warning("Failed to resolve %s: %s", target, exc)
        return out

    def _expand_collection(self, url: str) -> list[str]:
        """Flat-extract a playlist/channel into individual video URLs (capped)."""
        opts = {
            "extract_flat": "in_playlist",
            "skip_download": True,
            "playlistend": _MAX_PLAYLIST,
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        entries = list((info or {}).get("entries") or [])[:_MAX_PLAYLIST]
        out: list[str] = []
        for entry in entries:
            entry = entry or {}
            video_url = entry.get("url") or entry.get("webpage_url")
            if not video_url and entry.get("id"):
                video_url = f"https://www.youtube.com/watch?v={entry['id']}"
            if video_url:
                out.append(video_url)
        logger.info("Expanded %s -> %d video URLs (cap %d)", url, len(out), _MAX_PLAYLIST)
        return out

    # ----------------------------------------------------------------- download

    def download(self, url: str) -> SourceVideo:
        """Download one video into <workspace>/media/<id>.mp4; skip if on disk."""
        self.media_dir.mkdir(parents=True, exist_ok=True)
        opts = self._download_opts()
        with yt_dlp.YoutubeDL({**opts, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        if not info:
            raise RuntimeError(f"yt-dlp returned no info for {url}")
        vid = str(info.get("id") or "").strip()
        if not vid:
            raise RuntimeError(f"yt-dlp info has no id for {url}")

        local = self._locate_media(vid)
        if local is None:
            logger.info("Downloading %s (%s)", vid, url)
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True) or info
            # ytsearch/playlist results nest the real video under entries;
            # the real id (hence filename) comes from that entry.
            entries = (info or {}).get("entries")
            if entries:
                entry = next(iter(entries), None)
                if entry:
                    info = entry
            vid = str((info or {}).get("id") or vid).strip()
            local = self._locate_media(vid)
            if local is None:
                raise RuntimeError(f"Download completed but media file missing for {vid}")
        else:
            logger.info("Media already on disk for %s: %s", vid, local)

        duration = float(info.get("duration") or 0.0)
        if duration <= 0.0:
            duration = _ffprobe_duration(local) or 0.0
        return SourceVideo(
            video_id=vid,
            url=url,
            title=str(info.get("title") or ""),
            local_path=str(local),
            duration_s=duration,
        )

    def _download_opts(self) -> dict:
        fmt = (self.cfg.get("download") or {}).get("format") or DEFAULTS["download"]["format"]
        return {
            "format": fmt,
            "outtmpl": str(self.media_dir / "%(id)s.%(ext)s"),
            "merge_output_format": "mp4",
            "noplaylist": True,
            "retries": 3,
            "quiet": True,
            "no_warnings": True,
        }

    def _locate_media(self, vid: str) -> Path | None:
        """Find the downloaded file for vid (merged mp4 preferred)."""
        direct = self.media_dir / f"{vid}.mp4"
        if direct.is_file():
            return direct
        candidates = [
            p
            for p in self.media_dir.glob(f"{vid}.*")
            if p.is_file() and p.suffix not in (".part", ".temp", ".ytdl")
        ]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_size)
        return None

    # ------------------------------------------------------------------- ingest

    def ingest(self, target: str | list[str], force: bool = False) -> dict:
        """Resolve target(s), download new videos, index rows. Never partial-crashes.

        *target* may be a single URL/playlist or a list of them.
        Returns {'added': n, 'skipped': n, 'downloads_failed': n, 'video_ids': [...]}.
        Any per-URL failure is logged and counted in downloads_failed.
        """
        targets = [target] if isinstance(target, str) else list(target)
        urls: list[str] = []
        local: list[SourceVideo] = []
        for t in targets:
            if Path(t).expanduser().exists():  # local file or folder (e.g. workspace inbox/)
                local.extend(self.local_sources(Path(t).expanduser()))
            else:
                urls.extend(self.resolve_targets([t]))
        # dedupe, preserve order
        seen: set[str] = set()
        urls = [u for u in urls if not (u in seen or seen.add(u))]
        added = skipped = failed = 0
        video_ids: list[str] = []
        for video in local:
            if not force and self._is_indexed(video.video_id):
                skipped += 1
                continue
            self._insert_source(video)
            added += 1
            video_ids.append(video.video_id)
        for url in urls:
            try:
                vid = _peek_video_id(url)
                if vid and not force and self._is_indexed(vid):
                    skipped += 1
                    continue
                video = self.download(url)
                vid = video.video_id
                if not force and self._is_indexed(vid):
                    skipped += 1
                    continue
                self._insert_source(video)
                added += 1
                video_ids.append(vid)
            except Exception as exc:  # noqa: BLE001 — continue the batch on failure
                failed += 1
                logger.error("Ingest failed for %s: %s", url, exc)
        logger.info(
            "Ingest %r: added=%d skipped=%d failed=%d", target, added, skipped, failed
        )
        return {
            "added": added,
            "skipped": skipped,
            "downloads_failed": failed,
            "video_ids": video_ids,
        }

    def local_sources(self, path: Path) -> list[SourceVideo]:
        """Index video files in place (a file, or every video under a folder); ids are local-<name>."""
        files = [path] if path.is_file() else sorted(
            f for f in path.rglob("*") if f.suffix.lower() in LOCAL_VIDEO_EXTS)
        out = []
        for f in files:
            if f.suffix.lower() not in LOCAL_VIDEO_EXTS:
                logger.warning("Skipping %s: not a video file", f)
                continue
            out.append(SourceVideo(
                video_id="local-" + re.sub(r"[^A-Za-z0-9._-]+", "_", f.stem)[:80],
                url=f.resolve().as_uri(), title=f.stem, local_path=str(f.resolve()),
                duration_s=_ffprobe_duration(f) or 0.0))
        return out

    def _is_indexed(self, vid: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM sources WHERE video_id = ?", (vid,)
        ).fetchone()
        return row is not None

    def _insert_source(self, video: SourceVideo) -> int:
        """INSERT OR IGNORE then SELECT id (same pattern as the rest of the repo)."""
        self.db.execute(
            "INSERT OR IGNORE INTO sources (video_id, url, title, local_path, duration_s)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                video.video_id,
                video.url,
                video.title,
                video.local_path,
                video.duration_s,
            ),
        )
        self.db.commit()
        row = self.db.execute(
            "SELECT id FROM sources WHERE video_id = ?", (video.video_id,)
        ).fetchone()
        source_id = int(row[0]) if row else -1
        logger.debug("Indexed %s as sources.id=%s", video.video_id, source_id)
        return source_id
