"""ytp CLI — pull, analyze, match, and render YouTube Poops.

Usage examples:
    ytp ingest <url-or-playlist> [--force]
    ytp analyze [video_id ...] | --all
    ytp generate "text to speak" -o out.mp4 [--mode word|phone|hybrid]
    ytp auto <video_id> [--target 60]
    ytp render timeline.edl -o out.mp4
    ytp stats
    ytp serve [--port 7777]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ytpgen.config import load_config, workspace_dir

logger = logging.getLogger("ytp")


def _db(cfg: dict):
    from ytpgen.index import Index

    return Index(workspace_dir(cfg) / "index.db")


def _open_config(args: argparse.Namespace) -> dict:
    cfg_path = getattr(args, "config", None) or "config/ytp.yaml"
    cfg = load_config(cfg_path if Path(cfg_path).exists() else None)
    if getattr(args, "workspace", None):
        cfg["workspace"] = args.workspace
    return cfg


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    from ytpgen.acquire.ingest import Ingestor

    cfg = _open_config(args)
    db = _db(cfg)
    result = Ingestor(cfg, db.conn).ingest(args.targets, force=args.force)
    print(json.dumps(result, indent=2))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from ytpgen.analyze.segment import Segmenter

    cfg = _open_config(args)
    db = _db(cfg)
    seg = Segmenter(cfg, db.conn)

    if args.all:
        rows = db.all_sources()
        video_ids = [r["video_id"] for r in rows]
    else:
        video_ids = args.video_ids

    if not video_ids:
        print("Nothing to analyze: pass video ids or --all", file=sys.stderr)
        return 1

    for vid in video_ids:
        row = db.conn.execute(
            "SELECT * FROM sources WHERE video_id = ?", (vid,)
        ).fetchone()
        if row is None:
            print(f"Unknown video id: {vid}", file=sys.stderr)
            continue
        result = seg.analyze(
            {
                "video_id": row["video_id"],
                "title": row["title"],
                "local_path": row["local_path"],
            },
            db,
        )
        print(f"{vid}: {result}")
    print(json.dumps(db.stats(), indent=2))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from ytpgen.compose.matcher import Matcher
    from ytpgen.compose.edl import save_edl
    from ytpgen.compose.renderer import Renderer

    cfg = _open_config(args)
    db = _db(cfg)
    clips = Matcher(cfg).resolve_sentence(args.text, db, mode=args.mode)
    if not clips:
        print("No clips matched — run `ytp ingest` first.", file=sys.stderr)
        return 1

    edl = Matcher.edl_from_clips(
        clips, fps=cfg["compose"]["fps"],
        width=cfg["compose"]["width"], height=cfg["compose"]["height"],
    )
    edl_path = Path(args.output).with_suffix(".edl.json")
    save_edl(edl, edl_path)
    out = Renderer(cfg).render(edl, Path(args.output))
    print(f"edl:     {edl_path}")
    print(f"output:  {out}")
    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    from ytpgen.compose.auto import AutoPooper
    from ytpgen.compose.edl import save_edl
    from ytpgen.compose.renderer import Renderer

    cfg = _open_config(args)
    db = _db(cfg)
    edl = AutoPooper(cfg, db.conn).plan(args.video_id, target_s=args.target)
    if not edl.events:
        print("Plan produced no events — is the video analyzed?", file=sys.stderr)
        return 1
    edl_path = Path(args.output).with_suffix(".edl.json")
    save_edl(edl, edl_path)
    out = Renderer(cfg).render(edl, Path(args.output))
    print(f"events:  {len(edl.events)}")
    print(f"edl:     {edl_path}")
    print(f"output:  {out}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from ytpgen.compose.edl import load_edl
    from ytpgen.compose.renderer import Renderer

    cfg = _open_config(args)
    edl = load_edl(Path(args.edl))
    out = Renderer(cfg).render(edl, Path(args.output))
    print(f"output:  {out}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    cfg = _open_config(args)
    print(json.dumps(_db(cfg).stats(), indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    cfg = _open_config(args)
    from ytpgen.web.app import create_app

    app = create_app(cfg)
    app.run(host="127.0.0.1", port=args.port, debug=False)
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ytp", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", help="Override workspace dir")
    p.add_argument("--config", help="Path to config yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("ingest", help="Expand + download playlists/channels/videos")
    sp.add_argument("targets", nargs="+")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("analyze", help="Scene-detect + transcribe + phone-index")
    sp.add_argument("video_ids", nargs="*")
    sp.add_argument("--all", action="store_true")
    sp.set_defaults(func=cmd_analyze)

    sp = sub.add_parser("generate", help="Type a sentence; the library says it")
    sp.add_argument("text")
    sp.add_argument("-o", "--output", default="outputs/ytp_output.mp4")
    sp.add_argument("--mode", choices=["word", "phone", "hybrid"], default="hybrid")
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("auto", help="Auto-generate a poop from one source")
    sp.add_argument("video_id")
    sp.add_argument("--target", type=int, default=60, help="Target duration seconds")
    sp.add_argument("-o", "--output", default="outputs/auto_poop.mp4")
    sp.set_defaults(func=cmd_auto)

    sp = sub.add_parser("render", help="Render an existing EDL")
    sp.add_argument("edl")
    sp.add_argument("-o", "--output", default="outputs/render.mp4")
    sp.set_defaults(func=cmd_render)

    sub.add_parser("stats", help="Library statistics").set_defaults(func=cmd_stats)

    sp = sub.add_parser("serve", help="Local web UI")
    sp.add_argument("--port", type=int, default=7777)
    sp.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
