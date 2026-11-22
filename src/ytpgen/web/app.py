"""Local web UI: browse the library, generate poops, edit EDLs.

create_app(cfg) -> Flask, served by ``ytp serve`` (127.0.0.1 only).
Heavy work (ingest / analyze / generate / auto / render) runs on
background threads via ytpgen.web.jobs.JobRunner — POSTs return
immediately and the UI polls /jobs for progress. The db connection
is opened per-request via flask.g.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from flask import (
    Flask,
    abort,
    current_app,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from ytpgen.config import workspace_dir
from ytpgen.index import Index
from ytpgen.web.jobs import JobRunner

logger = logging.getLogger(__name__)

WORDS_PER_PAGE = 100
OCCURRENCE_LIMIT = 20
_EDL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
MEDIA_EXTS = {
    ".mp4", ".m4v", ".mov", ".webm", ".mkv", ".mp3",
    ".m4a", ".wav", ".jpg", ".jpeg", ".png", ".gif", ".webp",
}


def create_app(cfg: dict) -> Flask:
    app = Flask(__name__)
    app.secret_key = "ytpgen-local-ui"  # localhost-only tool UI
    app.config["WORKSPACE"] = workspace_dir(cfg)
    app.config["COMPOSE_CFG"] = cfg.get("compose", {})
    app.config["CFG"] = cfg
    app.config["JOBS"] = JobRunner(workers=2)
    app.jinja_env.globals["workspace_name"] = Path(cfg["workspace"]).name

    @app.teardown_appcontext
    def close_db(_exc) -> None:  # pragma: no cover - flask plumbing
        idx = g.pop("ytp_db", None)
        if idx is not None:
            idx.conn.close()

    _register_routes(app)
    return app


def get_db() -> Index:
    """Per-request Index (closed at request teardown)."""
    if "ytp_db" not in g:
        g.ytp_db = Index(app_workspace() / "index.db")
    return g.ytp_db


def app_workspace() -> Path:
    return Path(current_app.config["WORKSPACE"])


def get_jobs() -> JobRunner:
    return current_app.config["JOBS"]


# --- safe path helpers (never expose anything outside the workspace) ---


def _safe_rel_path(rel: str) -> Path:
    """Resolve *rel* under the workspace; 404 on traversal or escape."""
    ws = app_workspace().resolve()
    try:
        target = (ws / rel).resolve()
    except (OSError, ValueError):
        abort(404)
    if target != ws and ws not in target.parents:
        abort(404)
    return target


def _edl_path(name: str) -> Path:
    """outputs/<name> for a validated EDL file name; 404 otherwise."""
    if not _EDL_NAME_RE.match(name) or ".." in name:
        abort(404)
    return app_workspace() / "outputs" / name


def _out_dir() -> Path:
    d = app_workspace() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(text: str) -> str:
    slug = _SLUG_RE.sub("_", text.strip().lower())[:40].strip("_")
    return slug or "untitled"


def _selected_source_ids() -> set[int] | None:
    """Parse the source_ids multi-select from the current request."""
    raw = request.form.getlist("source_ids")
    if not raw:
        return None
    try:
        return {int(v) for v in raw}
    except ValueError:
        return None


def _media_urls(db: Index, hits: list[dict]) -> None:
    """Attach source_title + media_url (workspace-safe) to each hit."""
    ws = app_workspace().resolve()
    titles = {
        r["id"]: (r["title"] or r["video_id"])
        for r in db.conn.execute("SELECT id, title, video_id FROM sources")
    }
    for h in hits:
        h["source_title"] = titles.get(h["source_id"], "?")
        rel = Path(h["local_path"]).resolve()
        if rel.is_file() and ws in rel.parents:
            span = f"#t={h['start_s']:.2f},{h['end_s']:.2f}"
            h["media_url"] = "/media/" + rel.relative_to(ws).as_posix() + span
        else:
            h["media_url"] = None


# --- job bodies (run on worker threads) ---


def _job_ingest(cfg: dict, urls: list[str]) -> dict:
    from ytpgen.acquire.ingest import Ingestor

    added = skipped = failed = 0
    db = Index(Path(cfg["workspace"]) / "index.db")
    try:
        ing = Ingestor(cfg, db.conn)
        for url in urls:
            res = ing.ingest(url, force=False)
            added += int(res.get("added", 0))
            skipped += int(res.get("skipped", 0))
            failed += int(res.get("downloads_failed", 0))
    finally:
        db.close()
    return {"added": added, "skipped": skipped, "failed": failed}


def _job_analyze(cfg: dict, source_ids: set[int] | None) -> dict:
    from ytpgen.analyze.segment import Segmenter
    from ytpgen.models import SourceVideo

    db = Index(Path(cfg["workspace"]) / "index.db")
    try:
        seg = Segmenter(cfg, db.conn)
        analyzed = db.analyze_source_ids()
        done = errors = 0
        for row in db.all_sources():
            if row["id"] in analyzed:
                continue
            if source_ids is not None and row["id"] not in source_ids:
                continue
            sv = SourceVideo(video_id=row["video_id"], title=row["title"],
                             local_path=row["local_path"])
            res = seg.analyze(sv, db.conn)
            if res.get("error"):
                errors += 1
            else:
                done += 1
        return {"analyzed": done, "errors": errors}
    finally:
        db.close()


def _job_generate(cfg: dict, text: str, mode: str,
                  source_ids: set[int] | None) -> dict:
    from ytpgen.compose.edl import save_edl
    from ytpgen.compose.matcher import Matcher
    from ytpgen.compose.renderer import Renderer

    db = Index(Path(cfg["workspace"]) / "index.db")
    try:
        clips = Matcher(cfg).resolve_sentence(text, db, mode=mode,
                                              source_ids=source_ids)
    finally:
        db.close()
    if not clips:
        raise RuntimeError("no clips matched — ingest + analyze sources first")
    c = cfg.get("compose", {})
    edl = Matcher.edl_from_clips(
        clips, fps=c.get("fps", 30), width=c.get("width", 1280),
        height=c.get("height", 720),
    )
    base = "gen_" + _slug(text)
    out_dir = Path(cfg["workspace"]) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_edl(edl, out_dir / f"{base}.edl.json")
    Renderer(cfg).render(edl, out_dir / f"{base}.mp4")
    return {"edl": f"{base}.edl.json", "mp4": f"{base}.mp4",
            "events": len(edl.events)}


def _job_auto(cfg: dict, video_id: str, target: int) -> dict:
    from ytpgen.compose.auto import AutoPooper
    from ytpgen.compose.edl import save_edl
    from ytpgen.compose.renderer import Renderer

    db = Index(Path(cfg["workspace"]) / "index.db")
    try:
        edl = AutoPooper(cfg, db.conn).plan(video_id, target_s=target)
    finally:
        db.close()
    if not edl.events:
        raise RuntimeError("plan produced no events — analyze the video first")
    base = "auto_" + _slug(video_id)
    out_dir = Path(cfg["workspace"]) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_edl(edl, out_dir / f"{base}.edl.json")
    Renderer(cfg).render(edl, out_dir / f"{base}.mp4")
    return {"edl": f"{base}.edl.json", "mp4": f"{base}.mp4",
            "events": len(edl.events)}


def _job_render(cfg: dict, name: str) -> dict:
    from ytpgen.compose.edl import load_edl
    from ytpgen.compose.renderer import Renderer

    path = Path(cfg["workspace"]) / "outputs" / name
    edl = load_edl(path)
    base = name[: -len(".edl.json")]
    out_dir = Path(cfg["workspace"]) / "outputs"
    Renderer(cfg).render(edl, out_dir / f"{base}.mp4")
    return {"edl": name, "mp4": f"{base}.mp4"}


def _job_reroll(cfg: dict, name: str) -> dict:
    """Re-pick sources for an existing generate EDL and re-render."""
    from ytpgen.compose.edl import load_edl, save_edl
    from ytpgen.compose.matcher import Matcher
    from ytpgen.compose.renderer import Renderer

    path = Path(cfg["workspace"]) / "outputs" / name
    old = load_edl(path)
    text = " ".join(ev.text for ev in old.events if ev.text.strip())
    mode = "hybrid"
    for ev in old.events:
        if "mode:" in ev.text:
            mode = ev.text.split("mode:")[1].strip().split("}")[0]
    source_ids = None
    if old.notes.startswith("sources:"):
        try:
            source_ids = {
                int(v) for v in old.notes[len("sources:"):].split(",") if v.strip()
            }
        except ValueError:
            source_ids = None
    db = Index(Path(cfg["workspace"]) / "index.db")
    try:
        clips = Matcher(cfg).resolve_sentence(text, db, mode=mode,
                                              source_ids=source_ids)
    finally:
        db.close()
    if not clips:
        raise RuntimeError("no clips matched for reroll")
    edl = Matcher.edl_from_clips(
        clips, fps=old.fps, width=old.width, height=old.height, seed=old.seed
    )
    save_edl(edl, path)
    mp4_name = name.replace(".edl.json", ".mp4")
    Renderer(cfg).render(edl, path.parent / mp4_name)
    return {"edl": name, "mp4": mp4_name, "events": len(edl.events)}


PLAY_MAX_FILMS = 4
PLAY_MOODS = {  # button -> planner tone
    "silly": "funny",
    "deep": "deep",
    "deadpan": "deadpan",
    "chaos": "chaotic and surreal: rapid non sequiturs, escalating absurdity, a voice that keeps "
             "interrupting itself, ending on the strangest possible line",
}


def _display_title(t: str) -> str:
    """Archive.org files titles library-style ("Date With Your Family, A"); flip the article back."""
    m = re.match(r"^(.*), (A|An|The)$", t.strip())
    return f"{m.group(2)} {m.group(1)}" if m else t


def _thumb(video_id: str, local_path: str, duration_s: float | None) -> str | None:
    """Cached poster frame (workspace/thumbs/<id>.jpg) from ~1/4 into the film."""
    import subprocess

    ws = app_workspace()
    out = ws / "thumbs" / f"{_SLUG_RE.sub('_', video_id)}.jpg"
    if not out.exists():
        out.parent.mkdir(exist_ok=True)
        at = max(1.0, (duration_s or 40.0) / 4)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{at:.1f}",
                        "-i", local_path, "-frames:v", "1", "-vf", "scale=320:-2", str(out)],
                       capture_output=True)
    return "/media/" + out.relative_to(ws).as_posix() if out.exists() else None


def _job_play(cfg: dict, ids: set[int], mood: str, about: str | None) -> dict:
    from ytpgen.compose.piece import make_piece

    ws = workspace_dir(cfg)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ws / "play" / f"{stamp}.mp4"
    out.parent.mkdir(exist_ok=True)
    db = Index(ws / "index.db")
    try:
        results = make_piece(cfg, db, out, PLAY_MOODS[mood], about, source_ids=ids,
                             drafts=4, render=1, seed=int(datetime.now().timestamp()))
    finally:
        db.close()
    best = results[0]
    order = sorted(ids)
    return {"video": "/media/" + Path(best["file"]).relative_to(ws).as_posix(),
            "title": best["title"],
            "chips": [dict(c, color=order.index(c["source_id"]) if c["source_id"] in order else -1)
                      for c in best["chips"]]}


# --- routes ---


def _register_routes(app: Flask) -> None:
    @app.route("/")
    def dashboard():
        db = get_db()
        sc = dict(db.conn.execute(
            "SELECT source_id, COUNT(*) FROM scenes GROUP BY source_id"
        ).fetchall())
        wd = dict(db.conn.execute(
            "SELECT source_id, COUNT(*) FROM words GROUP BY source_id"
        ).fetchall())
        prefs = db.word_prefs()
        pref_sources = {p["source_id"] for p in prefs.values()}
        sources = [{
            "id": r["id"], "video_id": r["video_id"], "title": r["title"],
            "duration_s": r["duration_s"], "scenes": sc.get(r["id"], 0),
            "words": wd.get(r["id"], 0), "preferred": r["id"] in pref_sources,
        } for r in db.all_sources()]
        return render_template(
            "dashboard.html", stats=db.stats(), sources=sources,
            jobs=get_jobs().list(10), prefs=prefs)

    @app.route("/ingest", methods=["POST"])
    def ingest():
        urls = [u.strip() for u in request.form.get("urls", "").splitlines()
                if u.strip()]
        if not urls:
            flash("No URLs submitted.", "error")
            return redirect(url_for("dashboard"))
        jid = get_jobs().submit(
            "ingest", f"ingest {len(urls)} url(s)",
            lambda: _job_ingest(app.config["CFG"], urls))
        flash(f"Ingest started as job {jid}.", "ok")
        return redirect(url_for("job_status", job_id=jid))

    @app.route("/analyze", methods=["POST"])
    def analyze():
        source_ids = _selected_source_ids()
        n = len(source_ids) if source_ids is not None else None
        jid = get_jobs().submit(
            "analyze", f"analyze {'all' if n is None else n} source(s)",
            lambda: _job_analyze(app.config["CFG"], source_ids))
        flash(f"Analysis started as job {jid}.", "ok")
        return redirect(url_for("job_status", job_id=jid))

    @app.route("/jobs")
    def jobs():
        return render_template("jobs.html", jobs=get_jobs().list(50))

    @app.route("/jobs/<job_id>")
    def job_status(job_id: str):
        job = get_jobs().get(job_id)
        if job is None:
            abort(404)
        return render_template("job_status.html", job=job)

    @app.route("/jobs/<job_id>/status")
    def job_status_json(job_id: str):
        data = get_jobs().status(job_id)
        if data is None:
            return jsonify({"error": "unknown job"}), 404
        return jsonify(data)

    @app.route("/words")
    def words():
        all_w = get_db().all_words()
        total = len(all_w)
        pages = max(1, -(-total // WORDS_PER_PAGE))
        page = min(pages, max(1, request.args.get("page", 1, type=int) or 1))
        chunk = all_w[(page - 1) * WORDS_PER_PAGE: page * WORDS_PER_PAGE]
        prefs = get_db().word_prefs()
        return render_template(
            "words.html", words=chunk, page=page, pages=pages, total=total,
            prefs=prefs)

    @app.route("/words/<word>")
    def word_detail(word: str):
        db = get_db()
        hits = db.find_word(word, limit=OCCURRENCE_LIMIT)
        _media_urls(db, hits)
        pref = db.word_prefs().get(word.lower())
        return render_template(
            "word_detail.html", word=word, hits=hits, pref=pref,
            sources=[dict(r) for r in db.all_sources()])

    @app.route("/words/<word>/pref", methods=["POST"])
    def word_pref(word: str):
        db = get_db()
        action = request.form.get("action", "set")
        if action == "clear":
            db.clear_word_pref(word)
            flash(f"Cleared preference for '{word}'.", "ok")
        else:
            sid = request.form.get("source_id", type=int)
            if sid is None:
                flash("Pick a source to prefer.", "error")
            else:
                db.set_word_pref(word, sid)
                flash(f"'{word}' will now prefer that source's clips.", "ok")
        return redirect(url_for("word_detail", word=word))

    @app.route("/edl")
    def edl_list():
        from ytpgen.compose.edl import load_edl

        files = []
        for p in sorted(_out_dir().glob("*.edl.json")):
            try:
                n_events = len(load_edl(p).events)
            except Exception:  # noqa: BLE001 — list corrupt files too
                n_events = -1
            files.append({
                "name": p.name, "path": str(p), "events": n_events,
                "created": datetime.fromtimestamp(
                    p.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")})
        return render_template("edl_list.html", files=files)

    @app.route("/edl/<name>")
    def edl_detail(name: str):
        path = _edl_path(name)
        if not path.is_file():
            abort(404)
        mp4_name = name[: -len(".edl.json")] + ".mp4"
        return render_template(
            "edl_detail.html", name=name,
            content=path.read_text(encoding="utf-8"),
            rendered=mp4_name,
            rendered_exists=(_out_dir() / mp4_name).is_file(),
        )

    @app.route("/edl/<name>", methods=["POST"])
    def edl_save(name: str):
        path = _edl_path(name)
        from ytpgen.models import EDL

        try:  # validate before overwriting; keep old file on bad input
            EDL.model_validate_json(request.form.get("content", ""))
        except Exception as exc:  # noqa: BLE001
            flash(f"Invalid EDL JSON: {exc}", "error")
        else:
            path.write_text(request.form.get("content", ""), encoding="utf-8")
            flash(f"Saved {name}.", "ok")
        return redirect(url_for("edl_detail", name=name))

    @app.route("/edl/<name>/render", methods=["POST"])
    def edl_render(name: str):
        _edl_path(name)  # 404 on bad name
        jid = get_jobs().submit(
            "render", f"render {name}",
            lambda: _job_render(app.config["CFG"], name))
        return redirect(url_for("job_status", job_id=jid))

    @app.route("/edl/<name>/reroll", methods=["POST"])
    def edl_reroll(name: str):
        _edl_path(name)  # 404 on bad name
        if not name.startswith("gen_"):
            flash("Reroll applies to generated EDLs (gen_*).", "error")
            return redirect(url_for("edl_detail", name=name))
        jid = get_jobs().submit(
            "reroll", f"reroll {name}",
            lambda: _job_reroll(app.config["CFG"], name))
        return redirect(url_for("job_status", job_id=jid))

    @app.route("/generate", methods=["GET", "POST"])
    def generate():
        if request.method == "GET":
            return render_template(
                "generate.html", sources=[dict(r) for r in get_db().all_sources()])
        text = request.form.get("text", "").strip()
        mode = request.form.get("mode", "hybrid")
        mode = mode if mode in ("word", "phone", "hybrid") else "hybrid"
        if not text:
            flash("No text submitted.", "error")
            return redirect(url_for("generate"))
        source_ids = _selected_source_ids()
        jid = get_jobs().submit(
            "generate", f"generate {text[:30]!r}",
            lambda: _job_generate(app.config["CFG"], text, mode, source_ids))
        return redirect(url_for("job_status", job_id=jid))

    @app.route("/auto", methods=["GET", "POST"])
    def auto():
        if request.method == "GET":
            sources = [{"video_id": r["video_id"], "title": r["title"]}
                       for r in get_db().all_sources()]
            return render_template("auto.html", sources=sources)
        video_id = request.form.get("video_id", "").strip()
        target = request.form.get("target", 60, type=int) or 60
        if not video_id:
            flash("No video id submitted.", "error")
            return redirect(url_for("auto"))
        jid = get_jobs().submit(
            "auto", f"auto {video_id} ({target}s)",
            lambda: _job_auto(app.config["CFG"], video_id, target))
        return redirect(url_for("job_status", job_id=jid))

    # --- /play: kid-simple "pick films, pick a mood, make it" page (v0.2) ---

    @app.route("/play")
    def play():
        db = get_db()
        done = db.analyze_source_ids()
        films = [{"id": r["id"], "title": _display_title(r["title"] or r["video_id"]),
                  "thumb": _thumb(r["video_id"], r["local_path"], r["duration_s"])}
                 for r in db.all_sources() if r["id"] in done]
        return render_template("play.html", films=films, moods=PLAY_MOODS)

    @app.route("/play/make", methods=["POST"])
    def play_make():
        data = request.get_json(silent=True) or {}
        known = get_db().analyze_source_ids()
        try:
            ids = {int(i) for i in data.get("films", [])} & known
        except (TypeError, ValueError):
            ids = set()
        mood = data.get("mood", "silly")
        if not ids or len(ids) > PLAY_MAX_FILMS or mood not in PLAY_MOODS:
            return jsonify({"error": f"pick 1-{PLAY_MAX_FILMS} films and a mood"}), 400
        about = str(data.get("about", ""))[:120].strip() or None
        jid = get_jobs().submit("play", f"play {mood} x{len(ids)}",
                                lambda: _job_play(app.config["CFG"], ids, mood, about))
        return jsonify({"job": jid})

    @app.route("/media/<path:rel>")
    def media(rel: str):
        target = _safe_rel_path(rel)  # resolved + guaranteed inside workspace
        if not target.is_file() or target.suffix.lower() not in MEDIA_EXTS:
            abort(404)
        ws = app_workspace().resolve()
        return send_from_directory(
            ws, target.relative_to(ws).as_posix(), conditional=True)
