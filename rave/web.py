"""Web UI and API: upload songs, queue a mashup, poll its progress, play and download the result.

    uvicorn rave.web:app --host 0.0.0.0 --port 8000
"""
import json
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from importlib import resources
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from . import __version__, engine, separate, sources, transform

DATA = os.environ.get("RAVE_DATA", os.path.join(os.path.expanduser("~"), "rave-data"))
JOBS = os.path.join(DATA, "jobs")
os.environ.setdefault("RAVE_CACHE", os.path.join(DATA, "cache"))
MAX_SONGS = 8
AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".webm", ".mp4"}
OUT_FORMATS = {"mp3", "wav", "flac", "m4a"}



@asynccontextmanager
async def lifespan(_app):
    _load_jobs()
    yield


app = FastAPI(title="Rave", version=__version__, lifespan=lifespan)
_pool = ThreadPoolExecutor(max_workers=int(os.environ.get("RAVE_WORKERS", "1")))
_jobs = {}
_lock = threading.Lock()


def _job_dir(jid):
    if not re.fullmatch(r"[0-9a-f]{12}", jid):
        raise HTTPException(404, "No such job")
    return os.path.join(JOBS, jid)


def _save_state(job):
    with open(os.path.join(_job_dir(job["id"]), "job.json"), "w", encoding="utf-8") as f:
        json.dump(job, f)


def _load_jobs():
    os.makedirs(JOBS, exist_ok=True)
    for jid in os.listdir(JOBS):
        try:
            with open(os.path.join(JOBS, jid, "job.json"), encoding="utf-8") as f:
                job = json.load(f)
        except (OSError, ValueError):
            continue
        if job.get("status") in ("queued", "running"):
            job.update(status="failed", error="The server restarted while this mashup was being made.")
        _jobs[jid] = job


def _safe_name(name):
    base = os.path.basename(name or "song")
    stem, ext = os.path.splitext(base)
    stem = re.sub(r"[^\w\-. ()&',]+", "_", stem).strip(" .") or "song"
    return stem[:100] + ext.lower()


@app.get("/", response_class=HTMLResponse)
def index():
    return resources.files("rave").joinpath("static/index.html").read_text(encoding="utf-8")


@app.get("/api/info")
def info():
    return {"version": __version__, "demucs": separate.demucs_available(),
            "stretch": transform.backend(), "max_songs": MAX_SONGS,
            "youtube": sources.youtube_enabled(), "spotify_search": sources.spotify_enabled()}


@app.get("/api/jobs")
def list_jobs():
    with _lock:
        jobs = sorted(_jobs.values(), key=lambda j: j["created"], reverse=True)
    return [_public(j) for j in jobs[:30]]


@app.get("/api/search")
def search(q: str, source: str = "youtube"):
    """Search YouTube or Spotify; a pasted YouTube/Spotify link resolves to that one track."""
    q = q.strip()[:300]
    if not q:
        return []
    try:
        if re.match(r"^https?://", q):
            item = sources.resolve_link(q)
            if not item:
                raise HTTPException(400, "Paste a YouTube or Spotify track link, or search by name.")
            return [item]
        if source == "spotify":
            return sources.search_spotify(q)
        if not sources.youtube_enabled():
            raise HTTPException(400, "YouTube support (yt-dlp) isn't installed.")
        return sources.search_youtube(q)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, sources._short(e))


def _track(t):
    """A song picked from search, cleaned down to the fields the job needs."""
    src = t.get("source")
    tid = str(t.get("id") or "")
    if src == "youtube" and not sources.YT_ID.match(tid) or src == "spotify" and not sources.SP_ID.match(tid):
        raise HTTPException(400, "Bad track")
    if src not in ("youtube", "spotify"):
        raise HTTPException(400, "Bad track source")
    dur = t.get("duration")
    yt = str(t.get("youtube_id") or "")
    extra = {"youtube_id": yt} if src == "spotify" and sources.YT_ID.match(yt) else {}
    return {**extra, "source": src, "id": tid, "title": str(t.get("title") or tid)[:200],
            "artist": str(t.get("artist") or "")[:200],
            "duration": int(dur) if isinstance(dur, (int, float)) else None,
            "thumb": str(t.get("thumb") or "")[:500] if str(t.get("thumb") or "").startswith("https://") else ""}


@app.post("/api/jobs")
async def create_job(files: List[UploadFile] = File(default=[]), options: str = Form("{}")):
    try:
        o = json.loads(options)
    except ValueError:
        raise HTTPException(400, "Bad options")
    fmt = str(o.get("format", "mp3")).lower()
    if fmt not in OUT_FORMATS:
        raise HTTPException(400, "Unknown output format")
    sep = o.get("separation", "auto")
    if sep not in separate.METHODS:
        raise HTTPException(400, "Unknown separation method")
    if sep == "demucs" and not separate.demucs_available():
        raise HTTPException(400, "Demucs isn't installed in this image.")
    for f in files:
        if os.path.splitext(f.filename or "")[1].lower() not in AUDIO_EXT:
            raise HTTPException(400, f"Not an audio file: {f.filename}")
    # The lineup, in order: {"upload": n} refers to the n-th uploaded file, anything else is a search pick.
    lineup = o.get("tracks") or [{"upload": k} for k in range(len(files))]
    if not 2 <= len(lineup) <= MAX_SONGS:
        raise HTTPException(400, f"Pick between 2 and {MAX_SONGS} songs.")

    jid = uuid.uuid4().hex[:12]
    d = os.path.join(JOBS, jid)
    tracks = []
    for k, t in enumerate(lineup):
        if "upload" in t:
            n = int(t["upload"])
            if not 0 <= n < len(files):
                raise HTTPException(400, "Missing upload")
            name = _safe_name(files[n].filename)
            tracks.append({"source": "upload", "title": os.path.splitext(name)[0], "artist": "",
                           "thumb": "", "file": os.path.join(d, "songs", str(k), name), "n": n})
        else:
            tracks.append(_track(t))
    for t in tracks:
        if t["source"] == "upload":
            os.makedirs(os.path.dirname(t["file"]))
            f = files[t.pop("n")]
            await f.seek(0)
            with open(t["file"], "wb") as out:
                shutil.copyfileobj(f.file, out, 1 << 20)
    os.makedirs(d, exist_ok=True)

    transpose = {}
    for k, v in (o.get("transpose") or {}).items():
        if v is not None and str(v) != "auto":
            transpose[int(k)] = max(-12, min(12, int(v)))
    phrase = int(o.get("phrase_bars", 8))
    opts = engine.Options(
        base=int(o.get("base", -1)),
        length=max(30.0, min(600.0, float(o.get("length", 150)))),
        phrase_bars=phrase if phrase in (4, 8, 16) else 8,
        bed="rotate" if o.get("bed") == "rotate" else "base",
        separation=sep,
        match_key=bool(o.get("match_key", True)),
        transpose=transpose,
        acappella_drop=bool(o.get("acappella_drop", True)),
        riser=bool(o.get("riser", True)))
    names = [t["title"] for t in tracks]
    title = (str(o.get("title") or "").strip() or " x ".join(names))[:120]
    job = {"id": jid, "title": title, "songs": names, "tracks": tracks, "status": "queued",
           "progress": 0.0, "message": "Waiting in line", "created": time.time(), "format": fmt,
           "report": None, "error": None}
    with _lock:
        _jobs[jid] = job
        _save_state(job)
    _pool.submit(_run, jid, opts)
    return _public(job)


def _run(jid, opts):
    job = _jobs[jid]
    out = os.path.join(_job_dir(jid), f"mashup.{job['format']}")
    tracks = job["tracks"]
    fetch_share = 0.12 if any(t["source"] != "upload" for t in tracks) else 0.0

    def progress(frac, msg):
        job.update(progress=round(fetch_share + (1 - fetch_share) * frac, 3), message=msg)

    job.update(status="running", message="Starting", started=time.time())
    try:
        paths = []
        for k, t in enumerate(tracks):
            if t["source"] == "upload":
                paths.append(t["file"])
                continue
            job.update(progress=round(fetch_share * k / len(tracks), 3),
                       message=f"Grabbing {t['title']}" + (f" by {t['artist']}" if t["artist"] else ""))
            paths.append(sources.fetch(t, os.environ["RAVE_CACHE"]))
        job["report"] = engine.make_mashup(paths, out, opts, progress, names=[t["title"] for t in tracks])
        job.update(status="done", progress=1.0, message="Done", finished=time.time())
    except Exception as e:
        traceback.print_exc()
        job.update(status="failed", error=str(e) or e.__class__.__name__, finished=time.time())
    finally:
        # Uploaded songs aren't needed again (their stems are cached), so don't keep them.
        shutil.rmtree(os.path.join(_job_dir(jid), "songs"), ignore_errors=True)
        for t in tracks:
            t.pop("file", None)
        with _lock:
            _save_state(job)


def _public(job):
    j = dict(job)
    j["tracks"] = [{k: v for k, v in t.items() if k in ("source", "title", "artist", "thumb", "duration")}
                   for t in job.get("tracks") or []]
    with _lock:
        queued = sorted((x for x in _jobs.values() if x["status"] == "queued"), key=lambda x: x["created"])
    if job["status"] == "queued":
        j["position"] = next((k + 1 for k, x in enumerate(queued) if x["id"] == job["id"]), None)
    return j


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    _job_dir(jid)
    job = _jobs.get(jid)
    if not job:
        raise HTTPException(404, "No such job")
    return _public(job)


@app.get("/api/jobs/{jid}/audio")
def get_audio(jid: str, download: bool = False):
    job = _jobs.get(jid)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Not ready")
    path = os.path.join(_job_dir(jid), f"mashup.{job['format']}")
    name = _safe_name(job["title"]) + "." + job["format"]
    return FileResponse(path, filename=name if download else None,
                        media_type={"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac",
                                    "m4a": "audio/mp4"}[job["format"]])


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    d = _job_dir(jid)
    job = _jobs.get(jid)
    if not job:
        raise HTTPException(404, "No such job")
    if job["status"] in ("queued", "running"):
        raise HTTPException(409, "Still being made")
    with _lock:
        _jobs.pop(jid, None)
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}
