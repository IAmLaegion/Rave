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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from . import __version__, engine, separate, sources, stemdeck, stempack, styles, transform

DATA = os.environ.get("RAVE_DATA", os.path.join(os.path.expanduser("~"), "rave-data"))
JOBS = os.path.join(DATA, "jobs")
UPLOADS = os.path.join(DATA, "uploads")
MAX_PACK = 4 * 1024 ** 3                 # bytes per stem pack
CHUNK = 48 * 1024 ** 2                   # largest chunk accepted (under Cloudflare's 100 MB limit)
os.environ.setdefault("RAVE_CACHE", os.path.join(DATA, "cache"))
MAX_SONGS = 8
AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".webm", ".mp4"}
OUT_FORMATS = {"mp3", "wav", "flac", "m4a"}



@asynccontextmanager
async def lifespan(_app):
    _load_jobs()
    yield


app = FastAPI(title="Rave", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def no_store(request, call_next):
    """Job status must never come from a cache (a proxy or tunnel in front would freeze the progress)."""
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response
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
            "roformer": separate.roformer_available(), "stemdeck": stemdeck.available(),
            "best": separate.best_available(),
            "stem_packs": stempack.ARCHIVE_EXT + stempack.STEM_EXT,
            "youtube": sources.youtube_enabled(), "spotify_search": sources.spotify_enabled(),
            "styles": [{"key": s.key, "name": s.name, "blurb": s.blurb}
                       for k, s in styles.STYLES.items() if k == s.key]}


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


def _length(v):
    """Target length in seconds; empty/0/"auto" = the style's own (DJ Earworm ~4 min, riff-off ~3)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v <= 0 else max(30.0, min(600.0, v))


def _nightcore(v):
    """Nightcore speed from the request: true = the measured norm (~1.22x), or a number 1.05-1.5."""
    if v is True:
        return 1.22
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return min(1.5, v) if v >= 1.05 else 0.0


def _upload_dir(uid):
    if not re.fullmatch(r"[0-9a-f]{16}", uid or ""):
        raise HTTPException(404, "No such upload")
    return os.path.join(UPLOADS, uid)


@app.post("/api/uploads")
def new_upload(name: str = Form(...), size: int = Form(...)):
    """Start a stem-pack upload. The file then arrives in chunks (PUT), so a large pack gets through a
    proxy that limits the size of one request."""
    if not stempack.is_archive(name):
        raise HTTPException(400, f"Not a supported archive: {name} (zip, 7z, rar, tar, .stem.mp4)")
    if not 0 < size <= MAX_PACK:
        raise HTTPException(400, "A stem pack can be up to 4 GB.")
    uid = uuid.uuid4().hex[:16]
    d = _upload_dir(uid)
    os.makedirs(d)
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"name": _safe_name(name), "size": size, "created": time.time()}, f)
    open(os.path.join(d, "pack"), "wb").close()
    return {"id": uid, "chunk": CHUNK}


@app.put("/api/uploads/{uid}")
async def put_chunk(uid: str, offset: int, request: Request):
    d = _upload_dir(uid)
    path = os.path.join(d, "pack")
    if not os.path.isfile(path):
        raise HTTPException(404, "No such upload")
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    have = os.path.getsize(path)
    if offset != have:
        return {"received": have}                      # tell the client where to resume
    data = await request.body()
    if len(data) > CHUNK or have + len(data) > meta["size"]:
        raise HTTPException(400, "Chunk too large")
    with open(path, "ab") as f:
        f.write(data)
    return {"received": have + len(data)}


def _clean_uploads(max_age=24 * 3600):
    """Stem packs left behind by uploads that never became a mashup."""
    if not os.path.isdir(UPLOADS):
        return
    for uid in os.listdir(UPLOADS):
        meta = os.path.join(UPLOADS, uid, "meta.json")
        try:
            if time.time() - json.load(open(meta, encoding="utf-8"))["created"] > max_age:
                shutil.rmtree(os.path.join(UPLOADS, uid), ignore_errors=True)
        except (OSError, ValueError, KeyError):
            pass


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
    _clean_uploads()
    for k, t in enumerate(lineup):
        if "stems" in t:
            ud = _upload_dir(str(t["stems"]))
            meta = json.load(open(os.path.join(ud, "meta.json"), encoding="utf-8"))
            if os.path.getsize(os.path.join(ud, "pack")) != meta["size"]:
                raise HTTPException(400, f"The upload of {meta['name']} didn't finish.")
            tracks.append({"source": "stems", "title": os.path.splitext(meta["name"])[0], "artist": "Stems",
                           "thumb": "", "upload": str(t["stems"])})
        elif "upload" in t:
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
    style = str(o.get("style") or "earworm")
    if style not in styles.STYLES:
        raise HTTPException(400, "Unknown engine")
    opts = engine.Options(
        style=style,
        base=int(o.get("base", -1)),
        length=_length(o.get("length")),
        phrase_bars=phrase if phrase in (4, 8, 16) else 8,
        bed="rotate" if o.get("bed") == "rotate" else "base",
        separation=sep,
        match_key=bool(o.get("match_key", True)),
        transpose=transpose,
        acappella_drop=bool(o.get("acappella_drop", True)),
        riser=bool(o.get("riser", True)),
        nightcore=_nightcore(o.get("nightcore")),
        spatial="8d" if o.get("spatial") == "8d" else "")
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
    fetch_share = 0.12 if any(t["source"] not in ("upload", "stems") for t in tracks) else 0.0

    def progress(frac, msg):
        if msg != job.get("message"):
            job["step_started"] = time.time()
        job.update(progress=round(fetch_share + (1 - fetch_share) * frac, 3), message=msg)

    job.update(status="running", message="Starting", started=time.time())
    try:
        paths, names = [], []
        for k, t in enumerate(tracks):
            if t["source"] == "upload":
                paths.append(t["file"])
                names.append(t["title"])
                continue
            if t["source"] == "stems":
                ud = _upload_dir(t["upload"])
                for name, path in stempack.prepare(os.path.join(ud, "pack"), ud, os.environ["RAVE_CACHE"],
                                                   t["title"], lambda m: progress(0.0, m)):
                    paths.append(path)
                    names.append(name)
                continue
            names.append(t["title"])
            progress(0.0, f"Grabbing {t['title']}" + (f" by {t['artist']}" if t["artist"] else ""))
            job["progress"] = round(fetch_share * k / len(tracks), 3)
            paths.append(sources.fetch(t, os.environ["RAVE_CACHE"]))
        if not 2 <= len(paths) <= MAX_SONGS:
            raise RuntimeError(f"The stem packs hold {len(paths)} songs; a mashup takes 2 to {MAX_SONGS}.")
        job["songs"] = names
        job["report"] = engine.make_mashup(paths, out, opts, progress, names=names)
        job.update(status="done", progress=1.0, message="Done", finished=time.time())
    except Exception as e:
        traceback.print_exc()
        job.update(status="failed", error=str(e) or e.__class__.__name__, finished=time.time())
    finally:
        # Uploaded songs aren't needed again (their stems are cached), so don't keep them.
        shutil.rmtree(os.path.join(_job_dir(jid), "songs"), ignore_errors=True)
        for t in tracks:
            if t.get("source") == "stems":
                shutil.rmtree(_upload_dir(t["upload"]), ignore_errors=True)
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
    if job["status"] == "running":
        j["elapsed"] = round(time.time() - job.get("step_started", job.get("started", time.time())), 1)
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
