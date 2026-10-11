"""Where songs come from besides uploads: YouTube (search, links, audio via yt-dlp) and Spotify (search
with an API key, or pasted links). Spotify audio can't be downloaded, so like rave.dj a Spotify track is
matched to the same recording on YouTube."""
import base64
import glob
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request

MAX_DURATION = 15 * 60          # skip hour-long mixes and streams
YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
SP_ID = re.compile(r"^[A-Za-z0-9]{22}$")
# A plain client name: with a browser user agent Spotify serves the JS app instead of the page with the tags.
UA = "Rave/0.1"


class SourceError(Exception):
    pass


def youtube_enabled():
    import importlib.util
    return importlib.util.find_spec("yt_dlp") is not None


def spotify_enabled():
    return bool(os.environ.get("SPOTIFY_CLIENT_ID") and os.environ.get("SPOTIFY_CLIENT_SECRET"))


# YouTube --------------------------------------------------------------------------------------------

def _ydl(extra=None):
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "socket_timeout": 20}
    cookies = os.environ.get("YTDLP_COOKIES")
    if cookies and os.path.isfile(cookies):
        opts["cookiefile"] = cookies
    opts.update(extra or {})
    return yt_dlp.YoutubeDL(opts)


def _yt_item(e):
    vid = e.get("id") or ""
    return {"source": "youtube", "id": vid, "title": e.get("title") or vid,
            "artist": e.get("channel") or e.get("uploader") or "",
            "duration": e.get("duration"), "thumb": f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg",
            "url": f"https://www.youtube.com/watch?v={vid}"}


def search_youtube(q, n=12):
    with _ydl({"extract_flat": "in_playlist", "skip_download": True}) as y:
        info = y.extract_info(f"ytsearch{n}:{q}", download=False)
    out = []
    for e in info.get("entries") or []:
        if not e or not YT_ID.match(e.get("id") or ""):
            continue
        if e.get("duration") and e["duration"] > MAX_DURATION:
            continue
        if e.get("live_status") in ("is_live", "is_upcoming"):
            continue
        out.append(_yt_item(e))
    return out


def youtube_id(url):
    u = urllib.parse.urlparse(url.strip())
    host = (u.hostname or "").lower()
    if host == "youtu.be":
        vid = u.path.lstrip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        q = urllib.parse.parse_qs(u.query)
        vid = q.get("v", [""])[0]
        m = re.match(r"^/(?:shorts|embed|live)/([^/?#]+)", u.path)
        if not vid and m:
            vid = m.group(1)
    else:
        return None
    return vid if YT_ID.match(vid or "") else None


def youtube_info(vid):
    with _ydl({"skip_download": True}) as y:
        return _yt_item(y.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False))


def download_youtube(vid, cache_dir):
    """Download a video's audio into the cache (once) and return the file path."""
    if not YT_ID.match(vid or ""):
        raise SourceError("Bad YouTube id")
    d = os.path.join(cache_dir, "youtube")
    os.makedirs(d, exist_ok=True)
    have = [p for p in glob.glob(os.path.join(d, vid + ".*")) if not p.endswith((".part", ".ytdl"))]
    if have:
        return have[0]

    def too_long(info, *, incomplete=False):
        dur = info.get("duration")
        return "Longer than 15 minutes" if dur and dur > MAX_DURATION else None

    try:
        with _ydl({"format": "bestaudio/best", "outtmpl": os.path.join(d, "%(id)s.%(ext)s"),
                   "match_filter": too_long, "retries": 3}) as y:
            y.download([f"https://www.youtube.com/watch?v={vid}"])
    except Exception as e:
        if "confirm you" in str(e) and "not a bot" in str(e):
            raise SourceError("YouTube wants this server to sign in (bot check). Export your YouTube cookies "
                              "to cookies.txt in the data volume; see the README.")
        raise SourceError(f"Couldn't download from YouTube: {_short(e)}")
    have = [p for p in glob.glob(os.path.join(d, vid + ".*")) if not p.endswith((".part", ".ytdl"))]
    if not have:
        raise SourceError("YouTube download produced no audio (too long, or unavailable here)")
    return have[0]


def _short(e):
    msg = re.sub(r"\x1b\[[0-9;]*m", "", str(e)).replace("ERROR: ", "")
    return msg.splitlines()[0][:200] if msg else e.__class__.__name__


# Spotify --------------------------------------------------------------------------------------------

_token = {"value": None, "exp": 0}


def _spotify_token():
    if _token["value"] and time.time() < _token["exp"] - 60:
        return _token["value"]
    cred = f"{os.environ['SPOTIFY_CLIENT_ID']}:{os.environ['SPOTIFY_CLIENT_SECRET']}".encode()
    req = urllib.request.Request(
        "https://accounts.spotify.com/api/token", data=b"grant_type=client_credentials",
        headers={"Authorization": "Basic " + base64.b64encode(cred).decode(),
                 "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.load(r)
    _token.update(value=data["access_token"], exp=time.time() + data.get("expires_in", 3600))
    return _token["value"]


def _spotify_get(path, params=None):
    url = "https://api.spotify.com/v1/" + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + _spotify_token()})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def _sp_item(t):
    imgs = sorted((t.get("album") or {}).get("images") or [], key=lambda i: i.get("width") or 0)
    thumb = next((i["url"] for i in imgs if (i.get("width") or 0) >= 200), imgs[-1]["url"] if imgs else "")
    return {"source": "spotify", "id": t["id"], "title": t["name"],
            "artist": ", ".join(a["name"] for a in t.get("artists") or []),
            "duration": round((t.get("duration_ms") or 0) / 1000) or None, "thumb": thumb,
            "url": f"https://open.spotify.com/track/{t['id']}"}


def search_spotify(q, n=12):
    if not spotify_enabled():
        raise SourceError("Spotify search needs SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET "
                          "(pasting a Spotify track link works without them)")
    data = _spotify_get("search", {"q": q, "type": "track", "limit": n})
    return [_sp_item(t) for t in data.get("tracks", {}).get("items", []) if t]


def spotify_id(url):
    u = urllib.parse.urlparse(url.strip())
    if (u.hostname or "").lower() != "open.spotify.com":
        return None
    m = re.search(r"/track/([A-Za-z0-9]{22})", u.path)
    return m.group(1) if m else None


def spotify_track(tid):
    if not SP_ID.match(tid or ""):
        raise SourceError("Bad Spotify id")
    if spotify_enabled():
        return _sp_item(_spotify_get(f"tracks/{tid}"))
    # No API key: the public track page carries the title, artist and cover art.
    req = urllib.request.Request(f"https://open.spotify.com/track/{tid}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=15) as r:
        page = r.read(600_000).decode("utf-8", "replace")
    meta = dict((k, html.unescape(v)) for k, v in
                re.findall(r'<meta (?:property|name)="((?:og|music):[a-z:_]+)" content="([^"]*)"', page))
    title = meta.get("og:title")
    if not title:
        raise SourceError("Couldn't read that Spotify track")
    artist = (meta.get("og:description") or "").split(" · ")[0]
    dur = meta.get("music:duration")
    return {"source": "spotify", "id": tid, "title": title, "artist": artist,
            "duration": int(dur) if dur and dur.isdigit() else None, "thumb": meta.get("og:image", ""),
            "url": f"https://open.spotify.com/track/{tid}"}


BAD_WORDS = ("live", "remix", "cover", "karaoke", "instrumental", "sped up", "slowed", "nightcore", "8d",
             "reverb", "acoustic", "lyrics video", "reaction", "1 hour", "loop", "mashup", "tutorial")


def match_on_youtube(item):
    """Find the YouTube upload of a Spotify track: right length, right title, the artist's own channel
    or its auto-generated "- Topic" audio, and no live/remix/cover versions unless that's what it is."""
    title, artist = item.get("title", ""), item.get("artist", "")
    main_artist = artist.split(",")[0].strip()
    results = search_youtube(f"{main_artist} - {title} audio", 10)
    if not results:
        raise SourceError(f"No YouTube match for {artist} - {title}")
    words = lambda s: set(re.findall(r"\w+", s.lower()))
    want = words(title)

    def score(r):
        s = 0.0
        rt = r["title"].lower()
        s += 3.0 * len(want & words(rt)) / max(1, len(want))
        if main_artist.lower() in (rt + " " + r["artist"].lower()):
            s += 1.5
        if r["artist"].lower().endswith("- topic") or main_artist.lower() in r["artist"].lower():
            s += 1.5
        for w in BAD_WORDS:
            if w in rt and w not in title.lower():
                s -= 2.0
        if item.get("duration") and r.get("duration"):
            s -= min(3.0, abs(item["duration"] - r["duration"]) / 8.0)
        return s

    return max(results, key=score)


# Resolve and fetch ----------------------------------------------------------------------------------

def resolve_link(text):
    """A pasted YouTube or Spotify link -> a search-result style item, or None if it isn't one."""
    vid = youtube_id(text)
    if vid:
        return youtube_info(vid)
    tid = spotify_id(text)
    if tid:
        return spotify_track(tid)
    return None


def fetch(item, cache_dir):
    """Audio file for a YouTube or Spotify item (downloads into the cache once)."""
    if item.get("source") == "youtube":
        return download_youtube(item.get("id"), cache_dir)
    if item.get("source") == "spotify":
        yt = item.get("youtube_id")
        if not yt:
            m = match_on_youtube(item)
            yt = item["youtube_id"] = m["id"]
        return download_youtube(yt, cache_dir)
    raise SourceError("Unknown source")
