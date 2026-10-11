"""Download every reference, original and source song; write dev/refs/manifest.json.

Source songs are found by YouTube search, preferring official audio and avoiding live/cover/nightcore/
sped-up versions (unless the song asks for a remix). Re-running skips anything already resolved.
"""
import json
import os
import re
import sys

sys.path.insert(0, "/dev-scripts/refs")
from rave import sources  # noqa: E402
from songs import REFS  # noqa: E402

CACHE = "/data/cache"
OUT = "/dev-scripts/refs/manifest.json"
BAD = re.compile(r"\b(live|cover|nightcore|sped|slowed|8d|karaoke|instrumental|acapella|reverb|"
                 r"tiktok|1 hour|loop|mashup|megamix|reaction|lyrics? video|piano|acoustic)\b", re.I)

man = json.load(open(OUT)) if os.path.isfile(OUT) else {}


def pick(query):
    want_remix = "remix" in query.lower()
    for q in (f"{query} official audio", query):
        for r in sources.search_youtube(q, 10):
            title = r["title"]
            if not r.get("duration") or not 100 <= r["duration"] <= 480:
                continue
            if BAD.search(title) or ("remix" in title.lower() and not want_remix):
                continue
            return r
    return None


for ref, info in REFS.items():
    entry = man.setdefault(ref, {"video": info["video"], "songs": {}})
    for vid_key in ("video", "original"):
        vid = info.get(vid_key)
        if vid and not entry.get(vid_key + "_path"):
            entry[vid_key + "_path"] = sources.download_youtube(vid, CACHE)
            print(ref, vid_key, entry[vid_key + "_path"], flush=True)
    if info.get("original_query") and not entry.get("original_path"):
        for r in sources.search_youtube(info["original_query"], 8):
            if "nightcore" not in r["title"].lower() and r.get("duration", 0) > 120:
                entry["original"] = r["id"]
                entry["original_title"] = r["title"]
                entry["original_path"] = sources.download_youtube(r["id"], CACHE)
                print(ref, "original", r["title"], flush=True)
                break
    for song in info["songs"]:
        if song in entry["songs"]:
            continue
        try:
            r = pick(song)
            if not r:
                print("  NOT FOUND", song, flush=True)
                continue
            path = sources.download_youtube(r["id"], CACHE)
            entry["songs"][song] = {"id": r["id"], "title": r["title"], "duration": r["duration"],
                                    "path": path}
            print(f"  {song}  <-  {r['title']} ({r['duration']}s)", flush=True)
        except Exception as e:
            print("  FAILED", song, e, flush=True)
        json.dump(man, open(OUT, "w"), indent=1)
    json.dump(man, open(OUT, "w"), indent=1)
print("done")
