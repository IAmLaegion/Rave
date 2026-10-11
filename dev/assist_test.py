"""Try the official-instrumental route on a few songs: does an instrumental exist, does it null?"""
import json
import os
import time

from rave import assist, audio, separate

man = json.load(open("/dev-scripts/refs/manifest.json"))["summermash16"]["songs"]
for song in ["Flo Rida - My House", "Zara Larsson - Lush Life", "Justin Timberlake - Can't Stop The Feeling",
             "Lukas Graham - 7 Years"]:
    p = man[song]["path"]
    y = audio.load(p)
    key = separate.file_key(p)
    for f in os.listdir("/data/cache"):
        if f.startswith(f"{key}.assist"):
            os.remove(os.path.join("/data/cache", f))      # test fresh
    t = time.time()
    got = assist.separate(y, song, "/data/cache", key, separate.split_instrumental, print)
    info = open(f"/data/cache/{key}.assist.txt").read().strip().replace("\n", " | ") if got else \
        open(f"/data/cache/{key}.assist.none").read().strip()
    print(f"== {song}: {'STUDIO STEMS' if got else 'fallback to models'} ({time.time() - t:.0f}s) {info}\n", flush=True)
