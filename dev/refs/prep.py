"""Separate and analyze every reference, original and source song in the manifest (cached), so decoding
is fast. Order: refs and their songs one at a time, so the first reference is decodable soonest."""
import json
import sys
import time

sys.path.insert(0, "/dev-scripts/refs")
from features import features  # noqa: E402

only = sys.argv[1:]
man = json.load(open("/dev-scripts/refs/manifest.json"))
t0 = time.time()
for ref, e in man.items():
    if only and ref not in only:
        continue
    paths = [e.get("video_path"), e.get("original_path")] + [s["path"] for s in e["songs"].values()]
    for p in [p for p in paths if p]:
        try:
            features(p, progress=lambda m: print(f"[{time.time() - t0:6.0f}s] {ref}: {m}", flush=True))
        except Exception as ex:
            print(f"FAILED {ref} {p}: {ex}", flush=True)
    print(f"[{time.time() - t0:6.0f}s] {ref} ready", flush=True)
