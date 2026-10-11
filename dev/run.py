import sys, time, glob
from rave import engine, analysis as an, audio
songs = {"a-UyKC9ApI4": "Peking Duk - Fire", "-ehVlh25p4A": "Galantis - Runaway (U & I)"}
paths = [f"/data/cache/youtube/{k}.webm" for k in songs]
t0 = time.time()
def say(f, m): print(f"[{time.time()-t0:6.1f}s {f*100:5.1f}%] {m}", flush=True)
out = sys.argv[1] if len(sys.argv) > 1 else "/dev-scripts/out.mp3"
opts = engine.Options(cache_dir="/data/cache")
print(engine.make_mashup(paths, out, opts, say, names=list(songs.values())))
