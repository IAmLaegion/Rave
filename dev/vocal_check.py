"""Render styles and measure vocal flow: loudness at every vocal cut (0 = silence, 1 = full voice) and
how long two lead voices sound at once."""
import copy, sys, time
import numpy as np
from rave import engine, styles, audio, analysis as an, separate, mix, lyrics
from rave.engine import Track, ENV_HOP
songs = {"a-UyKC9ApI4": "Peking Duk - Fire", "-ehVlh25p4A": "Galantis - Runaway (U & I)"}
paths = [f"/data/cache/youtube/{k}.webm" for k in songs]
opts = engine.Options(cache_dir="/data/cache", seed=7)
tracks = []
for p, name in zip(paths, songs.values()):
    y = audio.load(p)
    v, d, b, o = separate.separate(y, "demucs", "/data/cache", separate.file_key(p))
    t = Track(name, p, an.analyze(y, o + b), v, d, b, o); t.tuning, t.f0 = an.tuning(o + b), an.vocal_pitch(v)
    t0 = time.time(); t.words = lyrics.transcribe(v, "/data/cache", separate.file_key(p) + ".demucs"); print(name, len(t.words), "words in", round(time.time() - t0), "s:", " ".join(w for _, _, w in t.words[:25]), flush=True)
    tracks.append(t)
base = an.choose_base([t.a for t in tracks]); T = engine._beat_len(tracks[base].a.bpm)
engine._plan_keys(tracks, base, opts, "base")
for i, t in enumerate(tracks):
    t.fold = 1.0 if i == base else an.tempo_fold(t.a.bpm, tracks[base].a.bpm)
    engine._conform(t, T); engine._profile(t, 8, T)

log = []
real_place = mix.place
def spy(dst, src, s, d, length, pre=0, post=0, fade_in=0, fade_out=0, gain=1.0, pan=0.0):
    for t in ts:
        if src is t.vocals:
            log.append((t, s - pre, s + length + post, d - pre, d + length + post, gain))
    return real_place(dst, src, s, d, length, pre, post, fade_in, fade_out, gain, pan)
engine.mix.place = spy

def level(t, x):
    f = min(max(0, x // ENV_HOP), len(t.env) - 1)
    peak = np.percentile(t.env, 95) + 1e-9
    return float(t.env[max(0, f - 2):f + 3].mean() / peak)

for key in sys.argv[1:] or ["riffoff", "earworm", "danthology", "swap", "ravedj", "automashupper"]:
    ts = [copy.copy(t) for t in tracks]; log.clear(); opts.style = key
    parts = styles.STYLES[key].arrange(styles.Ctx(ts, base, opts, 4 * T))
    out = engine.render(ts, parts, 4 * T, opts)
    audio.save(f"/dev-scripts/style-{key}.mp3", out)
    cuts = [level(t, a) for t, a, b, da, db, g in log] + [level(t, b) for t, a, b, da, db, g in log]
    solo = [(t, a, b, da, db) for t, a, b, da, db, g in log if g >= 0.99]
    clash = 0.0
    for x in solo:
        for y in solo:
            if x is y or x[0] is y[0] and False: continue
            lo, hi = max(x[3], y[3]), min(x[4], y[4])
            if hi > lo and x[3] < y[3]:
                # both voices audible in the overlap?
                for q in range(lo, hi, ENV_HOP):
                    if level(x[0], x[1] + q - x[3]) > 0.15 and level(y[0], y[1] + q - y[3]) > 0.15:
                        clash += ENV_HOP / audio.SR
    cuts = np.array(cuts)
    print(f"{key:14s} vocal runs {len(log):3d}  cut loudness median {np.median(cuts):.2f}  "
          f"cuts in a word (>0.3): {int((cuts > 0.3).sum())}/{len(cuts)}  two leads at once: {clash:.1f}s", flush=True)
    if key in ("riffoff", "earworm", "danthology"):
        print(engine.describe(ts, base, parts, 4 * T, out.shape[1], "demucs", 0, styles.STYLES[key], (0, False)).split("Arrangement:")[1], flush=True)
