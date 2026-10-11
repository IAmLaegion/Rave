"""Render every style from one analysis pass (key planning uses the base key for all)."""
import copy, sys, time
from rave import engine, styles, audio, analysis as an, separate
from rave.engine import Track
songs = {"a-UyKC9ApI4": "Peking Duk - Fire", "-ehVlh25p4A": "Galantis - Runaway (U & I)"}
paths = [f"/data/cache/youtube/{k}.webm" for k in songs]
t0 = time.time()
opts = engine.Options(cache_dir="/data/cache", seed=7)
tracks = []
for p, name in zip(paths, songs.values()):
    y = audio.load(p)
    v, d, b, o = separate.separate(y, "demucs", "/data/cache", separate.file_key(p))
    pitched = o + b
    t = Track(name, p, an.analyze(y, pitched), v, d, b, o)
    t.tuning, t.f0 = an.tuning(pitched), an.vocal_pitch(v)
    print(f"{name}: {t.a.bpm:.2f} BPM {t.a.key_name} steady={t.a.steady} tuning={t.tuning:+.2f} f0={t.f0:.0f}Hz  [{time.time()-t0:.0f}s]", flush=True)
    tracks.append(t)
base = an.choose_base([t.a for t in tracks])
T = engine._beat_len(tracks[base].a.bpm)
for mode in ("base", "least_damage"):
    k = engine._plan_keys(copy.copy(tracks), base, opts, mode)
    print(mode, "->", an.NOTES[k[0]], [ (t.shift, round(t.tune,2)) for t in tracks])
engine._plan_keys(tracks, base, opts, "base")
for i, t in enumerate(tracks):
    t.fold = 1.0 if i == base else an.tempo_fold(t.a.bpm, tracks[base].a.bpm)
    engine._conform(t, T); engine._profile(t, 8, T)
    print(f"{t.name}: grid from bar {t.grid[0]}, phrases {t.phrases[:8]}, chorus {t.chorus}, choruses {sorted(t.choruses)}, hook {t.hook}  [{time.time()-t0:.0f}s]", flush=True)
only = sys.argv[1:] or list(styles.STYLES)
for key in only:
    st = styles.STYLES[key]
    ts = [copy.copy(t) for t in tracks]
    opts.style = key
    parts = st.arrange(styles.Ctx(ts, base, opts, 4 * T))
    out = engine.render(ts, parts, 4 * T, opts)
    audio.save(f"/dev-scripts/style-{key}.mp3", out)
    print(engine.describe(ts, base, parts, 4 * T, out.shape[1], "demucs", time.time() - t0, st, (tracks[base].a.key, tracks[base].a.minor)).split("Arrangement:")[1], flush=True)
