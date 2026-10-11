"""Stage 2 check: transcribe both songs, render the lyric-aware engines, and measure vocal flow.

Cut loudness is the voice level at every vocal cut (0 = silence, 1 = full voice); "two leads" is how
long two lead voices sound at once. Renders go to dev/stage2-<engine>.mp3 with their cue sheets.
"""
import copy
import sys
import time

import numpy as np

from rave import analysis as an, audio, engine, lyrics, mix, separate, styles
from rave.engine import ENV_HOP, Track

# Same lineup order as the benchmark job (Runaway first).
import os
from rave import sources
songs = dict(x.split("=", 1) for x in os.environ["SONGS"].split(";")) if os.environ.get("SONGS") else     {"-ehVlh25p4A": "Galantis - Runaway (U & I)", "a-UyKC9ApI4": "Peking Duk - Fire"}
paths = [sources.download_youtube(k, "/data/cache") for k in songs]
opts = engine.Options(cache_dir="/data/cache", seed=7)
t0 = time.time()
tracks = []
for p, name in zip(paths, songs.values()):
    y = audio.load(p)
    key = separate.file_key(p)
    v, d, b, o = separate.separate(y, "demucs", "/data/cache", key)
    t = Track(name, p, an.analyze(y, o + b), v, d, b, o)
    t.tuning, t.f0 = an.tuning(o + b), an.vocal_pitch(v)
    s = time.time()
    ws = lyrics.transcribe(v, "/data/cache", f"{key}.demucs")
    t.words = [(x[0], x[1], x[2]) for x in ws]
    t.line_start = [bool(x[3]) if len(x) > 3 else False for x in ws]
    print(f"{name}: {len(t.words)} words ({time.time() - s:.0f}s): "
          + " ".join(w for _, _, w in t.words[:30]), flush=True)
    tracks.append(t)

base = an.choose_base([t.a for t in tracks])
T = engine._beat_len(tracks[base].a.bpm)
for i, t in enumerate(tracks):
    style = styles.get("earworm")
    t.fold = 1.0 if i == base else an.tempo_fold(t.a.bpm, tracks[base].a.bpm)
engine._plan_keys(tracks, base, opts, "least_damage")
for t in tracks:
    engine._conform(t, T)
    engine._profile(t, 8, T)
print(f"conformed [{time.time() - t0:.0f}s]", flush=True)

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
    return float(t.env[max(0, f - 2):f + 3].mean() / (np.percentile(t.env, 95) + 1e-9))


def in_word(t, x):
    return any(a + 0.02 * audio.SR < x < b - 0.02 * audio.SR for a, b, _ in t.words or [])


for key in [a for a in sys.argv[1:] if not a.startswith("--")] or ["earworm", "riffoff", "danthology"]:
    ts = [copy.copy(t) for t in tracks]
    log.clear()
    opts.style = key
    st = styles.STYLES[key]
    parts = st.arrange(styles.Ctx(ts, base, opts, 4 * T))
    out = engine.render(ts, parts, 4 * T, opts)
    audio.save(f"/dev-scripts/stage2-{os.environ.get('TAG', '')}{key}.mp3", out)
    cuts = [(t, a) for t, a, b, da, db, g in log] + [(t, b) for t, a, b, da, db, g in log]
    loud = np.array([level(t, x) for t, x in cuts])
    words = sum(in_word(t, x) for t, x in cuts)
    solo = [(t, a, da, db) for t, a, b, da, db, g in log if g >= 0.99]
    clash = 0.0
    for x in solo:
        for y in solo:
            lo, hi = max(x[2], y[2]), min(x[3], y[3])
            if x is not y and hi > lo and x[2] < y[2]:
                for q in range(lo, hi, ENV_HOP):
                    if level(x[0], x[1] + q - x[2]) > 0.15 and level(y[0], y[1] + q - y[2]) > 0.15:
                        clash += ENV_HOP / audio.SR
    links = sum(1 for p in parts if "(on " in p.label or "(rhymes " in p.label)
    for i, t in enumerate(ts):
        print(f"  {t.name}: {len(t.lines)} lyric lines, e.g. " + " | ".join(L['text'] for L in t.lines[:6]))
    j = getattr(styles.Ctx, "_last", None)
    report = engine.describe(ts, base, parts, 4 * T, out.shape[1], "demucs", time.time() - t0, st,
                             (tracks[base].a.key, tracks[base].a.minor))
    with open(f"/dev-scripts/stage2-{os.environ.get('TAG', '')}{key}.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n== {key}: vocal runs {len(log)}, cut loudness median {np.median(loud):.2f}, "
          f"cuts inside a word {words}/{len(cuts)}, two leads at once {clash:.1f}s, "
          f"word/rhyme handoffs {links}", flush=True)
    print(report.split("Arrangement:")[1], flush=True)

# Link diagnostics: for every bar line in each song, the last word before it, and the words that
# could answer it at the start of a passage in the other song.
if "--links" in sys.argv[1:]:
    c = styles.Ctx(tracks, base, opts, 4 * T)
    for i, t in enumerate(tracks):
        j = 1 - i
        ends = {}
        for e in range(1, t.nbars()):
            w = c.last_word(i, e)
            if w:
                ends.setdefault(w, []).append(e)
        starts = {}
        for s in range(t.nbars() and tracks[j].nbars()):
            for w in c.first_words(j, s):
                starts.setdefault(w, []).append(s)
        pairs = [(a, b) for a in ends for b in starts if lyrics.link(a, b) > 0]
        print(f"{t.name} -> {tracks[j].name}: last words {sorted(ends)[:40]}")
        print(f"   possible links: {pairs[:20]}")

if "--trace" in sys.argv[1:]:
    c = styles.Ctx(tracks, base, opts, 4 * T)
    for i in range(2):
        j = 1 - i
        L = c.lines[j]
        print(f"{tracks[j].name}: {len(L.bars)} candidate bars, verses {L.verses}")
        hits = 0
        for e in range(1, tracks[i].nbars()):
            w = c.last_word(i, e)
            if not w:
                continue
            best = max(((c.link((i, e), j, p)[0], p) for p in L.bars), default=(0, None))
            if best[0] > 0:
                hits += 1
                if hits <= 6:
                    print(f"  {tracks[i].name} ends bar {e} on {w!r} -> {tracks[j].name} bar {best[1]} "
                          f"opens {c.first_words(j, best[1])} ({best[0]})")
        print(f"  {hits} answerable endings")

if "--lines" in sys.argv[1:]:
    for t in tracks:
        print(f"\n{t.name}: {len(t.lines)} lines, {sum(L['clean'] for L in t.lines)} clean")
        for L in t.lines[:40]:
            ws = [(round((a - L['start']) / audio.SR, 2), w) for a, b, w in t.words if L['start'] <= a <= L['end']]
            print(f"  b{L['b0']:3d}-{L['b1']:3d} {'C' if L['clean'] else '-'} {(L['end']-L['start'])/audio.SR:4.1f}s  {L['text'][:70]}")
