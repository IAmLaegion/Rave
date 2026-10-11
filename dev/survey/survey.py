"""Survey of popular mashups, nightcore and 8D tracks: what do the successful ones have in common?

    python survey.py            (downloads, separates on StemDeck, measures; all cached; resumable)

For each track: length, tempo, key, time to the first vocal, share of the track with vocals, how the
energy rises and falls, loudness, stereo width, and for 8D tracks how fast the sound circles. Writes
survey.json and prints a summary per category.
"""
import json
import os
import sys
import time

import librosa
import numpy as np

from rave import analysis as an
from rave import audio, separate, sources

OUT = "/dev-scripts/survey/survey.json"
CACHE = "/data/cache"

QUERIES = {
    "earworm": ["DJ Earworm United State of Pop 2009", "DJ Earworm United State of Pop 2010",
                "DJ Earworm United State of Pop 2011", "DJ Earworm United State of Pop 2012",
                "DJ Earworm United State of Pop 2013", "DJ Earworm United State of Pop 2014",
                "DJ Earworm United State of Pop 2015", "DJ Earworm United State of Pop 2016",
                "DJ Earworm United State of Pop 2017", "DJ Earworm Summermash 14",
                "DJ Earworm Summermash 15", "DJ Earworm Summermash 17", "DJ Earworm Summermash 18"],
    "megamix": ["Pop Danthology 2012", "Pop Danthology 2013", "Pop Danthology 2015",
                "Daniel Kim Pop Danthology 2016", "minlee mashup most streamed songs in 1 beat",
                "Pop Danthology 2011", "Pop Danthology 2014", "Daniel Kim Pop Danthology 2017 mashup",
                "Kap Slap mashup 2014",
                "Kap Slap mashup", "AnDyWuMUSICLAND Perfect Struggle mashup"],
    "pitchperfect": ["Pitch Perfect riff off scene audio", "Pitch Perfect Bellas finals mashup",
                     "Pitch Perfect 2 world championship finals Bellas", "Pitch Perfect 3 riff off",
                     "Pitch Perfect Treblemakers finals"],
    "nightcore": ["nightcore mashup switching vocals", "nightcore mashup 2 songs",
                  "nightcore megamix mashup", "nightcore mashup lyrics popular",
                  "nightcore alan walker mashup", "nightcore chainsmokers mashup"],
    "8d": ["8D audio song use headphones", "8D audio Blinding Lights", "8D audio Shape of You",
           "8D audio Alan Walker Faded", "8D audio Billie Eilish", "8D audio Imagine Dragons Believer"],
}


def pick(query, seen):
    for r in sources.search_youtube(query, 8):
        d = r.get("duration") or 0
        if r["id"] in seen or not 90 <= d <= 600:
            continue
        if "reaction" in r["title"].lower() or "1 hour" in r["title"].lower():
            continue
        return r
    return None


def first_sustained(active, times, need=1.5):
    """Time of the first stretch of `need` seconds where `active` stays mostly true."""
    step = times[1] - times[0]
    k = max(1, int(need / step))
    for i in range(len(active) - k):
        if active[i:i + k].mean() > 0.7:
            return float(times[i])
    return None


def measure(path):
    y = audio.load(path)
    key = separate.file_key(path)
    method = "stemdeck" if os.environ.get("RAVE_STEMDECK_URL") else "demucs"
    v, d, b, o = separate.separate(y, method, CACHE, key)
    a = an.analyze(y, o + b)
    sr, hop = 22050, 1024
    mono = lambda s: librosa.resample(audio.mono(s), orig_sr=audio.SR, target_sr=sr)
    rv = librosa.feature.rms(y=mono(v), hop_length=hop)[0]
    rm = librosa.feature.rms(y=mono(y), hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rv)), sr=sr, hop_length=hop)
    vocal = rv > 0.15 * np.percentile(rv, 95)
    dur = y.shape[1] / audio.SR
    # Energy curve: loudness in 10 slices of the track, relative to the loudest.
    slices = np.array_split(rm, 10)
    curve = [round(float(20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-9)), 1) for s in slices]
    curve = [round(c - max(curve), 1) for c in curve]
    # Stereo: correlation of L and R, and how the balance moves (8D tracks swing it around).
    L, R = y[0][::4].astype(np.float64), y[1][::4].astype(np.float64)
    corr = float(np.corrcoef(L, R)[0, 1])
    fr = audio.SR // 4 // 20                               # 20 balance readings per second
    n = len(L) // fr
    el = np.sqrt(np.mean(L[:n * fr].reshape(n, fr) ** 2, axis=1))
    er = np.sqrt(np.mean(R[:n * fr].reshape(n, fr) ** 2, axis=1))
    bal = (er - el) / (er + el + 1e-9)
    bal = bal - bal.mean()
    ac = np.correlate(bal, bal, "full")[n - 1:]
    ac /= ac[0] + 1e-12
    lo, hi = 20 * 3, min(len(ac) - 1, 20 * 30)            # look for a period of 3-30 s
    period = (lo + int(np.argmax(ac[lo:hi]))) / 20 if hi > lo else None
    swing = float(np.std(bal))
    peak = float(np.abs(y).max())
    return dict(
        duration=round(dur, 1), bpm=round(a.bpm, 1), key=a.key_name,
        first_vocal=first_sustained(vocal, times), vocal_share=round(float(vocal.mean()), 3),
        energy_curve=curve, rms_db=round(float(20 * np.log10(audio.rms(y) + 1e-9)), 1),
        crest_db=round(float(20 * np.log10(peak / (audio.rms(y) + 1e-9))), 1),
        stereo_corr=round(corr, 3), balance_swing=round(swing, 3),
        rotation_period=round(period, 1) if period and ac[int(period * 20)] > 0.3 else None,
    )


def main():
    data = json.load(open(OUT)) if os.path.isfile(OUT) else {}
    seen = {e["id"] for cat in data.values() for e in cat}
    for cat, queries in QUERIES.items():
        data.setdefault(cat, [])
        done_q = {e["query"] for e in data[cat]}
        for q in queries:
            if q in done_q:
                continue
            try:
                r = pick(q, seen)
                if not r:
                    print("  nothing for", q, flush=True)
                    continue
                time.sleep(15)                         # gently: YouTube flags servers that hurry
                path = sources.download_youtube(r["id"], CACHE)
                m = measure(path)
                e = dict(query=q, id=r["id"], title=r["title"], **m)
                data[cat].append(e)
                seen.add(r["id"])
                print(f"[{cat}] {r['title'][:60]}: {m}", flush=True)
            except Exception as ex:
                print("  FAILED", q, ex, flush=True)
            json.dump(data, open(OUT, "w"), indent=1)
    summarize(data)


def summarize(data):
    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.median(xs)), 2) if xs else None
    print("\nSUMMARY (medians)")
    for cat, es in data.items():
        if not es:
            continue
        print(f"  {cat:13s} n={len(es):2d}  length {med([e['duration'] for e in es])}s  "
              f"bpm {med([e['bpm'] for e in es])}  first vocal {med([e['first_vocal'] for e in es])}s  "
              f"vocals {med([e['vocal_share'] for e in es])}  rms {med([e['rms_db'] for e in es])} dB  "
              f"crest {med([e['crest_db'] for e in es])} dB  L/R corr {med([e['stereo_corr'] for e in es])}  "
              f"swing {med([e['balance_swing'] for e in es])}  rotation {med([e['rotation_period'] for e in es])}s")
        curve = np.median(np.array([e["energy_curve"] for e in es]), axis=0)
        print(f"  {'':13s} energy by tenths: {[round(float(c), 1) for c in curve]}")


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summarize(json.load(open(OUT)))
    else:
        main()
