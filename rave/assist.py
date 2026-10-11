"""Studio-quality stems without a separation model, when the song has an official instrumental.

If the instrumental comes from the same master as the song, then song - instrumental is the exact
studio vocal (the trick mashup makers have always used), and the instrumental itself is clean. So for
each song: find its official instrumental on YouTube, line it up to the sample (and correct a slight
speed difference), match its level, and subtract. A null test decides whether it really is the same
master: in the passages without singing, what's left must drop to near silence. If it doesn't (a
different master, an edit, a remake) the song is separated with the models as usual.

The result is cached next to the stems (and so is "no usable instrumental"), so each song is looked
up once.
"""
import os
import re
import time

import numpy as np
import soundfile as sf
from scipy import signal

from . import audio

BAD = re.compile(r"\b(karaoke|cover|piano|acoustic|remix|live|sped|slowed|nightcore|8d|reverb|lyrics?|"
                 r"guitar|violin|orchestra|tutorial|lesson|backing track|minus one|re-?made|remake|"
                 r"fl studio|cover by)\b", re.I)
NULL_DB = -20.0          # quietest fifth of the residual, relative to the song, for a true null


def _search(name, duration):
    from . import sources
    seen = set()
    for q in (f"{name} official instrumental", f"{name} instrumental"):
        for r in sources.search_youtube(q, 8):
            t = r["title"]
            if r["id"] in seen or "instrumental" not in t.lower() or BAD.search(t.replace("Instrumental", "")):
                continue
            seen.add(r["id"])
            if r.get("duration") and abs(r["duration"] - duration) > 6:
                continue
            yield r


def _lag(a, b, sr):
    """Samples by which b must be delayed to line up with a (mono signals)."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    xc = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    k = int(np.argmax(xc))
    return k if k < n // 2 else k - n


def align(mix, inst):
    """`inst` (2, n) shifted (and time-scaled) to line up with `mix` sample for sample, plus the level
    that best matches; None if they don't line up."""
    sr = audio.SR
    m = audio.mono(mix)
    i = audio.mono(inst)
    # Coarse: on a 4 kHz version of the whole track.
    dec = 11
    lag = _lag(signal.decimate(m, dec), signal.decimate(i, dec), sr // dec) * dec
    # Fine, at the start and the end separately, to catch a speed difference between the two files.
    def fine(at):
        w = sr * 8
        a0 = max(0, min(len(m) - w, at))
        b0 = a0 - lag
        if b0 < 0 or b0 + w > len(i):
            return None
        return lag + _lag(m[a0:a0 + w], i[b0:b0 + w], sr)
    first, last = fine(sr * 20), fine(len(m) - sr * 28)
    if first is None or last is None:
        return None
    span = (len(m) - sr * 28) - sr * 20
    drift = (last - first) / span                          # extra samples per sample
    t = np.arange(mix.shape[1], dtype=np.float64)
    src = t - first - drift * (t - sr * 20)                # where in inst each mix sample is
    out = np.stack([np.interp(src, np.arange(inst.shape[1]), ch, left=0.0, right=0.0) for ch in inst])
    # Level: least squares per channel.
    g = [float(np.dot(mix[c], out[c]) / (np.dot(out[c], out[c]) + 1e-12)) for c in range(2)]
    return (out * np.array(g)[:, None]).astype(np.float32)


def null_depth(mix, residual):
    """How well the subtraction nulls: the quietest fifth of 1-second windows of the residual, in dB
    relative to the song in the same windows. A true null (same master) is far below -20 dB there."""
    sr = audio.SR
    n = mix.shape[1] // sr
    if n < 10:
        return 0.0
    r = np.sqrt(np.mean(audio.mono(residual)[: n * sr].reshape(n, sr) ** 2, axis=1))
    m = np.sqrt(np.mean(audio.mono(mix)[: n * sr].reshape(n, sr) ** 2, axis=1))
    loud = m > 0.1 * np.percentile(m, 90)
    ratio = 20 * np.log10((r[loud] + 1e-9) / (m[loud] + 1e-9))
    return float(np.percentile(ratio, 20)) if len(ratio) else 0.0


def separate(y, name, cache_dir, key, split_instrumental, progress=None):
    """(vocals, drums, bass, other) from the song's official instrumental, or None if there isn't a
    usable one. `split_instrumental(inst)` splits the clean instrumental into (drums, bass, other)."""
    say = progress or (lambda m: None)
    stems = ("vocals", "drums", "bass", "other")
    paths = [os.path.join(cache_dir, f"{key}.assist.{s}.flac") for s in stems]
    none = os.path.join(cache_dir, f"{key}.assist.none")
    if all(os.path.isfile(p) for p in paths):
        return tuple(sf.read(p, dtype="float32", always_2d=True)[0].T[:, :y.shape[1]] for p in paths)
    if os.path.isfile(none) or not name or os.environ.get("RAVE_ASSIST", "1") == "0":
        return None
    from . import sources
    duration = y.shape[1] / audio.SR
    best = None
    try:
        for k, r in enumerate(_search(name, duration)):
            if k >= 3:
                break
            say(f"Looking for {name}'s official instrumental...")
            time.sleep(3)                                  # gently: YouTube flags servers that hurry
            inst = audio.load(sources.download_youtube(r["id"], cache_dir))
            aligned = align(y, inst)
            if aligned is None:
                continue
            residual = y - aligned
            depth = null_depth(y, residual)
            if best is None or depth < best[0]:
                best = (depth, aligned, residual, r["title"])
            if depth <= NULL_DB:
                break
    except Exception as e:                                 # no network, YouTube refusing: just carry on
        say(f"No instrumental lookup ({str(e)[:60]})")
        return None
    if best is None or best[0] > NULL_DB:
        open(none, "w").write(f"best null {best[0]:.1f} dB ({best[3]})\n" if best else "none found\n")
        return None
    depth, inst, vocals, title = best
    say(f"Studio vocals from the official instrumental ({depth:.0f} dB null); splitting the instrumental...")
    d, b, o = split_instrumental(inst)
    out = (vocals.astype(np.float32), d, b, o)
    for p, s in zip(paths, out):
        sf.write(p, np.clip(s, -1, 1).T, audio.SR, subtype="PCM_24")
    with open(os.path.join(cache_dir, f"{key}.assist.txt"), "w", encoding="utf-8") as f:
        f.write(f"{title}\nnull {depth:.1f} dB\n")
    return out
