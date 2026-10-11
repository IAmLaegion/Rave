"""The mashup pipeline: load -> separate -> analyze -> conform -> arrange -> render -> master."""
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import analysis as an
from . import audio, mix, separate, transform


@dataclass
class Options:
    base: int = -1                  # index of the song whose beat/key everything follows; -1 = auto
    length: float = 150.0           # target length in seconds
    phrase_bars: int = 8            # bars per vocal turn
    bed: str = "base"               # "base": one backing track throughout; "rotate": backings rotate too
    separation: str = "auto"        # auto | demucs | fast
    match_key: bool = True
    transpose: dict = field(default_factory=dict)   # song index -> semitones, overrides auto key match
    acappella_drop: bool = True     # finale opens with one bar of voices only
    riser: bool = True
    stack_pan: float = 0.35         # how wide stacked voices are spread in the finale
    cache_dir: str = None


@dataclass
class Track:
    name: str
    path: str
    a: an.Analysis
    vocals: np.ndarray = None
    inst: np.ndarray = None
    fold: float = 1.0
    rate: float = 1.0
    shift: int = 0
    bars: np.ndarray = None          # bar starts (samples) in the conformed audio
    vocal_level: np.ndarray = None   # vocal RMS per bar
    active: np.ndarray = None        # bar has singing
    grid: list = None                # phrase start bars
    phrases: list = None             # sung phrase start bars, in song order
    vchroma: np.ndarray = None       # per-bar chroma of the vocals
    ichroma: np.ndarray = None       # per-bar chroma of the backing
    chorus: int = 0                  # start bar of the chorus phrase
    first_vocal: int = 0
    threshold: float = 0.0


@dataclass
class Part:
    label: str
    bars: int
    bed: list          # [(track, start_bar)], empty = a cappella
    vocals: list       # [(track, start_bar, gain, pan)]
    fade: bool = False


def _bar_len(bpm):
    return 4 * 60.0 / bpm * audio.SR


def make_mashup(paths, out_path, opts=None, progress=None, names=None):
    """Build the mashup and write it to `out_path`. Returns a text report (also saved next to it)."""
    opts = opts or Options()
    if len(paths) < 2:
        raise ValueError("Pick at least two songs.")
    say = progress or (lambda f, m: None)
    t0 = time.time()
    n = len(paths)
    sep_method = separate.resolve(opts.separation)
    cache = opts.cache_dir or default_cache_dir()

    tracks = []
    for i, p in enumerate(paths):
        name = names[i] if names else os.path.splitext(os.path.basename(p))[0]
        say(0.02 + 0.5 * i / n, f"Loading {name}")
        y = audio.load(p)
        say(0.02 + 0.5 * (i + 0.2) / n, f"Separating vocals: {name} ({sep_method})")
        vocals, inst = separate.separate(
            y, sep_method, cache, separate.file_key(p),
            progress=lambda m, f=0.02 + 0.5 * (i + 0.2) / n: say(f, m))
        say(0.02 + 0.5 * (i + 0.7) / n, f"Analyzing {name}")
        a = an.analyze(y, inst)
        tracks.append(Track(name, p, a, vocals, inst))
        del y

    base = opts.base if 0 <= opts.base < n else an.choose_base([t.a for t in tracks])
    B = tracks[base]
    # Keep the grid in a danceable range: a 60 BPM ballad is felt (and gridded) at 120.
    base_fold = 2.0 if B.a.bpm < 80 else 0.5 if B.a.bpm > 165 else 1.0
    bpm = B.a.bpm * base_fold
    bar_len = _bar_len(bpm)

    for i, t in enumerate(tracks):
        say(0.55 + 0.25 * i / n, f"Matching {t.name} to {bpm:.1f} BPM, {B.a.key_name}")
        t.fold = base_fold if i == base else an.tempo_fold(t.a.bpm, bpm)
        beats, _, phase = an.fold(t.a, t.fold)
        t.rate = 1.0 if i == base else bpm / (t.a.bpm * t.fold)
        if i in opts.transpose:
            t.shift = int(opts.transpose[i])
        elif opts.match_key and i != base:
            t.shift = an.key_shift(t.a.key, t.a.minor, B.a.key, B.a.minor)
        t.vocals = mix.highpass(transform.stretch_shift(t.vocals, t.rate, t.shift, vocal=True), 90)
        need_bed = i == base or opts.bed == "rotate"
        t.inst = transform.stretch_shift(t.inst, t.rate, t.shift) if need_bed else None
        t.bars = np.round(beats[phase::4] / t.rate * audio.SR).astype(np.int64)
        _profile(t, opts.phrase_bars)

    say(0.82, "Arranging")
    parts = arrange(tracks, base, opts, bar_len)
    say(0.86, "Mixing")
    out = render(tracks, parts, bar_len, opts)
    say(0.96, "Saving")
    audio.save(out_path, out)

    report = describe(tracks, base, parts, bar_len, out.shape[1], sep_method, time.time() - t0)
    with open(os.path.splitext(out_path)[0] + ".txt", "w", encoding="utf-8") as f:
        f.write(report)
    say(1.0, f"Done: {out_path}")
    return report


def default_cache_dir():
    return os.environ.get("RAVE_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "rave")


def _profile(t, phrase_bars):
    """Per-bar vocal activity, phrase grid, and chorus for a conformed track."""
    bars = t.bars[t.bars + 1 < t.vocals.shape[1]]
    t.bars = bars
    nb = len(bars) - 1
    level = np.array([audio.rms(t.vocals[:, bars[k]:bars[k + 1]]) for k in range(nb)])
    loud = np.percentile(level, 90) if nb else 0.0
    t.threshold = loud * 0.2
    t.vocal_level, t.active = level, level > t.threshold
    t.first_vocal = int(np.argmax(t.active)) if t.active.any() else 0

    anchor = t.first_vocal % phrase_bars
    t.grid = list(range(anchor, nb - phrase_bars + 1, phrase_bars)) or [0]
    frac = {g: t.active[g:g + phrase_bars].mean() for g in t.grid}
    t.phrases = [g for g in t.grid if g >= t.first_vocal and frac[g] >= 0.4] or t.grid[:1]

    t.vchroma = _bar_chroma(t.vocals, bars)
    t.ichroma = _bar_chroma(t.inst, bars) if t.inst is not None else None

    # Chorus: the phrase that repeats most, is loudest, and is sung throughout.
    feat = t.vchroma + (t.ichroma if t.ichroma is not None else 0)
    cands = [g for g in t.phrases if g + phrase_bars <= nb]
    if len(cands) < 2:
        t.chorus = cands[0] if cands else t.phrases[0]
        return
    vecs = np.stack([feat[g:g + phrase_bars].ravel() for g in cands])
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, 0)
    rep = np.sort(sim, axis=1)[:, -2:].sum(axis=1)
    energy = np.array([level[g:g + phrase_bars].mean() for g in cands])
    sung = np.array([frac[g] for g in cands])
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    t.chorus = cands[int(np.argmax(z(rep) + z(energy) + 2 * sung))]


def _bar_chroma(y, bars):
    """Mean chroma of each bar, shape (bars - 1, 12)."""
    import librosa
    sr, hop = 11025, 1024
    m = librosa.resample(audio.mono(y), orig_sr=audio.SR, target_sr=sr)
    chroma = librosa.feature.chroma_stft(y=m, sr=sr, hop_length=hop)
    f = np.minimum(chroma.shape[1] - 1, (bars / audio.SR * sr / hop).astype(int))
    return np.stack([chroma[:, f[k]:max(f[k] + 1, f[k + 1])].mean(axis=1)
                     for k in range(len(bars) - 1)]) if len(bars) > 1 else np.zeros((0, 12))


def fit(vt, vs, bt, bs, bars):
    """How well a vocal passage sits on a backing passage: the cosine between the notes being sung and
    the backing's chords, bar by bar, weighted by how much singing there is."""
    if bt.ichroma is None or vs < 0 or bs < 0 or vs + bars > len(vt.vchroma) or bs + bars > len(bt.ichroma):
        return -1.0
    v, b = vt.vchroma[vs:vs + bars], bt.ichroma[bs:bs + bars]
    cos = (v * b).sum(axis=1) / (np.linalg.norm(v, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    w = vt.vocal_level[vs:vs + bars] + 1e-9
    return float((cos * w).sum() / w.sum())


def arrange(tracks, base, opts, bar_len):
    """Pitch Perfect structure: intro on the base groove, every song takes a full vocal turn, the trades
    get faster and start overlapping, a riser into an a cappella hit, then every chorus stacked at once,
    and out. Each backing passage is chosen for how well its chords fit the voice singing over it."""
    n = len(tracks)
    P = opts.phrase_bars
    half = max(2, P // 2)
    B = tracks[base]
    order = [base] + [i for i in range(n) if i != base]
    total = max(1, int(round(opts.length * audio.SR / bar_len)))

    bed_pos = {}                                       # backing track -> where it would continue

    def bed_for(turn, bars, singer, vpos):
        i = base if opts.bed == "base" else order[(turn + 1) % n]
        t = tracks[i]
        last = len(t.bars) - 2 - 4                     # keep the backing's last bars for the outro
        if i == singer and 0 <= vpos and vpos + bars <= last:
            pos = vpos                                 # a song over its own backing stays in sync
        else:
            step = half if bars < P else P
            cands = sorted({g + o for g in t.grid for o in range(0, P, step)
                            if 0 <= g + o and g + o + bars <= last} or {0})
            cont = bed_pos.get(i)
            pos = max(cands, key=lambda c: fit(tracks[singer], vpos, t, c, bars) + (0.08 if c == cont else 0))
        bed_pos[i] = pos + bars
        return [(i, pos)]

    voc = {i: {"k": 0, "off": 0} for i in range(n)}

    def take_vocal(i, bars):
        t, c = tracks[i], voc[i]
        if c["k"] >= len(t.phrases):
            c["k"], c["off"] = 0, 0
        pos = t.phrases[c["k"]] + c["off"]
        c["off"] += bars
        if c["off"] >= P:
            c["k"], c["off"] = c["k"] + 1, 0
        return pos

    def peek_vocal(i):
        t, c = tracks[i], voc[i]
        return t.phrases[c["k"]] + c["off"] if c["k"] < len(t.phrases) else t.phrases[0]

    parts = []
    intro = int(min(8, max(4, B.first_vocal - B.first_vocal % 4)))
    intro_start = max(0, B.first_vocal - intro)
    parts.append(Part("Intro", intro, [(base, intro_start)], []))
    bed_pos[base] = intro_start + intro

    finale = 2 * P
    left = max(n * P, total - (intro + finale + 4))

    turn = 0
    for i in order:                                    # round 1: everyone gets a full phrase
        v = take_vocal(i, P)
        parts.append(Part(f"Verse: {tracks[i].name}", P, bed_for(turn, P, i, v), [(i, v, 1.0, 0.0)]))
        turn += 1
    trades = max(0, (left - n * P) // half)
    for k in range(trades):                            # round 2: faster trades
        i, j = order[k % n], order[(k + 1) % n]
        v = take_vocal(i, half)
        vocals, label = [(i, v, 1.0, 0.0)], f"Trade: {tracks[i].name}"
        if k >= trades - n:                            # the last few overlap: the next voice answers
            vocals.append((j, peek_vocal(j), 0.55, 0.3 if k % 2 else -0.3))
            label += f" + {tracks[j].name}"
        parts.append(Part(label, half, bed_for(turn, half, i, v), vocals))
        turn += 1

    # Finale: all the choruses stacked over the base's chorus.
    pans = np.linspace(-opts.stack_pan, opts.stack_pan, n)
    gain = 1.25 / np.sqrt(n)
    stack = lambda off: [(i, tracks[i].chorus + off, gain, float(pans[k])) for k, i in enumerate(order)]
    if opts.acappella_drop:
        parts.append(Part("Finale (a cappella)", 1, [], stack(0)))
        parts.append(Part("Finale: everyone", P - 1, [(base, B.chorus + 1)], stack(1)))
    else:
        parts.append(Part("Finale: everyone", P, [(base, B.chorus)], stack(0)))
    # Second time round: carry on into the rest of each chorus if it keeps going, else repeat it.
    second = []
    for k, i in enumerate(order):
        t = tracks[i]
        nxt = t.chorus + P
        keep = nxt + P <= len(t.active) and t.active[nxt:nxt + P].mean() >= 0.5
        second.append((i, nxt if keep else t.chorus, gain, float(pans[k])))
    bed2 = B.chorus + P if B.chorus + 2 * P + 5 < len(B.bars) else B.chorus
    parts.append(Part("Finale: everyone (again)", P, [(base, bed2)], second))
    outro = bed2 + P if bed2 + P + 5 < len(B.bars) else intro_start
    parts.append(Part("Outro", 4, [(base, outro)], [], fade=True))
    return parts


def render(tracks, parts, bar_len, opts):
    total_bars = sum(p.bars for p in parts)
    beat = int(bar_len / 4)
    n_out = int(np.ceil(total_bars * bar_len)) + audio.SR * 2
    bed = np.zeros((2, n_out), dtype=np.float32)
    voc = np.zeros((2, n_out), dtype=np.float32)

    # Level-match every stem before anything is placed.
    for t in tracks:
        t.vocals = mix.normalize_rms(t.vocals, -19.0, mask=_active_mask(t))
        if t.inst is not None:
            t.inst = mix.normalize_rms(t.inst, -18.0)

    # Flatten to per-bar events per lane, then merge into runs so joints inside a run are seamless.
    events = {"bed": [], "voc": []}
    d = 0
    for p in parts:
        for (i, s) in p.bed:
            for j in range(p.bars):
                events["bed"].append((i, s + j, d + j, 1.0, 0.0))
        for (i, s, g, pan) in p.vocals:
            for j in range(p.bars):
                events["voc"].append((i, s + j, d + j, g, pan))
        d += p.bars

    edge = mix.ms(5)
    for lane, dst in (("bed", bed), ("voc", voc)):
        for run in _runs(events[lane]):
            i, _, _, g, pan = run[0]
            t = tracks[i]
            src = t.inst if lane == "bed" else t.vocals
            for k, (_, s, dbar, _, _) in enumerate(run):
                if s < 0 or s + 1 >= len(t.bars):
                    continue
                first, last = k == 0, k == len(run) - 1
                pre, fin = edge, 2 * edge
                post, fout = edge, 2 * edge
                if lane == "voc" and first and s > 0:
                    # Let a pickup into the phrase through (lyrics often start before the downbeat).
                    a = t.bars[s]
                    if audio.rms(t.vocals[:, max(0, a - beat):a]) > t.threshold:
                        pre, fin = beat, mix.ms(20)
                if lane == "voc" and last:
                    post, fout = beat // 2, beat // 2
                if lane == "bed" and (first or last):
                    if first:
                        pre, fin = mix.ms(30), mix.ms(60)
                    if last:
                        post, fout = mix.ms(30), mix.ms(60)
                length = int(round((dbar + 1) * bar_len)) - int(round(dbar * bar_len))
                mix.place(dst, src, int(t.bars[s]), int(round(dbar * bar_len)), length,
                          pre, post, fin, fout, g, pan)

    bed = mix.duck(bed, voc, 3.0)
    out = bed + voc

    # Riser into the finale.
    if opts.riser:
        d = 0
        for p in parts:
            if p.label.startswith("Finale"):
                break
            d += p.bars
        start = int(round(max(0, d - 2) * bar_len))
        end = int(round(d * bar_len))
        if end > start:
            out[:, start:end] += mix.riser(end - start, peak=2.0 * audio.rms(out[:, :end]))

    # Outro fade, trim, master.
    end = int(round(total_bars * bar_len)) + audio.SR // 2
    out = out[:, :end]
    if parts[-1].fade:
        mix.fade_out(out, int(round(parts[-1].bars * bar_len)) + audio.SR // 2)
    out = mix.normalize_rms(out, -13.0)
    out = mix.limit(out, -1.5)   # headroom for lossy encoders
    return mix.fade_out(out, mix.ms(10))


def _active_mask(t):
    m = np.zeros(t.vocals.shape[1], dtype=bool)
    for k, on in enumerate(t.active):
        if on:
            m[t.bars[k]:t.bars[k + 1]] = True
    return m


def _runs(evts):
    """Group (track, src_bar, dst_bar, gain, pan) events into runs that continue seamlessly."""
    evts = sorted(evts, key=lambda e: (e[0], e[3], e[4], e[2]))
    runs = []
    for e in evts:
        r = runs[-1] if runs else None
        if r and r[-1][0] == e[0] and r[-1][3] == e[3] and r[-1][4] == e[4] \
                and r[-1][1] + 1 == e[1] and r[-1][2] + 1 == e[2]:
            r.append(e)
        else:
            runs.append([e])
    return runs


def describe(tracks, base, parts, bar_len, n_samples, sep_method, secs):
    bpm = 4 * 60.0 * audio.SR / bar_len
    lines = ["Rave cue sheet", "",
             f"Tempo {bpm:.1f} BPM, key {tracks[base].a.key_name} ({tracks[base].a.camelot}), "
             f"length {n_samples / audio.SR:.0f}s",
             f"Separation: {sep_method}; stretch/shift: {transform.backend()}; built in {secs:.0f}s", "",
             "Songs:"]
    for i, t in enumerate(tracks):
        tag = " [base]" if i == base else ""
        fold = {2.0: " (double time)", 0.5: " (half time)"}.get(t.fold, "")
        lines.append(f"  {i + 1}. {t.name}{tag}: {t.a.bpm:.1f} BPM {t.a.key_name} ({t.a.camelot})"
                     f" -> x{t.rate:.3f}{fold}, {t.shift:+d} st; chorus at bar {t.chorus + 1}")
    lines += ["", "Arrangement:"]
    d = 0
    for p in parts:
        sec = d * bar_len / audio.SR
        lines.append(f"  {int(sec // 60)}:{sec % 60:04.1f}  {p.bars:2d} bars  {p.label}")
        d += p.bars
    return "\n".join(lines) + "\n"
