"""The mashup pipeline: load -> separate -> analyze -> conform -> arrange -> render -> master.

The arrangement itself is pluggable: `styles.py` holds one arranger per approach (Pitch Perfect
riff-off, DJ Earworm, AutoMashUpper, ...). They all hand back a list of Parts, which say which stem of
which song plays where, and the renderer here turns those into audio.
"""
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import analysis as an
from . import audio, lyrics, mix, separate, transform

ENV_HOP = 441            # vocal envelope resolution: 10 ms
GAP = 6                  # envelope frames of quiet that count as a breath between lines (60 ms)
STEMS = ("vocals", "drums", "bass", "other")


@dataclass
class Options:
    style: str = "earworm"          # which arranger, see styles.STYLES
    base: int = -1                  # index of the song whose beat everything follows; -1 = auto
    length: float = 0.0             # target length in seconds; 0 = the style's own
    phrase_bars: int = 8            # bars per vocal turn
    bed: str = "base"               # "base": one backing track throughout; "rotate": backings rotate too
    separation: str = "auto"        # auto | demucs | fast
    match_key: bool = True
    transpose: dict = field(default_factory=dict)   # song index -> semitones, overrides auto key match
    acappella_drop: bool = True     # finale opens with one bar of voices only
    riser: bool = True
    stack_pan: float = 0.35         # how wide stacked voices are spread in the finale
    seed: int = None                # randomness for styles that use it (rave.dj); None = a new take
    nightcore: float = 0.0          # nightcore speed-up (references measured 1.20x and 1.23x); 0 = off
    spatial: str = ""               # "8d": binaural, voices circling the head (for headphones)
    bgv: float = 0.4                # level of the backing song's own vocal in the gaps between lead lines
    cache_dir: str = None


@dataclass
class Track:
    name: str
    path: str
    a: an.Analysis
    vocals: np.ndarray = None
    drums: np.ndarray = None         # None with the fast separator
    bass: np.ndarray = None          # None with the fast separator
    other: np.ndarray = None         # chords and hooks (the whole instrumental with the fast separator)
    tuning: float = 0.0              # semitones off A440
    f0: float = 0.0                  # median sung pitch, Hz (0 = unknown)
    fold: float = 1.0
    rate: float = 1.0
    shift: int = 0                   # key change, semitones
    tune: float = 0.0                # extra fraction of a semitone to line up the tuning
    beats: np.ndarray = None         # beat starts (samples, float) in the conformed audio
    bars: np.ndarray = None          # bar starts (samples) in the conformed audio
    vocal_level: np.ndarray = None   # vocal RMS per bar
    bed_level: np.ndarray = None     # backing RMS per bar
    active: np.ndarray = None        # bar has singing
    grid: list = None                # phrase start bars
    phrases: list = None             # sung phrase start bars, in song order
    choruses: set = None             # phrases that sound like the chorus
    vchroma: np.ndarray = None       # per-bar chroma of the vocals
    ichroma: np.ndarray = None       # per-bar chroma of the backing's pitched parts (bass + other)
    rhythm: np.ndarray = None        # per-bar 16-step drum pattern
    bands: np.ndarray = None         # per-bar backing energy in low / mid / high bands
    chorus: int = 0                  # start bar of the chorus phrase
    hook: int = 0                    # start bar of the most hooky 4 instrumental bars
    first_vocal: int = 0
    threshold: float = 0.0
    env: np.ndarray = None           # vocal envelope, one value per ENV_HOP samples
    gate: float = 0.0                # envelope level below which the singer is between words
    line_start: list = None          # per word: it starts a lyric line
    lines: list = None               # lyric lines placed on the bar grid, see _lines()
    words: list = None               # [(start, end, word)]: seconds in the original, then (after
                                     # conforming) samples in the conformed audio
    quiet: np.ndarray = None         # per envelope frame: between words
    across: np.ndarray = None        # per bar line: how much singing runs straight through it

    def nbars(self):
        return len(self.bars) - 1


@dataclass
class Part:
    """A stretch of the mashup. Every list says which song plays which stem from which of its bars."""
    label: str
    bars: int
    bed: list = field(default_factory=list)      # [(track, bar)] drums + bass + other, together
    vocals: list = field(default_factory=list)   # [(track, bar, gain, pan)]
    drums: list = field(default_factory=list)    # [(track, bar)]
    harm: list = field(default_factory=list)     # [(track, bar, gain)] bass + other, no drums
    hooks: list = field(default_factory=list)    # [(track, bar, gain, pan)] "other" only, layered on top
    fade: bool = False                           # fade out over this part
    spans: list = field(default_factory=list)    # [(track, start, end)]: a vocal starting in this part
                                                 # is one lyric line, cut exactly at its words
    sweep: bool = False                          # low-pass sweep opening up into this part
    riser: bool = False                          # noise riser over the two bars before this part


def _beat_len(bpm):
    return 60.0 / bpm * audio.SR


def make_mashup(paths, out_path, opts=None, progress=None, names=None):
    """Build the mashup and write it to `out_path`. Returns a text report (also saved next to it)."""
    from . import styles
    opts = opts or Options()
    if len(paths) < 2:
        raise ValueError("Pick at least two songs.")
    style = styles.get(opts.style)
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
        fkey = separate.file_key(p)
        say(0.02 + 0.5 * (i + 0.2) / n, f"Using the uploaded stems: {name}" if separate.has_pack(cache, fkey)
            else f"Separating vocals: {name} ({sep_method})")
        vocals, drums, bass, other = separate.separate(
            y, sep_method, cache, fkey,
            progress=lambda m, f=0.02 + 0.5 * (i + 0.2) / n: say(f, m))
        if lyrics.available():
            say(0.02 + 0.5 * (i + 0.55) / n, f"Reading the lyrics: {name}")
        words = lyrics.transcribe(vocals, cache, f"{fkey}.{sep_method}")
        say(0.02 + 0.5 * (i + 0.7) / n, f"Finding the beat: {name}")
        pitched = other if bass is None else other + bass
        a = an.analyze(y, pitched)
        t = Track(name, p, a, vocals, drums, bass, other)
        t.tuning = an.tuning(pitched)
        t.f0 = an.vocal_pitch(vocals)
        t.words = [(a, b, w) for a, b, w, *_ in words]
        t.line_start = [bool(x[3]) if len(x) > 3 else False for x in words]
        tracks.append(t)
        del y, pitched

    if 0 <= opts.base < n:
        base = opts.base
    else:
        base = 0 if style.base_first else an.choose_base([t.a for t in tracks])
    B = tracks[base]
    # Keep the grid in a danceable range: a 60 BPM ballad is felt (and gridded) at 120.
    base_fold = 2.0 if B.a.bpm < 80 else 0.5 if B.a.bpm > 165 else 1.0
    bpm = B.a.bpm * base_fold
    T = _beat_len(bpm)
    key, minor = _plan_keys(tracks, base, opts, style.key_mode)

    for i, t in enumerate(tracks):
        say(0.55 + 0.27 * i / n, f"Matching {t.name} to {bpm:.1f} BPM, {an.NOTES[key]}{'m' if minor else ''}")
        t.fold = base_fold if i == base else an.tempo_fold(t.a.bpm, bpm)
        _conform(t, T)
        _profile(t, opts.phrase_bars, T)

    say(0.84, f"Arranging ({style.name})")
    parts = style.arrange(styles.Ctx(tracks, base, opts, 4 * T))
    say(0.88, "Mixing")
    out = render(tracks, parts, 4 * T, opts)
    say(0.97, "Saving")
    audio.save(out_path, out)

    report = describe(tracks, base, parts, 4 * T, out.shape[1], sep_method, time.time() - t0, style,
                      (key, minor))
    with open(os.path.splitext(out_path)[0] + ".txt", "w", encoding="utf-8") as f:
        f.write(report)
    say(1.0, f"Done: {out_path}")
    return report


def default_cache_dir():
    return os.environ.get("RAVE_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "rave")


def _plan_keys(tracks, base, opts, mode):
    """Decide every song's transposition. "base": everyone moves to the base song's key. "least_damage"
    (DJ Earworm's spreadsheet): try every key and pick the one that moves the voices least in total,
    counting it worse to push a high voice up or a low voice down. Either way the songs' tuning is lined
    up with the base's, so nothing is a few cents flat against the rest. Returns the key used."""
    B = tracks[base]
    target, minor = B.a.key, B.a.minor
    if opts.match_key and mode == "least_damage":
        best = None
        for k in range(12):
            cost = 0.0
            for i, t in enumerate(tracks):
                s = an.key_shift(t.a.key, t.a.minor, k, B.a.minor)
                cost += an.shift_cost(s, t.f0) + (0.25 * abs(s) if i == base else 0.0)
            if best is None or cost < best[0]:
                best = (cost, k)
        target = best[1]
    for i, t in enumerate(tracks):
        if i in opts.transpose:
            t.shift = int(opts.transpose[i])
        elif opts.match_key and not (i == base and target == B.a.key):
            t.shift = an.key_shift(t.a.key, t.a.minor, target, minor)
        t.tune = 0.0 if i == base else float(np.clip(B.tuning - t.tuning, -0.5, 0.5))
        if abs(t.tune) < 0.05:
            t.tune = 0.0
    return target, minor


def _conform(t, T):
    """Stretch and transpose a track so that its beats fall exactly every T samples. With Rubber Band's
    CLI every beat is pinned to the grid (songs with a drifting tempo get straightened out too); without
    it the song is stretched by its average tempo, which is exact for songs with a fixed tempo."""
    beats, phase = an.fold(t.a.beats, t.a.downbeat_phase, t.fold)
    src = beats * audio.SR
    period = float(np.median(np.diff(src)))
    scale = T / period
    t.rate = 1.0 / scale
    n_in = t.vocals.shape[1]
    semis = t.shift + t.tune
    identity = t.a.steady and abs(scale - 1.0) < 1e-5 and semis == 0
    if identity:
        t.beats = src
        dst = src
    elif transform.can_warp():
        dst = src[0] * scale + np.arange(len(src)) * T
        out_len = int(round(dst[-1] + (n_in - src[-1]) * scale))
        for name in STEMS:
            y = getattr(t, name)
            if y is not None:
                warped = transform.warp(y, np.round(src), np.round(dst), out_len, semis, vocal=name == "vocals")
                # Vocals keep their formants (the tone envelope stays put); instruments move with the pitch.
                setattr(t, name, transform.restore_tone(y, warped, 0.0 if name == "vocals" else semis))
        t.beats = dst
    else:
        for name in STEMS:
            y = getattr(t, name)
            if y is not None:
                setattr(t, name, transform.stretch_shift(y, t.rate, semis, vocal=name == "vocals"))
        t.beats = dst = src * scale
    t.vocals = mix.highpass(t.vocals, 90)
    # The words move with the audio: through the beat map, extended at its ends by the overall stretch.
    if t.words:
        w = np.array([(a, b) for a, b, _ in t.words]) * audio.SR
        mapped = np.interp(w, src, dst)
        mapped = np.where(w < src[0], dst[0] - (src[0] - w) * scale, mapped)
        mapped = np.where(w > src[-1], dst[-1] + (w - src[-1]) * scale, mapped)
        t.words = [(int(a), int(b), word) for (a, b), (_, _, word) in zip(mapped, t.words)]
    bars = np.round(t.beats[phase::4]).astype(np.int64)
    t.bars = bars[(bars >= 0) & (bars + 1 < t.vocals.shape[1])]


def backing(t, drums=True):
    """The track's instrumental (or its pitched part without drums), summed from the stems."""
    y = t.other if t.bass is None else t.other + t.bass
    return y + t.drums if drums and t.drums is not None else y


def _profile(t, phrase_bars, T):
    """Per-bar vocal activity, where the lines start, the phrase grid, the chorus and the hook."""
    bars = t.bars
    nb = t.nbars()
    P = phrase_bars
    level = np.array([audio.rms(t.vocals[:, bars[k]:bars[k + 1]]) for k in range(nb)])
    loud = np.percentile(level, 90) if nb else 0.0
    t.threshold = loud * 0.2
    t.vocal_level, t.active = level, level > t.threshold

    t.env = _envelope(t.vocals)
    t.gate = float(np.percentile(t.env, 95)) * 0.1
    t.quiet = _quiet(t.env, t.gate)
    # With lyrics, a gap is only ever between words: never cut inside one, however soft it gets.
    t.in_word = np.zeros(len(t.env), dtype=bool)
    for a, b, _ in t.words or []:
        t.in_word[max(0, a // ENV_HOP + 2):b // ENV_HOP - 2] = True
    t.quiet &= ~t.in_word
    t.lines = _lines(t, T)
    # How much the singer is singing straight across each bar line (bad places to start or end a turn).
    w = 15
    fr = np.minimum(bars // ENV_HOP, len(t.quiet) - 1)
    t.across = np.array([1.0 - t.quiet[max(0, f - w):f + w].mean() for f in fr])

    # A line starts in a bar whose first beats are sung after a mostly quiet bar. Judging by the first
    # beats keeps a pickup (words sung just before the downbeat) from being taken for the start.
    beat = int(T)
    sung = np.array([t.env[bars[k] // ENV_HOP:(bars[k] + 2 * beat) // ENV_HOP].mean() > 2 * t.gate
                     for k in range(nb)])
    early = np.array([t.env[bars[k] // ENV_HOP:(bars[k] + 3 * beat) // ENV_HOP].mean() > 2 * t.gate
                      for k in range(nb)])
    starts = [k for k in range(1, nb) if sung[k] and not early[k - 1]]
    if nb and sung[0]:
        starts.insert(0, 0)
    t.first_vocal = starts[0] if starts else (int(np.argmax(t.active)) if t.active.any() else 0)

    pitched = backing(t, drums=False)
    t.vchroma = _bar_chroma(t.vocals, bars)
    t.ichroma = _bar_chroma(pitched, bars)
    full = backing(t)
    t.bed_level = np.array([audio.rms(full[:, bars[k]:bars[k + 1]]) for k in range(nb)])
    t.bands = _bar_bands(full, bars)
    t.rhythm = _bar_rhythm(t.drums if t.drums is not None else full, bars)
    del pitched, full

    # Phrases begin where the music changes and where lines start; pick the bar offset of the grid
    # that lines up best with both.
    feat = np.hstack([_unit(t.vchroma), _unit(t.ichroma), _z(np.log(t.bed_level + 1e-6))[:, None],
                      _z(np.log(level + 1e-6))[:, None]])
    nov = _novelty(feat, 4)
    line = np.zeros(nb)
    line[starts] = 1.0
    score = [(_z(nov)[ph::P].mean() if nb > ph else -9) + 2.0 * line[ph::P].mean() +
             0.5 * line[ph::P // 2].mean() for ph in range(P)]
    anchor = int(np.argmax(score))
    t.grid = list(range(anchor, nb - P + 1, P)) or [0]
    frac = {g: t.active[g:g + P].mean() for g in t.grid}
    t.phrases = [g for g in t.grid if g >= t.first_vocal - P // 2 and frac[g] >= 0.5] \
        or [g for g in t.grid if frac[g] >= 0.3] or t.grid[:1]

    # Hook: the 4 instrumental bars that repeat most, loud, ideally with nobody singing over them.
    cf = _unit(t.ichroma)
    hc = [g + o for g in t.grid for o in (0, 4) if g + o + 4 <= nb]
    if len(hc) >= 2:
        hv = np.stack([cf[h:h + 4].ravel() for h in hc])
        hv = hv / (np.linalg.norm(hv, axis=1, keepdims=True) + 1e-9)
        hs = hv @ hv.T
        np.fill_diagonal(hs, 0)
        rep = np.sort(hs, axis=1)[:, -3:].sum(axis=1)
        energy = np.array([t.bed_level[h:h + 4].mean() for h in hc])
        bare = np.array([1.0 - t.active[h:h + 4].mean() for h in hc])
        t.hook = hc[int(np.argmax(_z(rep) + 0.5 * _z(energy) + bare))]
    else:
        t.hook = t.grid[0]

    # Chorus: the phrase that repeats most, is loudest, and is sung throughout.
    cands = [g for g in t.phrases if g + P <= nb]
    t.choruses = set()
    if len(cands) < 2:
        t.chorus = cands[0] if cands else t.phrases[0]
        return
    cf = np.hstack([t.vchroma, t.ichroma])
    cf = cf - cf.mean(axis=0)
    vecs = np.stack([cf[g:g + P].ravel() for g in cands])
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, 0)
    rep = np.sort(sim, axis=1)[:, -2:].sum(axis=1)
    energy = np.array([level[g:g + P].mean() + t.bed_level[g:g + P].mean() for g in cands])
    sung_frac = np.array([frac[g] for g in cands])
    best = int(np.argmax(_z(rep) + _z(energy) + 2 * sung_frac))
    t.chorus = cands[best]
    t.choruses = {cands[j] for j in range(len(cands)) if j == best or sim[best, j] > 0.5}


def _unit(m):
    m = m - m.mean(axis=0)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def _z(v):
    v = np.asarray(v, dtype=np.float64)
    return (v - v.mean()) / (v.std() + 1e-9)


def _novelty(feat, w):
    """How different the `w` bars after each bar line are from the `w` bars before it."""
    n = len(feat)
    out = np.zeros(n)
    for k in range(1, n):
        a, b = feat[max(0, k - w):k].mean(axis=0), feat[k:k + w].mean(axis=0)
        out[k] = np.linalg.norm(a - b)
    return out


def _envelope(y):
    m = np.abs(audio.mono(y))
    n = len(m) // ENV_HOP
    return np.sqrt(np.mean(np.square(m[:n * ENV_HOP].reshape(n, ENV_HOP)), axis=1))


def _bar_chroma(y, bars):
    """Mean chroma of each bar, shape (bars - 1, 12)."""
    import librosa
    sr, hop = 11025, 1024
    m = librosa.resample(audio.mono(y), orig_sr=audio.SR, target_sr=sr)
    chroma = librosa.feature.chroma_stft(y=m, sr=sr, hop_length=hop)
    f = np.minimum(chroma.shape[1] - 1, (bars / audio.SR * sr / hop).astype(int))
    return np.stack([chroma[:, f[k]:max(f[k] + 1, f[k + 1])].mean(axis=1)
                     for k in range(len(bars) - 1)]) if len(bars) > 1 else np.zeros((0, 12))


def _bar_bands(y, bars):
    """Share of each bar's energy below 220 Hz, 220-1760 Hz and above (AutoMashUpper's bands)."""
    import librosa
    sr, hop = 11025, 512
    m = librosa.resample(audio.mono(y), orig_sr=audio.SR, target_sr=sr)
    S = np.abs(librosa.stft(m, n_fft=2048, hop_length=hop)) ** 2
    fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    e = np.stack([S[fr < 220].sum(0), S[(fr >= 220) & (fr < 1760)].sum(0), S[fr >= 1760].sum(0)])
    f = np.minimum(e.shape[1] - 1, (bars / audio.SR * sr / hop).astype(int))
    out = np.stack([e[:, f[k]:max(f[k] + 1, f[k + 1])].mean(axis=1) for k in range(len(bars) - 1)])
    return out / (out.sum(axis=1, keepdims=True) + 1e-12)


def _bar_rhythm(y, bars):
    """Each bar's drum pattern on a 16-step grid, normalized."""
    env = _envelope(y)
    onset = np.maximum(0, np.diff(env, prepend=env[:1]))
    out = np.zeros((len(bars) - 1, 16))
    for k in range(len(bars) - 1):
        edges = np.linspace(bars[k], bars[k + 1], 17) // ENV_HOP
        out[k] = [onset[int(a):max(int(a) + 1, int(b))].sum() for a, b in zip(edges[:-1], edges[1:])]
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


def fit(vt, vs, bt, bs, bars):
    """How well a vocal passage sits on a backing passage: the cosine between the notes being sung and
    the backing's chords, bar by bar, weighted by how much singing there is."""
    if bt.ichroma is None or vs < 0 or bs < 0 or vs + bars > len(vt.vchroma) or bs + bars > len(bt.ichroma):
        return -1.0
    v, b = vt.vchroma[vs:vs + bars], bt.ichroma[bs:bs + bars]
    cos = (v * b).sum(axis=1) / (np.linalg.norm(v, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    w = vt.vocal_level[vs:vs + bars] + 1e-9
    return float((cos * w).sum() / w.sum())


def render(tracks, parts, bar_len, opts):
    total_bars = sum(p.bars for p in parts)
    n_out = int(np.ceil(total_bars * bar_len)) + audio.SR * 2
    bus = {k: np.zeros((2, n_out), dtype=np.float32) for k in ("drums", "bass", "other", "hook", "voc")}

    # Level-match every stem before anything is placed. A song's instrumental stems share one gain so
    # its backing keeps its own balance.
    for t in tracks:
        t.vocals = mix.normalize_rms(t.vocals, -19.0, mask=_active_mask(t))
        level = audio.rms(backing(t))
        g = audio.undb(-18.0 - audio.db(level)) if level > 1e-5 else 1.0   # a pack may have no backing
        for name in ("drums", "bass", "other"):
            if getattr(t, name) is not None:
                setattr(t, name, getattr(t, name) * g)

    # Flatten to per-bar events per lane, then merge into runs that play straight through.
    spans = {}                                     # (track, destination bar) -> line span
    lanes = {k: [] for k in ("drums", "bass", "other", "hook", "voc", "bgv")}
    d = 0
    for p in parts:
        add = lambda lane, i, s, g=1.0, pan=0.0: lanes[lane].extend(
            (i, s + j, d + j, g, pan) for j in range(p.bars))
        for (i, s) in p.bed:
            for lane in ("drums", "bass", "other"):
                add(lane, i, s)
            add("bgv", i, s)                            # the backing song's own vocal, for the gaps
        for (i, s) in p.drums:
            add("bgv", i, s)                            # backing singers over just the beat, too
            if tracks[i].drums is not None:
                add("drums", i, s)
            else:
                add("other", i, s, 0.35)              # no drum stem: a quiet bed stands in
        for (i, s, g) in p.harm:
            add("bass", i, s, g)
            add("other", i, s, g)
        for (i, s, g, pan) in p.hooks:
            add("hook", i, s, g, pan)
        for (i, s, g, pan) in p.vocals:
            add("voc", i, s, g, pan)
        for (i, a, b) in p.spans:
            spans[(i, d)] = (a, b)
        d += p.bars

    pos = lambda bar: int(round(bar * bar_len))
    for lane in ("drums", "bass", "other", "hook"):
        for (i, s, dbar, n, g, pan) in _runs(lanes[lane], tracks):
            t = tracks[i]
            src = {"drums": t.drums, "bass": t.bass, "other": t.other, "hook": t.other}[lane]
            if src is None:
                continue
            if lane == "hook":
                src = _hook_src(t)
            mix.place(bus[lane], src, int(t.bars[s]), pos(dbar), pos(dbar + n) - pos(dbar),
                      mix.ms(30), mix.ms(30), mix.ms(60), mix.ms(60), g, pan)

    # Vocals: each run starts and ends in a gap between words (or, if the singer never pauses, at the
    # softest point between syllables), so no line is chopped mid-word. A pickup sung just before the
    # downbeat comes along. Where one singer hands over to the next, the two never sing at once: the
    # incoming line waits for the outgoing word to finish, or the outgoing singer stops at their last
    # breath before the next comes in, and only if neither works do they crossfade.
    beat = int(bar_len / 4)
    vruns = []
    for (i, s, dbar, n, g, pan) in _runs(lanes["voc"], tracks):
        t = tracks[i]
        a, b = int(t.bars[s]), int(t.bars[s + n])
        sp = spans.get((i, dbar))
        if sp:
            # A lyric line: from just before its first word to the gap after its last.
            pre, fin = a - (int(sp[0]) - mix.ms(20)), mix.ms(20)
            post, fout = int(sp[1]) + mix.ms(30) - b, mix.ms(40)
        else:
            pre, fin = _lead_in(t, a, bar_len)
            post, fout = _tail(t, b, bar_len)
        vruns.append(dict(t=t, a=a, b=b, d=pos(dbar), e=pos(dbar + n), pre=pre, fin=fin, post=post,
                          fout=fout, g=g, pan=pan, solo=g >= 0.99))
    for r in vruns:
        if not r["solo"]:
            continue
        for o in vruns:
            if o is r or not o["solo"] or not (r["d"] - bar_len / 2 < o["e"] <= r["d"]):
                continue
            _hand_over(o, r, beat)
    # Background vocals: the backing song's own voice (in time and in key with the backing, since it's
    # the same song), heard only in the gaps between lead lines and ducking out just before the next
    # lead comes in. The references keep singing through ~84% of the track.
    if opts.bgv and lanes["bgv"]:
        bgv = np.zeros_like(bus["voc"])
        for (i, s, dbar, n, g, pan) in _runs(lanes["bgv"], tracks):
            t = tracks[i]
            mix.place(bgv, t.vocals, int(t.bars[s]), pos(dbar), pos(dbar + n) - pos(dbar),
                      mix.ms(30), mix.ms(30), mix.ms(80), mix.ms(80), float(opts.bgv), 0.0)
        bus["voc"] += mix.gap_fill(bgv, bus["voc"])
    throws = np.zeros_like(bus["voc"])
    for r in vruns:
        t = r["t"]
        mix.place(bus["voc"], t.vocals, r["a"], r["d"], r["e"] - r["d"], r["pre"], r["post"],
                  r["fin"], r["fout"], r["g"], r["pan"])
        if r["solo"]:
            # A delay throw on the line's last word: quarter-note echoes, high-passed, fading, the way a
            # mix engineer hands one singer over to the next.
            end_src = r["b"] + r["post"]
            end_dst = r["e"] + r["post"]
            for k, g in ((1, 0.32), (2, 0.14), (3, 0.06)):
                mix.place(throws, t.vocals, end_src - beat, end_dst - beat + k * beat, beat,
                          0, 0, mix.ms(15), beat // 2, g * r["g"], 0.12 if k % 2 else -0.12)
    bus["voc"] += mix.highpass(throws, 300)
    # A vocal plate: everything the singers do gets a little air.
    bus["voc"] += mix.room(bus["voc"], seconds=1.4, wet_db=-19.0, seed=5)

    # Low-pass sweeps open up into the parts that ask for one.
    d = 0
    for p in parts:
        if p.sweep:
            for lane in ("drums", "bass", "other", "hook"):
                mix.lowpass_sweep(bus[lane], pos(d), pos(d + min(1, p.bars)) - pos(d))
        d += p.bars

    voc = bus["voc"]
    other, hook = mix.duck(bus["other"], voc, 4.0), mix.duck(bus["hook"], voc, 6.0)
    drums = mix.duck(bus["drums"], voc, 1.0)
    if opts.spatial == "8d":
        # 8D: the voices circle the listener's head, hooks circle the other way, the chords sway
        # across the front, and drums and bass stay planted in the middle; a small room around it all.
        n = voc.shape[1]
        voc = mix.binaural(voc, mix.orbit_angle(n, 9.0)) * 1.15
        hook = mix.binaural(hook, mix.orbit_angle(n, 13.0, start=np.pi, direction=-1.0))
        other = 0.5 * other + 0.6 * mix.binaural(other, mix.orbit_angle(n, 20.0, sway=np.pi / 3))
        dry = voc + hook + other
        out = drums + bus["bass"] + dry + mix.room(dry + 0.5 * drums, seconds=2.2, wet_db=-13.0)
    else:
        out = other + hook + bus["bass"] + drums + voc

    # Risers into the parts that ask for one.
    if opts.riser:
        d = 0
        for p in parts:
            if p.riser:
                start, end = pos(max(0, d - 2)), pos(d)
                if end > start:
                    out[:, start:end] += mix.riser(end - start, peak=2.0 * audio.rms(out[:, :end]))
            d += p.bars

    # Outro fade, trim, master.
    end = pos(total_bars) + audio.SR // 2
    out = out[:, :end]
    if parts[-1].fade:
        mix.fade_out(out, pos(parts[-1].bars) + audio.SR // 2)
    if opts.nightcore and opts.nightcore > 1.0:
        out = mix.nightcore(out, opts.nightcore)
    if opts.spatial != "8d":
        # The references sit a little narrower than a raw stem mix (L/R correlation ~0.77).
        mid, side = (out[0] + out[1]) / 2, (out[0] - out[1]) / 2
        out = np.stack([mid + 0.75 * side, mid - 0.75 * side]).astype(np.float32)
    out = mix.normalize_rms(out, -13.5)
    out = mix.limit(out, -0.3)   # the references peak near full scale, ~14 dB above their RMS
    return mix.fade_out(out, mix.ms(10))


def _hook_src(t):
    """A layered hook keeps out of the way: no bass (one bass line at a time) and a little less body."""
    if getattr(t, "_hook", None) is None:
        t._hook = mix.highpass(t.other, 200)
    return t._hook


def _quiet(env, gate):
    """Frames where the singer is between words: below the absolute gate, or well below what they've
    been singing in the surrounding second (reverb and separation bleed rarely reach true silence)."""
    from scipy import ndimage
    local = ndimage.maximum_filter1d(env, 100)
    return env < np.maximum(gate, 0.18 * local)


def _gap_end_before(t, f, lo):
    """The frame where the latest gap (>= GAP quiet frames) at or before frame f ends, searching back
    to lo; None if there isn't one."""
    run = 0
    for k in range(min(f, len(t.quiet)) - 1, max(0, lo) - 1, -1):
        run = run + 1 if t.quiet[k] else 0
        if run >= GAP:
            end = k + run
            while end < f and t.quiet[end]:
                end += 1
            return end
    return None


def _gap_start_after(t, f, hi):
    """The first frame of the first gap (>= GAP quiet frames) at or after frame f, up to hi; None if
    there isn't one."""
    run = 0
    for k in range(max(0, f), min(hi, len(t.quiet))):
        run = run + 1 if t.quiet[k] else 0
        if run >= GAP:
            return k - run + 1
    return None


def _softest(t, lo, hi):
    """The quietest frame in [lo, hi) that isn't inside a word: the best place to cut when a singer
    never pauses."""
    lo, hi = max(0, lo), min(len(t.env), hi)
    if hi <= lo:
        return lo
    e = t.env[lo:hi].astype(np.float64)
    inside = getattr(t, "in_word", None)
    if inside is not None and not inside[lo:hi].all():
        e = np.where(inside[lo:hi], np.inf, e)
    return lo + int(np.argmin(e))


def _lead_in(t, a, bar_len):
    """How far before sample `a` to start a vocal, and the fade. A pickup into the line comes along:
    the start is the end of the last gap within the bar before. If the singer is already quiet at `a`
    there's nothing to bring along."""
    f = a // ENV_HOP
    if f >= GAP and t.quiet[f - GAP:f].all():
        return mix.ms(10), mix.ms(10)
    g = _gap_end_before(t, f, f - int(bar_len // ENV_HOP))
    if g is not None:
        return (f - g) * ENV_HOP + mix.ms(25), mix.ms(25)
    m = _softest(t, f - int(bar_len / 8 // ENV_HOP), f + 1)    # between syllables, just before
    return (f - m) * ENV_HOP, mix.ms(30)


def _tail(t, b, bar_len):
    """How far past sample `b` to let a vocal ring so its last word isn't clipped, and the fade. Ends
    in the first gap within half a bar, or failing that between syllables."""
    f = b // ENV_HOP
    g = _gap_start_after(t, f, f + int(bar_len / 2 // ENV_HOP))
    if g is not None:
        return (g - f) * ENV_HOP + mix.ms(30), mix.ms(30)
    m = _softest(t, f, f + int(bar_len / 4 // ENV_HOP))
    return (m - f) * ENV_HOP + mix.ms(15), mix.ms(30)


def _hand_over(o, r, beat):
    """Make sure outgoing vocal run `o` has stopped before incoming run `r` starts singing."""
    ot, rt = o["t"], r["t"]
    to_src_o = lambda x: o["a"] + (x - o["d"])          # destination sample -> source sample
    to_src_r = lambda x: r["a"] + (x - r["d"])
    end = o["e"] + o["post"]                             # where the outgoing voice stops
    start = r["d"] - r["pre"]                            # where the incoming voice starts
    if end <= start + mix.ms(20):
        return
    # 1. Let the outgoing word finish: start the incoming line at its first gap after it, as long as
    #    that doesn't throw away more than its pickup and a beat of the line.
    lo = to_src_r(end) // ENV_HOP
    hi = to_src_r(r["d"] + beat) // ENV_HOP
    run, g = 0, None
    for k in range(max(0, lo), min(hi, len(rt.quiet))):
        run = run + 1 if rt.quiet[k] else 0
        if run >= GAP:
            k2 = k
            while k2 + 1 < len(rt.quiet) and rt.quiet[k2 + 1]:
                k2 += 1
            g = k2 + 1                                   # the next word begins here
            break
    if g is not None:
        r["pre"] = r["a"] - (g * ENV_HOP - mix.ms(25))
        r["fin"] = mix.ms(25)
        return
    # 2. Or stop the outgoing singer at their last gap before the incoming one starts.
    f = to_src_o(start) // ENV_HOP
    run = 0
    for k in range(min(f, len(ot.quiet)) - 1, max(0, to_src_o(o["e"] - beat) // ENV_HOP) - 1, -1):
        run = run + 1 if ot.quiet[k] else 0
        if run >= GAP:
            o["post"] = (k * ENV_HOP + mix.ms(30)) - o["b"]
            o["fout"] = mix.ms(30)
            return
    # 3. Neither: crossfade where the outgoing singer is softest around the incoming start.
    m = _softest(ot, f - int(beat / 2 // ENV_HOP), f + int(beat / 2 // ENV_HOP)) * ENV_HOP
    o["post"] = m + mix.ms(40) - o["b"]
    o["fout"] = mix.ms(80)
    r["fin"] = max(r["fin"], mix.ms(80))


def _active_mask(t):
    m = np.zeros(t.vocals.shape[1], dtype=bool)
    for k, on in enumerate(t.active):
        if on:
            m[t.bars[k]:t.bars[k + 1]] = True
    return m


def _runs(evts, tracks):
    """Group (track, src_bar, dst_bar, gain, pan) events into runs that continue seamlessly, as
    (track, src_bar, dst_bar, bars, gain, pan). Events past the end of a track are dropped."""
    evts = [e for e in evts if 0 <= e[1] and e[1] + 1 < len(tracks[e[0]].bars)]
    evts = sorted(evts, key=lambda e: (e[0], e[3], e[4], e[2]))
    runs = []
    for e in evts:
        r = runs[-1] if runs else None
        if r and r[-1][0] == e[0] and r[-1][3] == e[3] and r[-1][4] == e[4] \
                and r[-1][1] + 1 == e[1] and r[-1][2] + 1 == e[2]:
            r.append(e)
        elif r and r[-1][:5] == e[:5]:
            continue                                   # the same bar asked for twice
        else:
            runs.append([e])
    return [(r[0][0], r[0][1], r[0][2], len(r), r[0][3], r[0][4]) for r in runs]


def _lines(t, T):
    """The song's lyric lines on the bar grid: dicts with the words, their text, and the bars a turn
    singing the line covers, `b0` to `b1` (exclusive). A line whose first word comes in the last beat
    of a bar is a pickup into the next bar (the vocal lead-in brings it along); one that ends early in
    a bar ends at that bar line (the tail lets the last word ring). Long lines are split at their
    biggest pause so a turn is at most 4 bars."""
    if not t.words:
        return []
    groups, cur = [], []
    for k, (a, b, w) in enumerate(t.words):
        if cur and (t.line_start[k] if t.line_start else False):
            groups.append(cur)
            cur = []
        cur.append(k)
    if cur:
        groups.append(cur)
    # A fragment of a word or two ("i wanna" | "run away") belongs with its neighbour.
    merged = []
    for g in groups:
        if merged and (len(g) < 3 or len(merged[-1]) < 3) and                 t.words[g[0]][0] - t.words[merged[-1][-1]][1] < 1.0 * audio.SR:
            merged[-1] = merged[-1] + g
        else:
            merged.append(g)
    groups = merged
    bar_len = 4 * T

    def split(g):
        span = t.words[g[-1]][1] - t.words[g[0]][0]
        if span <= 4 * bar_len or len(g) < 4:
            return [g]
        gaps = [t.words[g[i + 1]][0] - t.words[g[i]][1] for i in range(len(g) - 1)]
        cut = int(np.argmax(gaps[1:-1])) + 2
        return split(g[:cut]) + split(g[cut:])

    out = []
    bars = t.bars
    for g in (x for grp in groups for x in split(grp)):
        start, end = t.words[g[0]][0], t.words[g[-1]][1]
        b0 = int(np.searchsorted(bars, start, side="right")) - 1
        if b0 < 0:
            continue
        if b0 + 1 < len(bars) and bars[b0 + 1] - start < T:
            b0 += 1                                      # sung as a pickup into the next bar
        b1 = int(np.searchsorted(bars, end - T / 2, side="left"))
        b1 = max(b0 + 1, b1)
        if b1 >= len(bars) - 1:
            continue
        ws = [t.words[k][2] for k in g]
        # Where to cut is heard, not read off the word times: the gap in the vocal just before the
        # first word and just after the last. A line with both is clean: it lifts out without clipping
        # a word of its own or of its neighbours.
        # Cut at the softest point between this line's first word and the word before it, and
        # between its last word and the next: never inside a word. `edge` is how loud the vocal is at
        # those cuts (0 = a real pause), so the planner can prefer lines that lift out cleanly.
        prev_end = t.words[g[0] - 1][1] if g[0] > 0 else start - 0.3 * audio.SR
        next_start = t.words[g[-1] + 1][0] if g[-1] + 1 < len(t.words) else end + 0.4 * audio.SR
        lo = int(max(prev_end, start - 0.3 * audio.SR)) // ENV_HOP
        hi = int(min(next_start, end + 0.4 * audio.SR)) // ENV_HOP
        fs, fe = int(start) // ENV_HOP, int(end) // ENV_HOP
        cs = _softest(t, lo, fs + 1) if lo < fs else fs
        ce = _softest(t, fe, hi + 1) if hi > fe else fe
        peak = float(np.percentile(t.env, 95)) + 1e-9
        edge = float(t.env[min(cs, len(t.env) - 1)] + t.env[min(ce, len(t.env) - 1)]) / peak
        if (end - start) < 0.4 * audio.SR:
            continue
        out.append(dict(b0=b0, b1=b1, start=cs * ENV_HOP, end=ce * ENV_HOP, words=ws, text=" ".join(ws),
                        edge=edge, clean=edge < 0.4))
    return out


def words_between(t, a, b):
    """The words sung between samples a and b of a conformed track."""
    return [w for s, e, w in (t.words or []) if a <= (s + e) / 2 < b]


def sung_words(tracks, p, bar_len, limit=8):
    """The first few words of a part's lead vocal, for the cue sheet: the lyric line itself where the
    part starts one, else the words in its bars."""
    if p.spans:
        i, a, b = p.spans[0]
        ws = words_between(tracks[i], a - 1, b + 1)
        return " ".join(ws[:limit + 4]) + (" ..." if len(ws) > limit + 4 else "")
    if any(v[2] >= 0.99 for v in p.vocals) and not p.label.startswith(("Chorus", "Final", "A cappella")):
        return ""                                      # a line carried on from the part before
    lead = [v for v in p.vocals if v[2] >= 0.99] or p.vocals[:1]
    if not lead:
        return ""
    i, s = lead[0][0], lead[0][1]
    t = tracks[i]
    if not t.words or s < 0 or s + p.bars >= len(t.bars):
        return ""
    ws = words_between(t, t.bars[s] - bar_len / 8, t.bars[s + p.bars] - bar_len / 8)
    return " ".join(ws[:limit]) + (" ..." if len(ws) > limit else "")


def describe(tracks, base, parts, bar_len, n_samples, sep_method, secs, style, key):
    bpm = 4 * 60.0 * audio.SR / bar_len
    lines = ["Rave cue sheet", "",
             f"Engine: {style.name}",
             f"Tempo {bpm:.1f} BPM, key {an.NOTES[key[0]]}{'m' if key[1] else ''} "
             f"({an.camelot(*key)}), length {n_samples / audio.SR:.0f}s",
             f"Separation: {sep_method}; beats: {tracks[base].a.tracker}; "
             f"stretch/shift: {'rubberband beat map' if transform.can_warp() else transform.backend()}; "
             f"built in {secs:.0f}s", "",
             "Songs:"]
    for i, t in enumerate(tracks):
        tag = " [base]" if i == base else ""
        fold = {2.0: " (double time)", 0.5: " (half time)"}.get(t.fold, "")
        grid = "" if t.a.steady else ", tempo drifts (beat-mapped)"
        tune = f" {t.tune:+.2f} tuning" if t.tune else ""
        lines.append(f"  {i + 1}. {t.name}{tag}: {t.a.bpm:.1f} BPM {t.a.key_name} ({t.a.camelot})"
                     f" -> x{t.rate:.3f}{fold}, {t.shift:+d} st{tune}{grid}; chorus at bar {t.chorus + 1}")
    lines += ["", "Arrangement:"]
    d = 0
    for p in parts:
        sec = d * bar_len / audio.SR
        sung = sung_words(tracks, p, bar_len)
        lines.append(f"  {int(sec // 60)}:{sec % 60:04.1f}  {p.bars:2d} bars  {p.label}" +
                     (f'  "{sung}"' if sung else ""))
        d += p.bars
    return "\n".join(lines) + "\n"
