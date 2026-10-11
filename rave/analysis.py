"""Musical analysis: tempo, beat grid, downbeats, key, and the maths for matching songs to each other."""
from dataclasses import dataclass, field

import numpy as np

from . import audio

NOTES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# Krumhansl-Kessler key profiles.
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

ASR = 22050  # analysis sample rate
HOP = 512


@dataclass
class Analysis:
    bpm: float
    beats: np.ndarray          # beat times in seconds, cleaned onto a grid (see clean_grid)
    key: int                   # tonic, 0 = C
    minor: bool
    key_confidence: float
    duration: float
    downbeat_phase: int = 0    # beats[downbeat_phase::4] are the downbeats
    steady: bool = True        # constant tempo: the grid is a straight line, not the tracked beats
    tracker: str = "librosa"

    @property
    def key_name(self):
        return NOTES[self.key] + ("m" if self.minor else "")

    @property
    def camelot(self):
        return camelot(self.key, self.minor)

    @property
    def downbeats(self):
        return self.beats[self.downbeat_phase::4]


def camelot(key, minor):
    """Camelot wheel code, e.g. 8B = C major, 8A = A minor."""
    major_tonic = (key + 3) % 12 if minor else key
    return f"{(major_tonic * 7 + 7) % 12 + 1}{'A' if minor else 'B'}"


def analyze(mix, inst=None):
    """Analyze a song. `mix` is the full track (beats come from it); `inst`, if given, is used for key."""
    import librosa

    y = librosa.resample(audio.mono(mix), orig_sr=audio.SR, target_sr=ASR)
    duration = mix.shape[1] / audio.SR
    tracked = _beat_this(y)
    if tracked is not None:
        beats, downbeats = tracked
        tracker = "beat_this"
    else:
        beats, downbeats = _librosa_beats(y), None
        tracker = "librosa"
    if len(beats) < 16:
        raise RuntimeError("Couldn't find a steady beat in this song.")
    beats, steady = clean_grid(beats, duration)
    if downbeats is not None and len(downbeats) >= 4:
        phase = downbeat_phase_from(beats, downbeats)
    else:
        phase = downbeat_phase(_beat_strength(y, beats))
    bpm = 60.0 / float(np.median(np.diff(beats)))

    harm = y if inst is None else librosa.resample(audio.mono(inst), orig_sr=audio.SR, target_sr=ASR)
    chroma = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(harm, margin=2), sr=ASR,
                                        hop_length=HOP)
    key, minor, conf = detect_key(chroma.mean(axis=1))
    return Analysis(bpm=bpm, beats=beats, key=key, minor=minor, key_confidence=conf,
                    duration=duration, downbeat_phase=phase, steady=steady, tracker=tracker)


_beat_model = None


def _beat_this(y):
    """Beats and downbeats (seconds) from the beat_this network, or None when it isn't installed."""
    global _beat_model
    try:
        if _beat_model is None:
            import torch
            from beat_this.inference import Audio2Beats
            _beat_model = Audio2Beats(checkpoint_path="final0",
                                      device="cuda" if torch.cuda.is_available() else "cpu", dbn=False)
        beats, downbeats = _beat_model(y, ASR)
    except ImportError:
        return None
    return np.asarray(beats, dtype=np.float64), np.asarray(downbeats, dtype=np.float64)


def _librosa_beats(y):
    import librosa
    onset = librosa.onset.onset_strength(y=y, sr=ASR, hop_length=HOP, aggregate=np.median)
    _, frames = librosa.beat.beat_track(onset_envelope=onset, sr=ASR, hop_length=HOP,
                                        start_bpm=110, tightness=120, trim=False)
    return librosa.frames_to_time(frames, sr=ASR, hop_length=HOP)


def clean_grid(beats, duration, steady_ms=15.0):
    """Turn tracked beats into a gap-free grid. Most produced music has a fixed tempo; then the grid is
    the straight line through the beats, extended over the whole song, which irons out tracker jitter
    and bridges breakdowns where it lost the beat. Otherwise the tracked beats are kept, with dropped
    beats filled in and doubled ones removed. Returns (beats, steady)."""
    beats = np.asarray(beats, dtype=np.float64)
    period = float(np.median(np.diff(beats)))
    keep = [beats[0]]
    for b in beats[1:]:
        if b - keep[-1] >= 0.5 * period:
            keep.append(b)
    beats = np.array(keep)
    d = np.diff(beats)
    k = np.r_[0, np.cumsum(np.maximum(1, np.round(d / period)))]

    # Robust straight-line fit: refit without outliers a few times.
    use = np.ones(len(beats), dtype=bool)
    for _ in range(3):
        slope, icpt = np.polyfit(k[use], beats[use], 1)
        res = beats - (icpt + slope * k)
        mad = np.median(np.abs(res[use])) + 1e-6
        use = np.abs(res) < max(4 * mad, 0.02)
    res = np.abs(beats - (icpt + slope * k))
    steady = np.median(res) * 1000 < steady_ms and np.percentile(res, 90) * 1000 < 3 * steady_ms
    if steady:
        k0 = int(np.ceil(-icpt / slope))
        k1 = int(np.floor((duration - icpt) / slope))
        return icpt + slope * np.arange(k0, k1 + 1), True

    filled = [beats[0]]
    for a, b, m in zip(beats[:-1], beats[1:], np.maximum(1, np.round(d / period)).astype(int)):
        filled.extend(a + (b - a) * np.arange(1, m + 1) / m)
    return np.array(filled), False


def downbeat_phase_from(beats, downbeats):
    """Which of the 4 beat phases the tracker's downbeats fall on (majority vote)."""
    period = float(np.median(np.diff(beats)))
    idx = np.searchsorted(beats, downbeats)
    votes = np.zeros(4)
    for t, i in zip(downbeats, idx):
        j = min((c for c in (i - 1, i) if 0 <= c < len(beats)), key=lambda c: abs(beats[c] - t),
                default=None)
        if j is not None and abs(beats[j] - t) < 0.25 * period:
            votes[j % 4] += 1
    return int(np.argmax(votes))


def _beat_strength(y, beats):
    """Downbeat evidence at each beat: harmonic change, bass energy and onset strength."""
    import librosa
    onset = librosa.onset.onset_strength(y=y, sr=ASR, hop_length=HOP, aggregate=np.median)
    chroma_full = librosa.feature.chroma_cqt(y=y, sr=ASR, hop_length=HOP)
    novelty = np.r_[0, np.linalg.norm(np.diff(chroma_full, axis=1), axis=0)]
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=ASR, n_fft=2048)
    bass = S[freqs < 150].sum(axis=0)
    feats = [_norm(novelty), _norm(bass), _norm(onset[:S.shape[1]])]
    n = min(len(f) for f in feats)
    frames = librosa.time_to_frames(beats, sr=ASR, hop_length=HOP)
    strength = np.zeros(len(beats))
    win = 3
    for i, f in enumerate(frames):
        lo, hi = max(0, f - win), min(n, f + win + 1)
        if lo < hi:
            strength[i] = sum(float(ft[lo:hi].max()) for ft in feats)
    return strength


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-9)


def downbeat_phase(strength):
    """Which of the 4 beat phases starts the bar (assumes 4/4)."""
    return int(np.argmax([strength[p::4].mean() if len(strength[p::4]) else -1e9 for p in range(4)]))


def detect_key(chroma_mean):
    best = (-2.0, 0, False)
    scores = []
    for minor, prof in ((False, MAJOR), (True, MINOR)):
        for k in range(12):
            r = float(np.corrcoef(np.roll(prof, k), chroma_mean)[0, 1])
            scores.append(r)
            if r > best[0]:
                best = (r, k, minor)
    scores.sort(reverse=True)
    return best[1], best[2], float(scores[0] - scores[1])


def tempo_fold(src_bpm, target_bpm):
    """Half/double-time factor (0.5, 1 or 2) that makes `src_bpm` closest to `target_bpm`."""
    return min((0.5, 1.0, 2.0), key=lambda f: abs(np.log(target_bpm / (src_bpm * f))))


def fold(beats, phase, factor):
    """Re-grid for half/double time and return (beats, downbeat_phase). Double time inserts the
    off-beats; half time keeps every other beat, starting on a downbeat."""
    if factor == 2.0:
        nb = np.empty(2 * len(beats) - 1)
        nb[0::2], nb[1::2] = beats, (beats[:-1] + beats[1:]) / 2
        return nb, (2 * phase) % 4
    if factor == 0.5:
        start = phase % 2
        return beats[start::2], ((phase - start) // 2) % 4
    return beats, phase


def wrap_semitones(d):
    """Wrap to the nearest equivalent shift, -6..+5."""
    return int((d + 6) % 12 - 6)


def key_shift(src_key, src_minor, dst_key, dst_minor):
    """Semitones to transpose a song so it sits in the target key. A minor key and its relative major
    share the same notes, so those are matched rather than forcing major onto minor."""
    s = src_key + (3 if src_minor else 0)   # relative major
    d = dst_key + (3 if dst_minor else 0)
    return wrap_semitones(d - s)


def choose_base(analyses):
    """The song the others will be stretched to: the one that needs the least total stretching."""
    def cost(i):
        t = analyses[i].bpm
        return sum(abs(np.log(t / (a.bpm * tempo_fold(a.bpm, t)))) for a in analyses)
    return min(range(len(analyses)), key=cost)


def tuning(y):
    """How far the song's tuning is off A440, in semitones (-0.5 .. 0.5)."""
    import librosa
    m = librosa.resample(audio.mono(y), orig_sr=audio.SR, target_sr=ASR)
    return float(librosa.estimate_tuning(y=m, sr=ASR, bins_per_octave=12))


def vocal_pitch(vocals):
    """Median sung pitch in Hz (0 if there's too little singing to tell). Used to keep a high voice from
    being pushed up, or a low one down."""
    import librosa
    sr = 11025
    m = librosa.resample(audio.mono(vocals), orig_sr=audio.SR, target_sr=sr)
    f0 = librosa.yin(m, fmin=70, fmax=1000, sr=sr, frame_length=1024, hop_length=512)
    rms = librosa.feature.rms(y=m, frame_length=1024, hop_length=512)[0][:len(f0)]
    sung = f0[rms > 0.25 * np.percentile(rms, 95)]
    return float(np.median(sung)) if len(sung) > 50 else 0.0


def shift_cost(semitones, f0):
    """How much a transposition hurts a voice: every semitone counts, and pushing a high voice up or a
    low voice down counts double (AutoMashUpper's listening tests; DJ Earworm's "least damage")."""
    s = semitones
    cost = abs(s)
    if f0 > 250 and s > 0:
        cost += abs(s)
    if 0 < f0 < 170 and s < 0:
        cost += abs(s)
    return cost
