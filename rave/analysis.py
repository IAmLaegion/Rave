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
    beats: np.ndarray          # beat times, seconds
    key: int                   # tonic, 0 = C
    minor: bool
    key_confidence: float
    duration: float
    beat_strength: np.ndarray = field(repr=False, default=None)   # per-beat downbeat evidence
    downbeat_phase: int = 0

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
    onset = librosa.onset.onset_strength(y=y, sr=ASR, hop_length=HOP, aggregate=np.median)
    tempo, frames = librosa.beat.beat_track(onset_envelope=onset, sr=ASR, hop_length=HOP,
                                            start_bpm=110, tightness=120, trim=False)
    beats = librosa.frames_to_time(frames, sr=ASR, hop_length=HOP)
    if len(beats) < 16:
        raise RuntimeError("Couldn't find a steady beat in this song.")
    bpm = 60.0 / float(np.median(np.diff(beats)))

    harm = y if inst is None else librosa.resample(audio.mono(inst), orig_sr=audio.SR, target_sr=ASR)
    chroma = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(harm, margin=2), sr=ASR,
                                        hop_length=HOP)
    key, minor, conf = detect_key(chroma.mean(axis=1))

    # Downbeat evidence at each beat: harmonic change, bass energy and onset strength.
    chroma_full = librosa.feature.chroma_cqt(y=y, sr=ASR, hop_length=HOP)
    novelty = np.r_[0, np.linalg.norm(np.diff(chroma_full, axis=1), axis=0)]
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=ASR, n_fft=2048)
    bass = S[freqs < 150].sum(axis=0)
    feats = [_norm(novelty), _norm(bass), _norm(onset[:S.shape[1]])]
    n = min(len(f) for f in feats)
    strength = np.zeros(len(beats))
    win = 3
    for i, f in enumerate(frames):
        lo, hi = max(0, f - win), min(n, f + win + 1)
        if lo < hi:
            strength[i] = sum(float(ft[lo:hi].max()) for ft in feats)
    a = Analysis(bpm=bpm, beats=beats, key=key, minor=minor, key_confidence=conf,
                 duration=mix.shape[1] / audio.SR, beat_strength=strength)
    a.downbeat_phase = downbeat_phase(strength)
    return a


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


def fold(a, factor):
    """Re-grid an analysis for half/double time (double inserts off-beats, half keeps every other
    beat, keeping the downbeats) and return (beats, beat_strength, downbeat_phase)."""
    beats, strength = a.beats, a.beat_strength
    if factor == 2.0:
        nb = np.empty(2 * len(beats) - 1)
        nb[0::2], nb[1::2] = beats, (beats[:-1] + beats[1:]) / 2
        ns = np.full(len(nb), strength.min())
        ns[0::2] = strength
        beats, strength = nb, ns
    elif factor == 0.5:
        start = a.downbeat_phase % 2
        beats, strength = beats[start::2], strength[start::2]
    return beats, strength, downbeat_phase(strength)


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
