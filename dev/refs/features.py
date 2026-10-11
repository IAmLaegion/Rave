"""Per-beat features of a song or mashup, for matching a mashup back to its sources.

Everything is beat-synchronous, so songs at different tempos line up beat for beat, and chroma can be
rotated to test every transposition. Stems come from the same Demucs cache the engine uses. Results
are cached in /data/cache/feat/<file key>.npz.
"""
import os

import librosa
import numpy as np

from rave import analysis as an
from rave import audio, separate

SR = 22050
HOP = 512
FEAT_DIR = "/data/cache/feat"


def _beat_chroma(y, beats):
    c = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    f = librosa.time_to_frames(beats, sr=SR, hop_length=HOP)
    f = np.clip(f, 0, c.shape[1] - 1)
    out = np.stack([c[:, a:max(a + 1, b)].mean(axis=1) for a, b in zip(f[:-1], f[1:])])
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


def _beat_rhythm(y, beats, steps=4):
    """Onset strength at `steps` points per beat (16th notes)."""
    o = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    t = librosa.frames_to_time(np.arange(len(o)), sr=SR, hop_length=HOP)
    grid = np.concatenate([np.linspace(a, b, steps, endpoint=False) for a, b in zip(beats[:-1], beats[1:])])
    return np.interp(grid, t, o).reshape(-1, steps)


def _beat_rms(y, beats):
    s = (beats * SR).astype(int)
    return np.array([np.sqrt(np.mean(y[a:max(a + 1, b)] ** 2)) for a, b in zip(s[:-1], s[1:])])


def features(path, progress=print):
    key = separate.file_key(path)
    os.makedirs(FEAT_DIR, exist_ok=True)
    cache = os.path.join(FEAT_DIR, f"{key}.npz")
    if os.path.isfile(cache):
        return dict(np.load(cache, allow_pickle=True))
    y = audio.load(path)
    progress(f"  separating {os.path.basename(path)}")
    method = "demucs" if separate.has_pack("/data/cache", key) or os.path.isfile(
        f"/data/cache/{key}.demucs.vocals.flac") else os.environ.get("RAVE_SEP", "demucs")
    v, d, b, o = separate.separate(y, method, "/data/cache", key)
    a = an.analyze(y, o + b)
    beats = a.beats
    mono = lambda s: librosa.resample(audio.mono(s), orig_sr=audio.SR, target_sr=SR)
    yv, yh, yd = mono(v), mono(o + b), mono(d)
    f = dict(
        beats=beats, bpm=a.bpm, key=a.key, minor=a.minor, phase=a.downbeat_phase, steady=a.steady,
        duration=y.shape[1] / audio.SR,
        chroma_v=_beat_chroma(yv, beats), chroma_h=_beat_chroma(yh, beats),
        rhythm_v=_beat_rhythm(yv, beats), rhythm_d=_beat_rhythm(yd, beats),
        rhythm_h=_beat_rhythm(yh, beats),
        rms_v=_beat_rms(yv, beats), rms_h=_beat_rms(yh, beats), rms_d=_beat_rms(yd, beats),
    )
    np.savez(cache, **f)
    return f
