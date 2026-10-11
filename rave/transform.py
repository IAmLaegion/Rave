"""Time-stretching and pitch-shifting.

Rubber Band (through ffmpeg's `rubberband` filter) is used when the ffmpeg build has it; it sounds far
better on vocals and keeps formants, so shifted voices don't go chipmunk. Otherwise librosa's phase
vocoder is the fallback.
"""
import os
import tempfile

import numpy as np
import soundfile as sf

from . import audio


def backend():
    return "rubberband" if audio.has_filter("rubberband") else "librosa"


def stretch_shift(y, rate=1.0, semitones=0.0, vocal=False, sr=audio.SR):
    """Speed `y` (2, n) up by `rate` (tempo factor, >1 is faster) and transpose by `semitones`."""
    if abs(rate - 1.0) < 1e-4 and abs(semitones) < 1e-3:
        return y
    if backend() == "rubberband":
        return _rubberband(y, rate, semitones, vocal, sr)
    return _librosa(y, rate, semitones, sr)


def _rubberband(y, rate, semitones, vocal, sr):
    opts = [f"tempo={rate:.6f}", f"pitch={2.0 ** (semitones / 12.0):.6f}",
            "pitchq=quality", "channels=together"]
    if vocal:
        opts += ["formant=preserved", "transients=smooth"]
    else:
        opts += ["transients=crisp", "detector=percussive"]
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.wav")
        sf.write(src, y.T, sr, subtype="FLOAT")
        raw = audio.run_ffmpeg(["-i", src, "-af", "rubberband=" + ":".join(opts),
                                "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr), "-"])
    return np.ascontiguousarray(np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T)


def _librosa(y, rate, semitones, sr):
    import librosa
    out = y
    if abs(rate - 1.0) >= 1e-4:
        out = librosa.effects.time_stretch(out, rate=rate)
    if abs(semitones) >= 1e-3:
        out = librosa.effects.pitch_shift(out, sr=sr, n_steps=semitones)
    return np.ascontiguousarray(out.astype(np.float32))
