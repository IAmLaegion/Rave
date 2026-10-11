"""Time-stretching and pitch-shifting.

Rubber Band (through ffmpeg's `rubberband` filter) is used when the ffmpeg build has it; it sounds far
better on vocals and keeps formants, so shifted voices don't go chipmunk. Otherwise librosa's phase
vocoder is the fallback.
"""
import os
import shutil
import subprocess
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


def can_warp():
    """Rubber Band's command-line tool, which takes a beat map (ffmpeg's filter can't)."""
    return shutil.which("rubberband") is not None


def warp(y, src, dst, out_len, semitones=0.0, vocal=False, sr=audio.SR):
    """Time-warp `y` (2, n) so that sample src[i] lands on sample dst[i], and transpose it. Used to put
    every beat of a song exactly on the mashup's grid, so the songs stay locked bar after bar."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    ok = (src > 0) & (src < y.shape[1]) & (dst > 0) & (dst < out_len)
    src, dst = src[ok], dst[ok]
    keep = np.r_[True, (np.diff(src) > 0) & (np.diff(dst) > 0)]
    src, dst = src[keep], dst[keep]
    args = ["rubberband", "-q", "--fine", "-D", f"{out_len / sr:.6f}",
            "-p", f"{semitones:.4f}", "--timemap", None]
    if vocal:
        args.append("--formant")
    with tempfile.TemporaryDirectory() as tmp:
        inp, out, tm = (os.path.join(tmp, f) for f in ("in.wav", "out.wav", "map.txt"))
        sf.write(inp, y.T, sr, subtype="FLOAT")
        with open(tm, "w") as f:
            f.writelines(f"{a} {b}\n" for a, b in zip(src, dst))
        args[args.index(None)] = tm
        p = subprocess.run(args + [inp, out], capture_output=True)
        if p.returncode != 0:
            raise RuntimeError("rubberband failed: " + p.stderr.decode(errors="replace").strip())
        z = sf.read(out, dtype="float32", always_2d=True)[0].T
    if z.shape[1] < out_len:
        z = np.pad(z, ((0, 0), (0, out_len - z.shape[1])))
    return np.ascontiguousarray(z[:, :out_len])


def _librosa(y, rate, semitones, sr):
    import librosa
    out = y
    if abs(rate - 1.0) >= 1e-4:
        out = librosa.effects.time_stretch(out, rate=rate)
    if abs(semitones) >= 1e-3:
        out = librosa.effects.pitch_shift(out, sr=sr, n_steps=semitones)
    return np.ascontiguousarray(out.astype(np.float32))


def restore_tone(original, processed, semitones=0.0, max_db=6.0, taps=2047, sr=audio.SR):
    """Give a stretched/shifted stem back the tone it had. Rubber Band smears the 5-10 kHz presence of a
    voice: everything above ~700 Hz comes out 1-2 dB down against the low mids (measured), which is the
    slightly muffled "processed" sound. Compare the long-term spectrum before and after (1/3-octave
    smoothed, relative to 150-600 Hz; for an instrument the original's spectrum is moved by the pitch
    shift first, while a voice keeps its formants) and correct from ~700 Hz up with a linear-phase EQ,
    limited to +-`max_db`."""
    from scipy import signal
    seg = lambda y: audio.mono(y)[: sr * 120]
    f, po = signal.welch(seg(original), sr, nperseg=4096)
    _, pp = signal.welch(seg(processed), sr, nperseg=4096)
    ratio = 2.0 ** (semitones / 12.0)
    po = np.interp(f, f * ratio, po, right=po[-1])         # where the original's energy lands after the shift
    smooth = lambda p: np.array([p[(f >= x / 2 ** (1 / 6)) & (f <= x * 2 ** (1 / 6))].mean()
                                 if x > 0 else p[0] for x in f])
    gain_db = 10 * np.log10((smooth(po) + 1e-20) / (smooth(pp) + 1e-20))
    # Only the change in tone, not in overall level: measure against the body of the sound.
    body = (f >= 150) & (f <= 600)
    gain_db -= np.median(gain_db[body])
    gain_db = np.clip(gain_db, -max_db, max_db)
    gain_db[f < 500] = 0.0                                 # the upper mids, presence and air
    ramp = (f >= 500) & (f < 900)
    gain_db[ramp] *= (f[ramp] - 500) / 400
    if np.max(np.abs(gain_db)) < 0.3:
        return processed
    fir = signal.firwin2(taps, f / (sr / 2), 10 ** (gain_db / 20))
    out = signal.fftconvolve(processed, fir[None, :], mode="full", axes=-1)
    d = taps // 2
    return np.ascontiguousarray(out[:, d:d + processed.shape[1]].astype(np.float32))
