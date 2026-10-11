"""Audio I/O. Decoding and encoding go through ffmpeg so any format works (mp3, m4a, flac, ...)."""
import functools
import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

SR = 44100


@functools.lru_cache(maxsize=None)
def find_ffmpeg():
    """ffmpeg from $RAVE_FFMPEG, PATH, or the imageio-ffmpeg package."""
    for c in (os.environ.get("RAVE_FFMPEG"), shutil.which("ffmpeg")):
        if c and os.path.isfile(c):
            return c
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise RuntimeError("ffmpeg was not found. Install it or set RAVE_FFMPEG.")


def run_ffmpeg(args, input_bytes=None):
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin"] + args
    p = subprocess.run(cmd, input=input_bytes, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + p.stderr.decode(errors="replace").strip())
    return p.stdout


@functools.lru_cache(maxsize=None)
def has_filter(name):
    try:
        out = run_ffmpeg(["-filters"]).decode(errors="replace")
    except Exception:
        return False
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


def load(path, sr=SR):
    """Decode any audio file to float32 stereo, shape (2, n)."""
    raw = run_ffmpeg(["-i", path, "-vn", "-f", "f32le", "-acodec", "pcm_f32le",
                      "-ac", "2", "-ar", str(sr), "-"])
    y = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T
    if y.shape[1] == 0:
        raise RuntimeError(f"No audio decoded from {path}")
    return np.ascontiguousarray(y)


def save(path, y, sr=SR):
    """Write (2, n) audio. WAV/FLAC directly; anything else (mp3, m4a, ogg) through ffmpeg."""
    ext = os.path.splitext(path)[1].lower()
    data = np.clip(y, -1.0, 1.0).T.astype(np.float32)
    if ext in (".wav", ".flac"):
        sf.write(path, data, sr, subtype="PCM_24")
        return path
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "mix.wav")
        sf.write(wav, data, sr, subtype="PCM_24")
        codec = {".mp3": ["-codec:a", "libmp3lame", "-b:a", "320k"],
                 ".m4a": ["-codec:a", "aac", "-b:a", "256k"],
                 ".ogg": ["-codec:a", "libvorbis", "-q:a", "8"]}.get(ext, [])
        run_ffmpeg(["-y", "-i", wav] + codec + [path])
    return path


def mono(y):
    return y.mean(axis=0) if y.ndim == 2 else y


def rms(x):
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))


def db(x):
    return 20.0 * np.log10(max(x, 1e-9))


def undb(d):
    return 10.0 ** (d / 20.0)
