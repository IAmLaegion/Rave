"""Vocal / instrumental separation.

`demucs` (Meta's Hybrid Transformer Demucs) gives studio-grade stems and is what makes a mashup sound
clean. It needs PyTorch and downloads its model (~80 MB) on first use. `fast` is a dependency-light
fallback (REPET-SIM: the repeating background is the instrumental, the rest is the voice); it works,
but expect bleed.
"""
import hashlib
import os

import numpy as np
import soundfile as sf

from . import audio

METHODS = ("auto", "demucs", "fast")
_model = None


def demucs_available():
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "demucs"))


def resolve(method):
    if method == "auto":
        return "demucs" if demucs_available() else "fast"
    if method == "demucs" and not demucs_available():
        raise RuntimeError("Demucs is not installed. Build the image with WITH_DEMUCS=1, or choose 'fast'.")
    return method


def file_key(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def separate(y, method="auto", cache_dir=None, key=None, progress=None):
    """Return (vocals, instrumental), each (2, n), for stereo audio `y`."""
    method = resolve(method)
    paths = None
    if cache_dir and key:
        os.makedirs(cache_dir, exist_ok=True)
        paths = [os.path.join(cache_dir, f"{key}.{method}.{stem}.flac") for stem in ("vocals", "inst")]
        if all(os.path.isfile(p) for p in paths):
            stems = [sf.read(p, dtype="float32", always_2d=True)[0].T for p in paths]
            n = y.shape[1]
            return tuple(_fit(s, n) for s in stems)
    vocals, inst = (_demucs if method == "demucs" else _fast)(y, progress)
    if paths:
        for p, s in zip(paths, (vocals, inst)):
            sf.write(p, np.clip(s, -1, 1).T, audio.SR, subtype="PCM_24")
    return vocals, inst


def _fit(s, n):
    if s.shape[1] >= n:
        return np.ascontiguousarray(s[:, :n])
    return np.pad(s, ((0, 0), (0, n - s.shape[1])))


def _demucs(y, progress):
    global _model
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if _model is None:
        if progress:
            progress("Loading Demucs model (downloads ~80 MB the first time)...")
        _model = get_model("htdemucs")
        _model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav = torch.from_numpy(np.ascontiguousarray(y))
    if _model.samplerate != audio.SR:
        import julius
        wav = julius.resample_frac(wav, audio.SR, _model.samplerate)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        out = apply_model(_model, ((wav - mean) / std)[None], device=device, shifts=1, split=True,
                          overlap=0.25, progress=False, num_workers=0)[0]
    out = out * std + mean
    if _model.samplerate != audio.SR:
        import julius
        out = julius.resample_frac(out, _model.samplerate, audio.SR)
    out = out.cpu().numpy()
    vi = _model.sources.index("vocals")
    vocals = out[vi]
    inst = out[[i for i in range(len(_model.sources)) if i != vi]].sum(axis=0)
    n = y.shape[1]
    return _fit(vocals.astype(np.float32), n), _fit(inst.astype(np.float32), n)


def _fast(y, progress):
    import librosa

    n_fft, hop = 2048, 512
    specs = [librosa.stft(ch, n_fft=n_fft, hop_length=hop) for ch in y]
    mag = np.abs(specs[0]) + np.abs(specs[1])
    # Work on a downsampled magnitude for the similarity search, it's quadratic in frames.
    step = 2
    small = mag[:, ::step]
    width = int(librosa.time_to_frames(2.0, sr=audio.SR, hop_length=hop * step))
    bg = librosa.decompose.nn_filter(small, aggregate=np.median, metric="cosine", width=width)
    bg = np.repeat(bg, step, axis=1)[:, :mag.shape[1]]
    bg = np.minimum(mag, bg)
    fg = mag - bg
    mask_v = librosa.util.softmask(fg, 2.0 * bg, power=2)
    mask_i = librosa.util.softmask(bg, 1.0 * fg, power=2)
    # Voices live roughly in 100 Hz - 10 kHz; anything below is bass and kick.
    freqs = librosa.fft_frequencies(sr=audio.SR, n_fft=n_fft)
    band = ((freqs > 100) & (freqs < 10000)).astype(np.float32)[:, None]
    mask_v *= band
    mask_i = np.maximum(mask_i, 1.0 - band)
    n = y.shape[1]
    vocals = np.stack([librosa.istft(s * mask_v, hop_length=hop, length=n) for s in specs])
    inst = np.stack([librosa.istft(s * mask_i, hop_length=hop, length=n) for s in specs])
    return vocals.astype(np.float32), inst.astype(np.float32)
