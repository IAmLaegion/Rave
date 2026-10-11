"""Stem separation: vocals, drums, bass and everything else.

`demucs` (Meta's Hybrid Transformer Demucs) gives studio-grade stems and is what makes a mashup sound
clean. It needs PyTorch and downloads its model (~80 MB) on first use. `fast` is a dependency-light
fallback (REPET-SIM: the repeating background is the instrumental, the rest is the voice); it works,
but expect bleed. `roformer` uses BS-RoFormer (the best open vocal separator) for the vocals and
Demucs to split the rest into drums, bass and other: cleaner vocals, several times slower on a CPU.
Stems from an uploaded stem pack (see stempack.py) win over all of them.
"""
import hashlib
import os

import numpy as np
import soundfile as sf

from . import audio

METHODS = ("auto", "best", "stemdeck", "demucs", "roformer", "fast")
# "best" (GPU): the cleanest open models per stem: MelBand-RoFormer for the vocals (vocal SDR 12.6 dB,
# vs 9.9 for htdemucs and 9.6 for StemDeck's htdemucs_6s), then the fine-tuned htdemucs_ft on what's left
# for drums (10.0), bass (12.0) and other.
BEST_VOCALS = os.environ.get("RAVE_BEST_VOCALS", "vocals_mel_band_roformer.ckpt")
ROFORMER_MODEL = os.environ.get("RAVE_ROFORMER_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
ROFORMER_DIR = os.environ.get("RAVE_ROFORMER_DIR", "/opt/models/audio-separator")
_model = None


def demucs_available():
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "demucs"))


def has_pack(cache_dir, key):
    """The song came as an uploaded stem pack (its stems are already in the cache)."""
    return bool(cache_dir and key) and all(
        os.path.isfile(os.path.join(cache_dir, f"{key}.pack.{s}.flac")) for s in ("vocals", "drums", "bass", "other"))


def best_available():
    try:
        import torch
        return roformer_available() and torch.cuda.is_available()
    except ImportError:
        return False


def roformer_available():
    import importlib.util
    return demucs_available() and importlib.util.find_spec("audio_separator") is not None


def resolve(method):
    if method == "auto":
        if best_available():
            return "best"                              # our own GPU, the cleanest models
        from . import stemdeck
        if stemdeck.available():
            return "stemdeck"                          # StemDeck's GPU: seconds per song
        return "demucs" if demucs_available() else "fast"
    if method == "best" and not best_available():
        raise RuntimeError("The best-quality separation needs this image's GPU build; choose another.")
    if method == "demucs" and not demucs_available():
        raise RuntimeError("Demucs is not installed. Build the image with WITH_DEMUCS=1, or choose 'fast'.")
    if method == "stemdeck":
        from . import stemdeck
        if not stemdeck.available():
            raise RuntimeError("StemDeck isn't reachable (RAVE_STEMDECK_URL); choose Demucs.")
    if method == "roformer" and not roformer_available():
        raise RuntimeError("BS-RoFormer isn't installed in this image; choose Demucs.")
    return method


def file_key(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def separate(y, method="auto", cache_dir=None, key=None, progress=None):
    """Return (vocals, drums, bass, other), each (2, n), for stereo audio `y`. "Other" is the chords
    and hooks: keys, guitars, synths. The fast separator can't pull drums and bass apart, so it returns
    None for them and the whole instrumental as "other"."""
    method = resolve(method)
    four = ("vocals", "drums", "bass", "other")
    if cache_dir and key:
        pack = [os.path.join(cache_dir, f"{key}.pack.{stem}.flac") for stem in four]
        if all(os.path.isfile(p) for p in pack):          # the song came as a stem pack
            return tuple(_fit(sf.read(p, dtype="float32", always_2d=True)[0].T, y.shape[1]) for p in pack)
    stems = four if method in ("demucs", "roformer", "stemdeck", "best") else ("vocals", "inst")
    paths = None
    if cache_dir and key:
        os.makedirs(cache_dir, exist_ok=True)
        paths = [os.path.join(cache_dir, f"{key}.{method}.{stem}.flac") for stem in stems]
        if all(os.path.isfile(p) for p in paths):
            got = [_fit(sf.read(p, dtype="float32", always_2d=True)[0].T, y.shape[1]) for p in paths]
            return tuple(got) if len(got) == 4 else (got[0], None, None, got[1])
    out = {"demucs": _demucs, "roformer": _roformer, "fast": _fast, "stemdeck": _stemdeck,
           "best": _best}[method](y, progress)
    if paths:
        for p, s in zip(paths, [s for s in out if s is not None]):
            sf.write(p, np.clip(s, -1, 1).T, audio.SR, subtype="PCM_24")
    return out


def _roformer_vocals(y, model, progress, label):
    """The vocal stem of `y` from an audio-separator model (GPU when there is one)."""
    import tempfile
    from audio_separator.separator import Separator
    if progress:
        progress(f"Separating vocals with {label}...")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "song.wav")
        sf.write(src, y.T, audio.SR, subtype="FLOAT")
        sep = Separator(output_dir=tmp, model_file_dir=ROFORMER_DIR, output_format="WAV",
                        sample_rate=audio.SR, log_level=40)
        sep.load_model(model_filename=model)
        files = [os.path.join(tmp, f) for f in sep.separate(src)]
        vocals = next((f for f in files if "(vocals)" in f.lower()), None)
        if vocals is None:
            raise RuntimeError(f"{label} produced no vocal stem.")
        v = sf.read(vocals, dtype="float32", always_2d=True)[0].T
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return _fit(v, y.shape[1])


def _roformer(y, progress):
    """BS-RoFormer for the vocals, then Demucs on what's left for drums, bass and other."""
    v = _roformer_vocals(y, ROFORMER_MODEL, progress, "BS-RoFormer (slow on a CPU)")
    if progress:
        progress("Splitting the instrumental with Demucs...")
    _, d, b, o = _demucs(y - v, None)
    return v, d, b, o


def _best(y, progress):
    """MelBand-RoFormer for the vocals, then htdemucs_ft on the instrumental that's left (with the
    voice already gone, the drum/bass/other split is cleaner too). Falls back to StemDeck or plain
    Demucs if the shared GPU is full."""
    try:
        v = _roformer_vocals(y, BEST_VOCALS, progress, "MelBand-RoFormer on the GPU")
        if progress:
            progress("Splitting the instrumental with htdemucs_ft on the GPU...")
        _, d, b, o = _demucs(y - v, None, model="htdemucs_ft")
        return v, d, b, o
    except RuntimeError as e:
        if "out of memory" not in str(e).lower() and "cuda" not in str(e).lower():
            raise
        if progress:
            progress("The GPU is busy; separating another way instead...")
        from . import stemdeck
        return stemdeck.separate(y, None) if stemdeck.available() else _demucs(y, progress)


def _stemdeck(y, progress):
    from . import stemdeck
    return stemdeck.separate(y, progress)


def _fit(s, n):
    if s.shape[1] >= n:
        return np.ascontiguousarray(s[:, :n])
    return np.pad(s, ((0, 0), (0, n - s.shape[1])))


_models = {}


def _demucs(y, progress, model="htdemucs"):
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if model not in _models:
        if progress:
            progress("Loading the Demucs model...")
        _models[model] = get_model(model)
        _models[model].eval()
    _model = _models[model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav = torch.from_numpy(np.ascontiguousarray(y))
    if _model.samplerate != audio.SR:
        import julius
        wav = julius.resample_frac(wav, audio.SR, _model.samplerate)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    run = lambda dev: apply_model(_model, ((wav - mean) / std)[None], device=dev, shifts=1, split=True,
                                  overlap=0.25, progress=False, num_workers=0)[0]
    with torch.no_grad():
        try:
            out = run(device)
        except RuntimeError as e:                      # the GPU is shared (LocalAI, StemDeck): if it's
            if device != "cuda" or "out of memory" not in str(e).lower():   # full, use the CPU
                raise
            torch.cuda.empty_cache()
            if progress:
                progress("The GPU is busy; separating on the CPU instead (slower)...")
            out = run("cpu")
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()
    out = out * std + mean
    if _model.samplerate != audio.SR:
        import julius
        out = julius.resample_frac(out, _model.samplerate, audio.SR)
    out = out.cpu().numpy()
    src = list(_model.sources)
    vocals, drums, bass = (out[src.index(k)] for k in ("vocals", "drums", "bass"))
    other = out[[i for i, s in enumerate(src) if s not in ("vocals", "drums", "bass")]].sum(axis=0)
    n = y.shape[1]
    return tuple(_fit(s.astype(np.float32), n) for s in (vocals, drums, bass, other))


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
    return vocals.astype(np.float32), None, None, inst.astype(np.float32)
