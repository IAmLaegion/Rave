"""Measure a nightcore upload's speed-up against its original: scale the original's onset envelope by
each candidate speed and find where it lines up best (with any start offset) with the nightcore's.
Also checks the pitch: plain resampling should raise it by 12*log2(speed) semitones."""
import json
import sys

import librosa
import numpy as np
from scipy import signal

from rave import audio

SR, HOP = 22050, 256
man = json.load(open("/dev-scripts/refs/manifest.json"))
for ref in ("nightcore_happiersunflower", "nightcore_somethingjustlikethis"):
    e = man[ref]
    nc = librosa.resample(audio.mono(audio.load(e["video_path"])), orig_sr=audio.SR, target_sr=SR)
    og = librosa.resample(audio.mono(audio.load(e["original_path"])), orig_sr=audio.SR, target_sr=SR)
    on_nc = librosa.onset.onset_strength(y=nc, sr=SR, hop_length=HOP)
    on_og = librosa.onset.onset_strength(y=og, sr=SR, hop_length=HOP)
    on_nc = (on_nc - on_nc.mean()) / (on_nc.std() + 1e-9)
    t_og = np.arange(len(on_og))

    def score(r):
        # The original played r times faster: frame k of the nightcore = frame k*r of the original.
        warped = np.interp(np.arange(int(len(on_og) / r)) * r, t_og, on_og)
        warped = (warped - warped.mean()) / (warped.std() + 1e-9)
        n = min(len(warped), len(on_nc))
        xc = signal.correlate(on_nc[:n], warped[:n], mode="full", method="fft")[n - 1 - 400:n - 1 + 400] / n
        return float(xc.max()), int(np.argmax(xc)) - 400

    coarse = [(score(r), r) for r in np.arange(1.05, 1.60, 0.01)]
    (_, _), r0 = max(coarse)
    fine = [(score(r), r) for r in np.arange(r0 - 0.012, r0 + 0.012, 0.0005)]
    (best, lag), r = max(fine)
    # Pitch check: chroma of the nightcore vs the original rotated by the expected shift.
    cn = librosa.feature.chroma_cqt(y=nc, sr=SR).mean(axis=1)
    co = librosa.feature.chroma_cqt(y=og, sr=SR).mean(axis=1)
    rot = int(np.argmax([np.corrcoef(cn, np.roll(co, k))[0, 1] for k in range(12)]))
    print(f"{ref}: speed x{r:.4f} (corr {best:.2f}, offset {lag * HOP / SR:+.2f}s), "
          f"durations {len(og) / SR:.1f}s -> {len(nc) / SR:.1f}s; expected pitch +{12 * np.log2(r):.2f} st, "
          f"measured chroma shift +{rot} st", flush=True)
