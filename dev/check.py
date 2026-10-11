"""Objective checks on a render: is the beat steady through the whole mashup? Beats are counted on the
grid with a robust fit, so a drumless bridge (where the tracker may skip beats) doesn't skew it."""
import sys, numpy as np, librosa
from rave import analysis as an, audio
y = audio.load(sys.argv[1])
b, db = an._beat_this(librosa.resample(audio.mono(y), orig_sr=audio.SR, target_sr=an.ASR))
d = np.diff(b); p = np.median(d)
k = np.r_[0, np.cumsum(np.maximum(1, np.round(d / p)))]
use = np.ones(len(b), bool)
for _ in range(4):
    s, i = np.polyfit(k[use], b[use], 1)
    r = b - (i + s * k)
    use = np.abs(r) < max(4 * np.median(np.abs(r[use])), 0.02)
r = np.abs(r) * 1000
print(f"{sys.argv[1].split('/')[-1]}: {60/s:.2f} BPM, beats off the grid: median {np.median(r):.1f} ms, "
      f"95th pct {np.percentile(r, 95):.1f} ms, beats more than 40 ms off: {int((r > 40).sum())}/{len(r)}")
