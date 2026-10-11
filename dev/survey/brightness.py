"""How bright (vs muffled) is a track? Share of energy above 5 kHz and above 10 kHz, and the 85% spectral
rolloff, for renders vs the reference corpus, and for each processing stage of one vocal."""
import json
import os
import sys

import librosa
import numpy as np

from rave import audio, separate, transform

SR = 44100


def bright(y):
    m = audio.mono(y)
    m = m[: SR * 120]
    S = np.abs(librosa.stft(m, n_fft=4096, hop_length=2048)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=4096)
    tot = S.sum() + 1e-12
    roll = librosa.feature.spectral_rolloff(S=np.sqrt(S), sr=SR, roll_percent=0.85)[0]
    return dict(above5k=round(float(S[f > 5000].sum() / tot) * 100, 2),
                above10k=round(float(S[f > 10000].sum() / tot) * 100, 3),
                rolloff=round(float(np.median(roll))))


refs = json.load(open("/dev-scripts/survey/survey.json"))
rows = []
for e in refs["earworm"]:
    p = [x for x in os.listdir("/data/cache/youtube") if x.startswith(e["id"])]
    if p:
        rows.append(bright(audio.load("/data/cache/youtube/" + p[0])))
print("earworm refs (median):", {k: float(np.median([r[k] for r in rows])) for k in rows[0]})
for s in ("earworm", "riffoff", "megamix", "earworm_old"):
    p = f"/dev-scripts/survey/eval-{s}.mp3"
    if os.path.isfile(p):
        print(f"render {s:12s}", bright(audio.load(p)))

# One song's vocal through each stage: original mix, separated vocal (Demucs 4-stem, StemDeck 6-stem),
# then our pitch shift / stretch.
song = json.load(open("/dev-scripts/refs/manifest.json"))["summermash16"]["songs"]["Flo Rida - My House"]["path"]
y = audio.load(song)
k = separate.file_key(song)
print("\nFlo Rida - My House:")
print("  original mix        ", bright(y))
v_dm = separate.separate(y, "demucs", "/data/cache", k)[0]
print("  vocals Demucs htdemucs", bright(v_dm))
try:
    v_sd = separate.separate(y, "stemdeck", "/data/cache", k)[0]
    print("  vocals StemDeck 6s  ", bright(v_sd))
except Exception as ex:
    print("  StemDeck:", ex)
seg = v_dm[:, SR * 30: SR * 90]
print("  vocal segment       ", bright(seg))
print("  +2 st, x1.0 (R3)    ", bright(transform.warp(seg, [SR, 2 * SR], [SR, 2 * SR], seg.shape[1], 2.0, vocal=True)))
print("  0 st, x0.9 stretch  ", bright(transform.warp(seg, [SR, 50 * SR], [int(SR / .9), int(50 * SR / .9)],
                                                         int(seg.shape[1] / .9), 0.0, vocal=True)))
for label, out in (("+2 st", transform.warp(seg, [SR, 2 * SR], [SR, 2 * SR], seg.shape[1], 2.0, vocal=True)),
                   ("x0.9", transform.warp(seg, [SR, 50 * SR], [int(SR / .9), int(50 * SR / .9)],
                                           int(seg.shape[1] / .9), 0.0, vocal=True))):
    print(f"  {label} + tone restore  ", bright(transform.restore_tone(seg, out, 0.0)))
