import json, numpy as np
from scipy import signal
from rave import audio, separate, transform
SR = 44100
song = json.load(open("/dev-scripts/refs/manifest.json"))["summermash16"]["songs"]["Flo Rida - My House"]["path"]
y = audio.load(song)
v = separate.separate(y, "demucs", "/data/cache", separate.file_key(song))[0]
seg = v[:, SR * 30: SR * 90]
w = transform.warp(seg, [SR, 50 * SR], [int(SR / .9), int(50 * SR / .9)], int(seg.shape[1] / .9), 0.0, vocal=True)
r = transform.restore_tone(seg, w, 0.0)
edges = [0, 150, 300, 700, 1500, 3000, 5000, 7500, 10000, 15000, 22050]
def bands(x):
    f, p = signal.welch(audio.mono(x), SR, nperseg=4096)
    e = np.array([p[(f >= a) & (f < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
    body = p[(f >= 300) & (f <= 1500)].sum()
    return 10 * np.log10(e / body + 1e-20)
for name, x in (("original", seg), ("stretched", w), ("restored", r)):
    print(f"{name:10s}", " ".join(f"{v:6.1f}" for v in bands(x)), f" level {20*np.log10(audio.rms(x)):.1f} dB")
print("bands    ", " ".join(f"{a//1000 if a>=1000 else a}-{b//1000 if b>=1000 else b}".rjust(6) for a, b in zip(edges[:-1], edges[1:])))
