"""How much do tracked bar lengths deviate from a steady grid? Each deviation is audio skipped or repeated at a bar joint."""
import glob, sys
import numpy as np
from rave import analysis as an, audio

for p in sorted(glob.glob("/data/cache/youtube/*")):
    y = audio.load(p)
    a = an.analyze(y)
    b = a.beats
    d = np.diff(b)
    k = np.round((b - b[0]) / np.median(d))
    slope, icpt = np.polyfit(k, b, 1)
    resid = (b - (icpt + slope * k)) * 1000
    bars = b[a.downbeat_phase::4]
    bl = np.diff(bars)
    err = (bl - np.median(bl)) * 1000
    print(p.split("/")[-1], f"bpm(median)={a.bpm:.2f} bpm(fit)={60/slope:.2f} phase={a.downbeat_phase}")
    print(f"  beat jitter vs straight grid: mean |{np.abs(resid).mean():.1f}| ms, max {np.abs(resid).max():.1f} ms, missing/extra beats: {int((np.abs(np.diff(k)-1)>0).sum())}")
    print(f"  bar length error: mean |{np.abs(err).mean():.1f}| ms, >10ms in {int((np.abs(err)>10).sum())}/{len(err)} bars, max {np.abs(err).max():.0f} ms")
