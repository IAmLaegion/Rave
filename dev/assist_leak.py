import json, numpy as np, soundfile as sf
from rave import audio, separate
r, sr = sf.read("/dev-scripts/assist_vocals_myhouse.flac", dtype="float32", always_2d=True); r = r.T
man = json.load(open("/dev-scripts/refs/manifest.json"))["summermash16"]["songs"]
mix = audio.load(man["Flo Rida - My House"]["path"])[:, : r.shape[1]]
v_res = separate._roformer_vocals(r, separate.BEST_VOCALS, None, "check")
leak = r - v_res
v_mix = separate._roformer_vocals(mix, separate.BEST_VOCALS, None, "check")
db = lambda a, b: 20 * np.log10(audio.rms(a) / audio.rms(b))
print(f"residual (mix - official instrumental): non-vocal content {db(leak, r):6.1f} dB vs the residual")
print(f"model vocals from the mix:              level {db(v_mix, mix):6.1f} dB vs the mix")
print(f"residual level vs mix                   {db(r, mix):6.1f} dB")
print(f"residual vs model vocals: difference    {db(r - v_mix, v_mix):6.1f} dB (lower = same signal)")
