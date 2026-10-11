import glob, json, numpy as np
from scipy import signal
from rave import audio, assist, sources
man = json.load(open("/dev-scripts/refs/manifest.json"))["summermash16"]["songs"]
mix = audio.load(man["Flo Rida - My House"]["path"])
r = next(assist._search("Flo Rida - My House", mix.shape[1] / 44100))
inst = audio.load(sources.download_youtube(r["id"], "/data/cache"))
sr = 44100
print("mix", mix.shape[1] / sr, "inst", inst.shape[1] / sr, r["title"])
m, i = audio.mono(mix), audio.mono(inst)
lag = assist._lag(signal.decimate(m, 11), signal.decimate(i, 11), 4009) * 11
print("coarse lag", lag, "samples =", lag / sr, "s")
for at in (5, 20, 60, 120, 180):
    a0 = at * sr; w = sr * 6
    b0 = a0 - lag
    if b0 < 0 or b0 + w > len(i): continue
    l = lag + assist._lag(m[a0:a0 + w], i[b0:b0 + w], sr)
    seg_m = mix[:, a0:a0 + w]; seg_i = inst[:, a0 - l:a0 - l + w]
    g = np.dot(audio.mono(seg_m), audio.mono(seg_i)) / np.dot(audio.mono(seg_i), audio.mono(seg_i))
    res = seg_m - g * seg_i
    print(f"t={at:4d}s fine lag {l:7d}  gain {g:.3f}  residual {20*np.log10(audio.rms(res)/audio.rms(seg_m)):6.1f} dB"
          f"  (vs -inst {20*np.log10(audio.rms(seg_m + g*seg_i)/audio.rms(seg_m)):5.1f})")

# Adaptive (per-bin, per-2s) subtraction in the STFT domain.
l = 16844
n = min(mix.shape[1], inst.shape[1] + l)
ia = np.zeros_like(mix[:, :n]); ia[:, l:n] = inst[:, : n - l]
mm = mix[:, :n]
nfft, hop = 4096, 1024
def stft(x): return np.stack([signal.stft(c, sr, nperseg=nfft, noverlap=nfft - hop)[2] for c in x])
M, I = stft(mm), stft(ia)
for win in (16, 43, 86):
    R = np.empty_like(M)
    T = M.shape[2]
    for c in range(2):
        for t0 in range(0, T, win):
            sl = slice(max(0, t0 - win), min(T, t0 + 2 * win))
            H = (M[c, :, sl] * np.conj(I[c, :, sl])).sum(1) / ((np.abs(I[c, :, sl]) ** 2).sum(1) + 1e-9)
            R[c, :, t0:t0 + win] = M[c, :, t0:t0 + win] - H[:, None] * I[c, :, t0:t0 + win]
    res = np.stack([signal.istft(R[c], sr, nperseg=nfft, noverlap=nfft - hop)[1][:n] for c in range(2)])
    out = []
    for at in (5, 20, 60, 120, 180):
        a0, w = at * sr, 6 * sr
        out.append(f"{at}s {20*np.log10(audio.rms(res[:, a0:a0+w])/audio.rms(mm[:, a0:a0+w])):5.1f}")
    print(f"adaptive window {win * hop / sr:.1f}s:", "  ".join(out))
    if win == 43:
        import soundfile as sf
        sf.write("/dev-scripts/assist_vocals_myhouse.flac", np.clip(res, -1, 1).T, sr)
