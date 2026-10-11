import os, shutil, subprocess, numpy as np, soundfile as sf
from rave import stempack, audio
d = "/tmp/st"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
ins = []
for k, f in enumerate([0, 60, 110, 440, 880]):
    t = np.arange(44100 * 12) / 44100
    y = (0.2 * np.sin(2 * np.pi * f * t) if f else 0.1 * np.random.randn(len(t))).astype(np.float32)
    p = f"{d}/t{k}.wav"; sf.write(p, np.stack([y, y]).T, 44100); ins += ["-i", p]
maps = sum([["-map", f"{k}:a"] for k in range(5)], [])
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, *maps, "-c:a", "aac", f"{d}/Song A.stem.mp4"], check=True)
for name, path in stempack.prepare(f"{d}/Song A.stem.mp4", f"{d}/w1", f"{d}/cache", "Song A", print):
    key = __import__("rave.separate", fromlist=["x"]).file_key(path)
    for s in ("vocals", "drums", "bass", "other"):
        y, _ = sf.read(f"{d}/cache/{key}.pack.{s}.flac")
        sp = np.abs(np.fft.rfft(y[:44100 * 4, 0])); print(name, s, "peak Hz", np.argmax(sp) / 4)
shutil.copy(f"{d}/Song A.stem.mp4", f"{d}/Song B.stem.mp4")
subprocess.run(["7z", "a", "-bd", f"{d}/pack.zip", f"{d}/Song A.stem.mp4", f"{d}/Song B.stem.mp4"], check=True, capture_output=True)
print([n for n, _ in stempack.prepare(f"{d}/pack.zip", f"{d}/w2", f"{d}/cache2", "pack", None)])
print(stempack.is_archive("x.stem.mp4"), stempack.is_archive("x.mp4"))
