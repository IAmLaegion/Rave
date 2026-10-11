"""Where are the vocal gaps in a render? Lists every stretch of 1.5 s or more without singing, next to
the cue-sheet part playing there, and totals the gap time by part type."""
import re
import sys
from collections import defaultdict

import librosa
import numpy as np

sys.path.insert(0, "/dev-scripts/survey")
from rave import audio, separate  # noqa: E402

path, cue = sys.argv[1], sys.argv[2]
y = audio.load(path)
v, *_ = separate.separate(y, "stemdeck", "/data/cache", separate.file_key(path))
sr, hop = 22050, 1024
rv = librosa.feature.rms(y=librosa.resample(audio.mono(v), orig_sr=audio.SR, target_sr=sr), hop_length=hop)[0]
t = librosa.frames_to_time(np.arange(len(rv)), sr=sr, hop_length=hop)
on = rv > 0.15 * np.percentile(rv, 95)
parts = []
for line in open(cue, encoding="utf-8"):
    m = re.match(r"\s+(\d+):(\d+\.\d)\s+(\d+) bars\s+(.*)", line)
    if m:
        parts.append((int(m.group(1)) * 60 + float(m.group(2)), m.group(4)))
label_at = lambda x: next((p[1] for p in reversed(parts) if p[0] <= x), "?")
gaps, start = [], None
for i, a in enumerate(on):
    if not a and start is None:
        start = t[i]
    if a and start is not None:
        if t[i] - start >= 1.5:
            gaps.append((start, t[i]))
        start = None
if start is not None and t[-1] - start >= 1.5:
    gaps.append((start, t[-1]))
by = defaultdict(float)
for a, b in gaps:
    lab = label_at(a).split(":")[0]
    by[lab] += b - a
    print(f"  {int(a // 60)}:{a % 60:05.2f}  {b - a:4.1f}s  {label_at(a)[:70]}")
print(f"vocal share {on.mean():.3f}; gap seconds by part: " +
      ", ".join(f"{k} {v:.0f}s" for k, v in sorted(by.items(), key=lambda kv: -kv[1])))
