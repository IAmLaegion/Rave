"""Compare Whisper settings on a song's separated vocals: how many words, and a sample."""
import sys
import time

import librosa
import numpy as np
from faster_whisper import WhisperModel

from rave import audio, separate

path = sys.argv[1]
y = audio.load(path)
v, *_ = separate.separate(y, "demucs", "/data/cache", separate.file_key(path))
m = librosa.resample(audio.mono(v), orig_sr=audio.SR, target_sr=16000).astype(np.float32)
m = m / (np.abs(m).max() + 1e-9) * 0.9
model = WhisperModel("small.en", device="cpu", compute_type="int8")
for label, kw in [("vad (current)", dict(vad_filter=True)),
                  ("no vad", dict(vad_filter=False)),
                  ("vad, loose", dict(vad_filter=True, vad_parameters=dict(threshold=0.2,
                                      min_silence_duration_ms=1000, speech_pad_ms=400)))]:
    t0 = time.time()
    segs, _ = model.transcribe(m, word_timestamps=True, beam_size=5, condition_on_previous_text=False,
                               **kw)
    words = [w.word.strip() for s in segs for w in (s.words or [])]
    print(f"{label:14s} {len(words):4d} words in {time.time() - t0:4.0f}s: {' '.join(words[:40])}",
          flush=True)
