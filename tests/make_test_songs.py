"""Synthesize small test songs (drums, bass, chords, a sung-ish lead) at different tempos and keys,
so the pipeline can be exercised without real (copyrighted) music.

    python tests/make_test_songs.py OUT_DIR
"""
import os
import sys

import numpy as np
import soundfile as sf

SR = 44100
rng = np.random.default_rng(1)


def tone(freq, dur, harmonics=6, vibrato=0.0, bright=1.0):
    t = np.arange(int(dur * SR)) / SR
    f = freq * (1 + vibrato * np.sin(2 * np.pi * 5.5 * t))
    phase = 2 * np.pi * np.cumsum(f) / SR
    y = sum(np.sin(h * phase) / h ** (1.5 / bright) for h in range(1, harmonics + 1))
    env = np.minimum(1, t / 0.02) * np.minimum(1, (dur - t) / 0.05).clip(0)
    return y * env


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def song(bpm, tonic, minor, seed):
    r = np.random.default_rng(seed)
    beat = 60 / bpm
    bar = 4 * beat
    third = 3 if minor else 4
    # verse: i-VI-III-VII (minor) / I-V-vi-IV (major); chorus: IV-V-I-I-ish lift
    prog_v = [0, 8, 3, 10] if minor else [0, 7, 9, 5]
    prog_c = [5, 7, 0, 0] if not minor else [8, 10, 0, 0]
    sections = ["intro"] * 4 + ["verse"] * 8 + ["chorus"] * 8 + ["verse"] * 8 + ["chorus"] * 16 + ["outro"] * 4
    n = int(len(sections) * bar * SR) + SR
    mix = np.zeros((2, n))
    voc = np.zeros(n)
    hook = [r.integers(0, 3) for _ in range(8)]
    verse_mel = [r.integers(0, 3) for _ in range(8)]
    for b, sec in enumerate(sections):
        start = b * bar
        prog = prog_c if sec == "chorus" else prog_v
        root = tonic + prog[b % 4]
        quality = third if prog[b % 4] == 0 else (4 if (root - tonic) % 12 in (5, 7, 8, 10) else 3)
        chord = [root + 48, root + 48 + quality, root + 55]
        loud = 1.0 if sec == "chorus" else 0.7
        # pad
        pad = sum(tone(midi(c), bar, 5, bright=0.6) for c in chord) * 0.06 * loud
        i0 = int(start * SR)
        mix[0, i0:i0 + len(pad)] += pad * 1.1
        mix[1, i0:i0 + len(pad)] += pad * 0.9
        # bass
        for k in range(4):
            bs = tone(midi(root + 36), beat * 0.9, 3) * 0.25
            j = int((start + k * beat) * SR)
            mix[:, j:j + len(bs)] += bs
        # drums
        for k in range(8):
            j = int((start + k * beat / 2) * SR)
            if k % 4 == 0:
                tt = np.arange(int(0.25 * SR)) / SR
                kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 12) * 0.9
                mix[:, j:j + len(kick)] += kick
            if k % 4 == 2:
                tt = np.arange(int(0.2 * SR)) / SR
                sn = r.standard_normal(len(tt)) * np.exp(-tt * 20) * 0.35
                mix[:, j:j + len(sn)] += sn
            hh = r.standard_normal(int(0.04 * SR)) * np.exp(-np.arange(int(0.04 * SR)) / SR * 120) * 0.08
            hh = np.diff(hh, prepend=0)
            mix[:, j:j + len(hh)] += hh
        # lead vocal: chord tones, 2 notes per bar, sung in verses and choruses
        if sec in ("verse", "chorus"):
            mel = hook if sec == "chorus" else verse_mel
            for k in range(2):
                note = chord[mel[(b * 2 + k) % 8]] + 24
                v = tone(midi(note), bar / 2 * 0.92, 10, vibrato=0.006, bright=1.4) * 0.22 * loud
                j = int((start + k * bar / 2) * SR)
                voc[j:j + len(v)] += v
    mix += voc
    lead_in = int(0.37 * SR)                      # a bit of silence before the music starts
    mix = np.pad(mix, ((0, 0), (lead_in, 0)))
    return (mix / np.abs(mix).max() * 0.9).T.astype(np.float32)


SONGS = [("Song A (100 C)", 100, 0, False), ("Song B (126 Em)", 126, 4, True),
         ("Song C (58 D)", 58, 2, False)]


def main(out):
    os.makedirs(out, exist_ok=True)
    paths = []
    for i, (name, bpm, tonic, minor) in enumerate(SONGS):
        p = os.path.join(out, name + ".wav")
        sf.write(p, song(bpm, tonic, minor, i), SR)
        paths.append(p)
        print(p)
    return paths


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_songs")
