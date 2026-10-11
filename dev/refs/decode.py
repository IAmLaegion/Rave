"""Decode a mashup back into its sources: which song sings, and which song's harmony and drums play,
beat by beat, from where in the original and in what transposition.

    python decode.py <ref name> [--original]      (ref names in songs.py; --original decodes the
                                                   un-sped original of a nightcore ref)

Each 2-bar window of the mashup is compared with every 2-bar window of every source song, at all 12
transpositions: beat-synchronous chroma (melody/chords) plus the 16th-note rhythm pattern. A Viterbi
pass then joins the best matches into segments, preferring to keep going through the same song from
the same place. Writes <ref>.decoded.json and prints a cue sheet.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/dev-scripts/refs")
from features import features  # noqa: E402
from rave import analysis as an  # noqa: E402

W = 8                       # window, beats
K = 8                       # candidates kept per window
NOTES = an.NOTES


def fold(f, factor, lane):
    """Re-grid a song's per-beat features for half/double time."""
    c, r = f[f"chroma_{lane}"], f[f"rhythm_{'d' if lane == 'd' else lane}"]
    rms = f[f"rms_{lane}"]
    if factor == 2.0:
        c2 = np.repeat(c, 2, axis=0)
        r2 = np.stack([np.interp(np.linspace(0, 1, 4, endpoint=False), [0, .25, .5, .75], row)
                       for row in r for _ in range(1)])
        halves = []
        for row in r:
            halves.append(np.interp([0, .5, 1, 1.5], [0, 1, 2, 3], row))
            halves.append(np.interp([2, 2.5, 3, 3.5], [0, 1, 2, 3], row))
        return c2, np.array(halves), np.repeat(rms, 2)
    if factor == 0.5:
        n = len(c) // 2 * 2
        c2 = (c[:n:2] + c[1:n:2]) / 2
        r2 = np.concatenate([r[:n:2, ::2], r[1:n:2, ::2]], axis=1)
        return c2, r2, (rms[:n:2] + rms[1:n:2]) / 2
    return c, r, rms


def windows(x):
    n = len(x) - W + 1
    if n <= 0:
        return np.zeros((0, W * x.shape[1]))
    return np.stack([x[i:i + W].ravel() for i in range(n)])


def unit(m):
    m = m - m.mean(axis=1, keepdims=True)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def match(ref, srcs, lane, wc=0.6, wr=0.4, fixed_key=True):
    """Top-K (score, source, source beat, semitones) for every window of the reference."""
    rr = ref[f"rhythm_{lane}"]
    Rc = unit(windows(ref[f"chroma_{lane}"])) if lane != "d" else None
    Rr = unit(windows(np.log1p(rr)))
    n = len(Rr)
    best = [[] for _ in range(n)]
    for si, (name, f) in enumerate(srcs):
        factor = an.tempo_fold(float(f["bpm"]), float(ref["bpm"]))
        if lane == "d":
            c, r, rms = None, fold(f, factor, "h")[1] if False else f["rhythm_d"], f["rms_d"]
            if factor != 1.0:
                _, r, rms = fold(dict(chroma_d=f["chroma_h"], rhythm_d=f["rhythm_d"], rms_d=f["rms_d"]),
                                 factor, "d")
        else:
            c, r, rms = fold(f, factor, lane)
        Sr = unit(windows(np.log1p(r)))
        if len(Sr) == 0:
            continue
        rhythm = Rr @ Sr.T                                        # (n, ns)
        # Windows of the source with nothing in this stem can't be what's playing.
        live = np.array([rms[j:j + W].mean() for j in range(len(Sr))]) > 0.15 * np.percentile(rms, 90)
        if lane == "d":
            score = rhythm
            rots = [0]
            scores = [score]
        else:
            scores = []
            want = an.key_shift(int(f["key"]), bool(f["minor"]), int(ref["key"]), bool(ref["minor"])) % 12
            rots = [want, (want + 1) % 12, (want - 1) % 12] if fixed_key else list(range(12))
            for rot in rots:
                Sc = unit(windows(np.roll(c, rot, axis=1)))
                scores.append(wc * (Rc @ Sc.T) + wr * rhythm)
        for rot, sc in zip(rots, scores):
            sc = np.where(live[None, :], sc, -1)
            top = np.argpartition(-sc, min(K, sc.shape[1] - 1), axis=1)[:, :K]
            for i in range(n):
                for j in top[i]:
                    best[i].append((float(sc[i, j]), si, int(j), rot))
    return [sorted(b, reverse=True)[:K] for b in best]


def viterbi(cands, active, none_score=0.3, keep=0.25, switch=0.6):
    """Join per-window candidates into a path; state -1 = nothing from the sources."""
    n = len(cands)
    states = [[(-1, -1, -1, none_score if not active[i] else 0.0)] +
              [(s, j, r, sc) for sc, s, j, r in cands[i]] for i in range(n)]
    score = [np.array([st[3] for st in states[0]])]
    back = []
    for i in range(1, n):
        prev, cur = states[i - 1], states[i]
        sc = np.full(len(cur), -1e9)
        bk = np.zeros(len(cur), dtype=int)
        for a, (s, j, r, e) in enumerate(cur):
            for b, (ps, pj, pr, _) in enumerate(prev):
                if s == -1 and ps == -1:
                    t = 0.0
                elif s == ps and r == pr and j == pj + 1:
                    t = keep
                else:
                    t = -switch
                v = score[-1][b] + t
                if v > sc[a]:
                    sc[a], bk[a] = v, b
            sc[a] += e
        score.append(sc)
        back.append(bk)
    path = [int(np.argmax(score[-1]))]
    for i in range(n - 2, -1, -1):
        path.append(back[i][path[-1]])
    path.reverse()
    return [states[i][k] for i, k in enumerate(path)]


def segments(path, ref, srcs):
    beats = ref["beats"]
    out, start = [], 0
    for i in range(1, len(path) + 1):
        a, b = path[start], path[i] if i < len(path) else None
        if b is not None and b[0] == a[0] and b[2] == a[2] and b[1] == a[1] + (i - start):
            continue
        if b is not None and a[0] == -1 and b[0] == -1:
            continue
        s, j, r, _ = a
        n = i - start
        scs = [path[k][3] for k in range(start, i)]
        seg = dict(t=float(beats[start]), beats=n, conf=round(float(np.mean(scs)), 3))
        if s >= 0:
            name, f = srcs[s]
            factor = an.tempo_fold(float(f["bpm"]), float(ref["bpm"]))
            fb = f["beats"] if factor == 1.0 else an.fold(f["beats"], int(f["phase"]), factor)[0]
            seg.update(song=name, src_t=float(fb[min(j, len(fb) - 1)]), shift=int(an.wrap_semitones(r)),
                       stretch=round(float(ref["bpm"]) / (float(f["bpm"]) * factor), 3))
        out.append(seg)
        start = i
    return out


def main():
    name = sys.argv[1]
    use_orig = "--original" in sys.argv
    man = json.load(open("/dev-scripts/refs/manifest.json"))[name]
    ref_path = man["original_path" if use_orig else "video_path"]
    print(f"Reference: {name} ({'original' if use_orig else 'video'}) {ref_path}", flush=True)
    ref = features(ref_path)
    srcs = [(song, features(info["path"])) for song, info in man["songs"].items()]
    print(f"Reference {float(ref['bpm']):.1f} BPM, key {NOTES[int(ref['key'])]}{'m' if ref['minor'] else ''}, "
          f"{len(ref['beats'])} beats; {len(srcs)} sources", flush=True)
    result = {"ref": name, "bpm": float(ref["bpm"]), "key": int(ref["key"]), "minor": bool(ref["minor"]),
              "sources": {s: {"bpm": float(f["bpm"]), "key": int(f["key"]), "minor": bool(f["minor"])}
                          for s, f in srcs}}
    for lane, label in (("v", "vocals"), ("h", "harmony"), ("d", "drums")):
        cands = match(ref, srcs, lane)
        rms = ref[f"rms_{lane}"]
        active = np.array([rms[i:i + W].mean() for i in range(len(cands))]) > 0.15 * np.percentile(rms, 90)
        path = viterbi(cands, active)
        segs = segments(path, ref, srcs)
        result[label] = segs
        print(f"\n== {label}", flush=True)
        for s in segs:
            t = s["t"]
            if "song" in s:
                print(f"  {int(t // 60)}:{t % 60:05.2f}  {s['beats']:3d} beats  {s['song'][:42]:42s} "
                      f"@{s['src_t']:6.1f}s  {s['shift']:+d} st  x{s['stretch']}  conf {s['conf']}")
            else:
                print(f"  {int(t // 60)}:{t % 60:05.2f}  {s['beats']:3d} beats  -")
    suffix = ".original" if use_orig else ""
    json.dump(result, open(f"/dev-scripts/refs/{name}{suffix}.decoded.json", "w"), indent=1)


if __name__ == "__main__":
    main()
