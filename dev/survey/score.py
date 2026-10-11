"""Score a Rave render against the reference corpus (the benchmark).

    python score.py <render.mp3> <category> [<render2.mp3> <category2> ...]
    categories: earworm, megamix, pitchperfect, nightcore, 8d

The render is measured exactly like the references (survey.measure). Each metric scores 100 at the
references' median, falling linearly to 0 at twice the distance of the farthest reference quartile
(so anything inside the references' middle half scores at least 50). The total is the mean. Prints a
table of render value vs. reference median and range, so it's clear what to change.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/dev-scripts/survey")
from survey import OUT, measure  # noqa: E402

METRICS = ["duration", "bpm", "first_vocal", "vocal_share", "rms_db", "crest_db", "stereo_corr",
           "balance_swing", "rotation_period"]


def metric_score(x, ref):
    ref = np.array([r for r in ref if r is not None], dtype=float)
    if x is None or len(ref) < 2:
        return None
    med = float(np.median(ref))
    q1, q3 = np.percentile(ref, [25, 75])
    spread = max(med - q1, q3 - med, 1e-6)
    return float(np.clip(100 * (1 - abs(x - med) / (4 * spread)), 0, 100))


def curve_score(curve, refs):
    refs = np.array(refs, dtype=float)
    med = np.median(refs, axis=0)
    dist = np.mean(np.abs(np.array(curve) - med))
    typical = np.median([np.mean(np.abs(r - med)) for r in refs]) + 1e-6
    return float(np.clip(100 * (1 - dist / (3 * typical)), 0, 100))


def score(path, category):
    corpus = json.load(open(OUT))
    refs = corpus.get(category) or []
    if len(refs) < 3:
        raise SystemExit(f"Not enough references for '{category}' yet ({len(refs)}).")
    m = measure(path)
    rows, scores = [], []
    for k in METRICS:
        vals = [r.get(k) for r in refs]
        s = metric_score(m.get(k), vals)
        good = [v for v in vals if v is not None]
        rows.append((k, m.get(k), float(np.median(good)) if good else None,
                     (min(good), max(good)) if good else None, s))
        if s is not None:
            scores.append(s)
    cs = curve_score(m["energy_curve"], [r["energy_curve"] for r in refs])
    scores.append(cs)
    total = float(np.mean(scores))
    print(f"\n{path.split('/')[-1]}  vs  {category} references (n={len(refs)}):  SCORE {total:.0f}/100")
    for k, v, med, rng, s in rows:
        if s is None:
            continue
        print(f"  {k:16s} {v!s:>8}   refs median {med:>7.2f}  range {rng[0]:.2f}..{rng[1]:.2f}   {s:5.0f}")
    med_curve = np.median(np.array([r["energy_curve"] for r in refs]), axis=0)
    print(f"  {'energy curve':16s} {m['energy_curve']}\n  {'':16s} refs {[round(float(c), 1) for c in med_curve]}   {cs:5.0f}")
    return total, m


if __name__ == "__main__":
    args = sys.argv[1:]
    for p, c in zip(args[::2], args[1::2]):
        score(p, c)
