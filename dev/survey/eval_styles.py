"""Render a song set with each style (and the old Earworm engine for comparison), then score every
render against the reference corpus.

    python eval_styles.py [style ...]      default: earworm_old earworm riffoff megamix
"""
import json
import sys
import time

sys.path.insert(0, "/dev-scripts/survey")
from rave import engine, styles  # noqa: E402
from score import score  # noqa: E402

SET = ["Justin Timberlake - Can't Stop The Feeling", "Flo Rida - My House", "Lukas Graham - 7 Years",
       "Beyonce - Formation", "Fifth Harmony - Work from Home"]
CATEGORY = {"earworm_old": "earworm", "earworm": "earworm", "riffoff": "pitchperfect", "megamix": "megamix"}

styles.STYLES["earworm_old"] = styles.Style("earworm_old", "DJ Earworm (old engine)", "", styles.earworm,
                                            key_mode="least_damage")
man = json.load(open("/dev-scripts/refs/manifest.json"))["summermash16"]["songs"]
paths = [man[s]["path"] for s in SET]
results = {}
for key in sys.argv[1:] or ["earworm_old", "earworm", "riffoff", "megamix"]:
    out = f"/dev-scripts/survey/eval-{key}.mp3"
    t0 = time.time()
    opts = engine.Options(style=key, separation="demucs", cache_dir="/data/cache", seed=7)
    report = engine.make_mashup(paths, out, opts, names=[s.split(" - ")[1] for s in SET])
    print(f"\n### {key} ({time.time() - t0:.0f}s)\n" + report.split("Arrangement:")[1][:1800], flush=True)
    cat = CATEGORY[key]
    try:
        total, m = score(out, cat)
    except SystemExit as e:
        print(e)
        try:
            total, m = score(out, "earworm")
            cat = "earworm"
        except SystemExit:
            continue
    results[key] = (cat, round(total))
print("\nSCORES:", results)
