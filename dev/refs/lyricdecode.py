"""Decode a mashup's vocals by their lyrics: transcribe the mashup's vocal stem and every source's
(Whisper, cached), then match each sung line of the mashup to the source line it was cut from.

    python lyricdecode.py <ref name> [--original]

Prints a vocal cue sheet (time, length, song, where in the song, the words) and writes
<ref>.lyrics.json. Lyrics are far more telling than melody across twenty pop songs, so this is the
vocal decode to trust; decode.py's audio match is the fallback for wordless parts.
"""
import difflib
import json
import os
import sys

sys.path.insert(0, "/dev-scripts/refs")
from rave import audio, lyrics, separate  # noqa: E402

GAP = 0.45            # seconds between words that ends a line


def words_of(path):
    key = separate.file_key(path)
    method = "demucs" if os.path.isfile(f"/data/cache/{key}.demucs.vocals.flac") else \
        os.environ.get("RAVE_SEP", "demucs")
    y = audio.load(path)
    v, _, _, _ = separate.separate(y, method, "/data/cache", key)
    return lyrics.transcribe(v, "/data/cache", f"{key}.{method}")


def lines_of(words):
    out, cur = [], []
    for w in words:
        if cur and (w[0] - cur[-1][1] > GAP or (len(w) > 3 and w[3] and len(cur) >= 3)):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def best_match(line, src_words):
    """(ratio, start index) of the stretch of `src_words` that best matches the words of `line`."""
    target = [w[2] for w in line]
    n = len(target)
    seq = [w[2] for w in src_words]
    best = (0.0, -1)
    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq2(target)
    for size in {max(1, n - 1), n, n + 1}:
        for i in range(0, max(1, len(seq) - size + 1)):
            sm.set_seq1(seq[i:i + size])
            if sm.real_quick_ratio() <= best[0] or sm.quick_ratio() <= best[0]:
                continue
            r = sm.ratio()
            if r > best[0]:
                best = (r, i)
    return best


def main():
    name = sys.argv[1]
    use_orig = "--original" in sys.argv
    man = json.load(open("/dev-scripts/refs/manifest.json"))[name]
    ref_path = man["original_path" if use_orig else "video_path"]
    print(f"Transcribing the reference vocals ({os.path.basename(ref_path)})", flush=True)
    ref_words = words_of(ref_path)
    srcs = {}
    for song, info in man["songs"].items():
        srcs[song] = words_of(info["path"])
        print(f"  {song}: {len(srcs[song])} words", flush=True)
    rows = []
    for line in lines_of(ref_words):
        if len(line) < 2:
            continue
        best = (0.0, None, -1)
        for song, sw in srcs.items():
            r, i = best_match(line, sw)
            if r > best[0]:
                best = (r, song, i)
        r, song, i = best
        row = dict(t=line[0][0], end=line[-1][1], text=" ".join(w[2] for w in line), ratio=round(r, 2))
        if song and r >= 0.55:
            row.update(song=song, src_t=srcs[song][i][0])
        rows.append(row)
    print(f"\nReference: {len(ref_words)} words in {len(rows)} lines\n", flush=True)
    for row in rows:
        t = row["t"]
        who = f"{row['song'][:40]:40s} @{row['src_t']:6.1f}s" if "song" in row else f"{'?':40s}        "
        print(f"  {int(t // 60)}:{t % 60:05.2f}  {row['end'] - t:4.1f}s  {who}  {row['ratio']:.2f}  {row['text'][:60]}")
    known = [r for r in rows if "song" in r]
    print(f"\nMatched {len(known)}/{len(rows)} lines to a source song.")
    suffix = ".original" if use_orig else ""
    json.dump(rows, open(f"/dev-scripts/refs/{name}{suffix}.lyrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
