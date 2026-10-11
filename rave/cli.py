"""Command line: python -m rave song1.mp3 song2.mp3 [...] -o mashup.mp3"""
import argparse
import os
import sys

from . import engine, separate, styles


def main(argv=None):
    p = argparse.ArgumentParser(prog="rave",
                                description="Make a Pitch Perfect-style mashup of two or more songs.")
    p.add_argument("songs", nargs="+", help="audio files (mp3, wav, flac, m4a, ...)")
    p.add_argument("-o", "--out", default="mashup.mp3", help="output file (.mp3, .wav, .flac, .m4a)")
    p.add_argument("--style", choices=list(styles.STYLES), default="earworm",
                   help="arrangement engine: " + ", ".join(f"{k} ({v.name})" for k, v in styles.STYLES.items()))
    p.add_argument("--seed", type=int, default=None, help="random seed for engines that use one (ravedj)")
    p.add_argument("--8d", dest="eightd", action="store_true", help="binaural 8D audio (headphones)")
    p.add_argument("--nightcore", type=float, default=0.0, metavar="SPEED", help="nightcore speed-up, e.g. 1.22")
    p.add_argument("--base", type=int, default=0,
                   help="1-based number of the song whose beat and key everyone follows (default: auto)")
    p.add_argument("--length", type=float, default=0, help="target length in seconds (default: the style's own)")
    p.add_argument("--phrase", type=int, default=8, choices=(4, 8, 16), help="bars per vocal turn")
    p.add_argument("--bed", choices=("base", "rotate"), default="base",
                   help="keep the base song's backing throughout, or rotate the backings too")
    p.add_argument("--separation", choices=separate.METHODS, default="auto")
    p.add_argument("--no-key-match", action="store_true", help="don't transpose songs to the base key")
    p.add_argument("--transpose", action="append", default=[], metavar="N:SEMITONES",
                   help="force song N (1-based) up/down by SEMITONES, e.g. --transpose 2:-3")
    p.add_argument("--no-drop", action="store_true", help="no a cappella bar at the start of the finale")
    p.add_argument("--no-riser", action="store_true")
    a = p.parse_args(argv)

    missing = [s for s in a.songs if not os.path.isfile(s)]
    if missing:
        p.error("not found: " + ", ".join(missing))
    transpose = {}
    for t in a.transpose:
        n, _, st = t.partition(":")
        transpose[int(n) - 1] = int(st)
    opts = engine.Options(style=a.style, seed=a.seed, spatial="8d" if a.eightd else "", nightcore=a.nightcore, base=a.base - 1, length=a.length, phrase_bars=a.phrase, bed=a.bed,
                          separation=a.separation, match_key=not a.no_key_match, transpose=transpose,
                          acappella_drop=not a.no_drop, riser=not a.no_riser)

    def progress(frac, msg):
        print(f"[{frac * 100:5.1f}%] {msg}", flush=True)

    report = engine.make_mashup(a.songs, a.out, opts, progress)
    print()
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
