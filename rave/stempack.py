"""Stem packs: archives of a song's separate tracks (official stems, multitracks, a cappella +
instrumental), used instead of separating the song with Demucs.

An archive is unpacked with 7-Zip (zip, 7z, rar, tar...). Each folder holding audio is one song (a
pack with a single song in it is the usual case). Every file is sorted by its name into vocals,
drums, bass or other ("Lead Vox.wav", "Kick.wav", "808.wav", "Synth Pad.wav" ...); an instrumental
or a full mix is used to fill in whatever the pack doesn't have, by separating just that with Demucs.
The result goes straight into the stem cache under the song's file key, so the engine finds the stems
and never separates the song itself.
"""
import hashlib
import os
import re
import shutil
import subprocess

import numpy as np
import soundfile as sf

from . import audio, separate

ARCHIVE_EXT = (".zip", ".7z", ".rar", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".tbz2")
AUDIO_EXT = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".aif", ".aiff", ".wma")
MAX_UNPACKED = 6 * 1024 ** 3        # bytes
MAX_FILES = 400

# Checked in order; a file matching none is "other" (keys, guitars, synths, strings, fx ...).
ROLES = [
    ("inst", r"instrumental|\binst\b|karaoke|backing ?track|accompaniment|minus ?one|no ?vocals?|novox|"
             r"without vocals|music only"),
    ("vocals", r"acapp?ella|a ?cappella|acap\b|vocal|\bvox\b|\bvoc\b|\bvo\b|\bbgv|\bbvs?\b|choir|adlib|"
               r"ad ?lib|harmon|\blead ?vo|\bvx\b|singer|rap\b|verse ?vo"),
    ("drums", r"drum|kick|snare|hi ?hats?|\bhh\b|\bhats?\b|perc|clap|cymbal|\btoms?\b|\bride\b|crash|"
              r"shaker|tamb|\bsnaps?\b|\bloop\b|\bbeat\b|\bkit\b"),
    ("bass", r"bass|\bsub\b|808"),
    ("mix", r"master|full ?mix|\bmix\b|original|\bfull\b|stereo ?mix|\bsong\b"),
]


def is_archive(name):
    n = name.lower()
    return n.endswith(ARCHIVE_EXT)


def role_of(filename):
    n = re.sub(r"[_\-.]+", " ", os.path.splitext(os.path.basename(filename))[0].lower())
    for role, pattern in ROLES:
        if re.search(pattern, n):
            return role
    return "other"


def unpack(archive, dest):
    """Unpack `archive` into `dest`, refusing anything that would land outside it or is too big."""
    seven = shutil.which("7zz") or shutil.which("7z")
    if not seven:
        raise RuntimeError("7-Zip isn't installed in this image, so stem packs can't be unpacked.")
    listing = subprocess.run([seven, "l", "-slt", "-ba", archive], capture_output=True, text=True)
    if listing.returncode != 0:
        raise RuntimeError(f"Couldn't read {os.path.basename(archive)} as an archive.")
    sizes = [int(m) for m in re.findall(r"^Size = (\d+)", listing.stdout, re.M)]
    if sum(sizes) > MAX_UNPACKED or len(sizes) > MAX_FILES:
        raise RuntimeError(f"{os.path.basename(archive)} is too big to unpack (over 6 GB or 400 files).")
    os.makedirs(dest, exist_ok=True)
    p = subprocess.run([seven, "x", "-y", "-snl-", f"-o{dest}", archive], capture_output=True, text=True)
    if p.returncode not in (0, 1):                     # 1 = warnings
        raise RuntimeError(f"Couldn't unpack {os.path.basename(archive)}: {p.stderr.strip()[-300:]}")
    root = os.path.realpath(dest)
    for dirpath, _, files in os.walk(dest):
        for f in files:
            if not os.path.realpath(os.path.join(dirpath, f)).startswith(root + os.sep):
                raise RuntimeError("The archive tried to write outside its folder.")
    # One level of nesting: a zip of zips.
    for dirpath, _, files in list(os.walk(dest)):
        for f in files:
            if is_archive(f) and not f.lower().endswith(".gz"):
                inner = os.path.join(dirpath, f)
                unpack(inner, os.path.splitext(inner)[0])
                os.remove(inner)


def songs_in(folder, fallback_name):
    """[(song name, [audio files])]: every folder holding audio is one song."""
    groups = {}
    for dirpath, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith((".", "__MACOSX"))]
        audio_files = sorted(os.path.join(dirpath, f) for f in files
                             if f.lower().endswith(AUDIO_EXT) and not f.startswith("."))
        if audio_files:
            groups[dirpath] = audio_files
    if not groups:
        return []
    if len(groups) == 1:
        return [(fallback_name, next(iter(groups.values())))]
    # Folders that are clearly parts of one song ("Drums", "Vocals" ...) merge into their parent.
    merged = {}
    for d, files in groups.items():
        key = d
        while key != folder and role_of(os.path.basename(key)) != "other":
            key = os.path.dirname(key)
        merged.setdefault(key, []).extend(files)
    return [(os.path.basename(d.rstrip(os.sep)) or fallback_name, files) for d, files in merged.items()]


def build(name, files, cache_dir, progress=None):
    """Turn one song's stem files into a mix file plus cached stems; returns the mix's path."""
    say = progress or (lambda m: None)
    roles = {}
    for f in files:
        say(f"Reading stems: {name} ({os.path.basename(f)})")
        y = audio.load(f)
        r = role_of(f)
        roles.setdefault(r, []).append(y)
    n = max(y.shape[1] for ys in roles.values() for y in ys)
    pad = lambda y: np.pad(y, ((0, 0), (0, n - y.shape[1])))
    tot = {r: np.sum([pad(y) for y in ys], axis=0).astype(np.float32) for r, ys in roles.items()}

    parts = [tot[r] for r in ("vocals", "inst", "drums", "bass", "other") if r in tot]
    mix = tot.get("mix", np.sum(parts, axis=0) if parts else None)
    if mix is None:
        raise RuntimeError(f"No usable audio in the stems for {name}.")
    peak = float(np.abs(mix).max())
    if peak > 1.0:
        mix = mix / peak

    h = hashlib.sha1()
    for f in sorted(files):
        h.update(os.path.basename(f).encode())
        h.update(str(os.path.getsize(f)).encode())
    os.makedirs(os.path.join(cache_dir, "stems"), exist_ok=True)
    slug = re.sub(r"[^\w-]+", "_", name)[:60]
    mix_path = os.path.join(cache_dir, "stems", f"{slug}-{h.hexdigest()[:10]}.flac")
    sf.write(mix_path, np.clip(mix, -1, 1).T, audio.SR, subtype="PCM_24")
    key = separate.file_key(mix_path)
    out = {s: os.path.join(cache_dir, f"{key}.pack.{s}.flac") for s in ("vocals", "drums", "bass", "other")}
    if all(os.path.isfile(p) for p in out.values()):
        return mix_path

    stems = {s: tot.get(s) for s in ("vocals", "drums", "bass", "other")}
    missing = [s for s, y in stems.items() if y is None]
    if missing:
        # Fill the gaps by separating only what the pack gives as a whole: the instrumental for the
        # backing parts, the full mix for anything else.
        source = tot.get("inst") if "inst" in tot and "vocals" not in missing else tot.get("mix", tot.get("inst"))
        if source is not None and separate.demucs_available():
            say(f"Separating what the stems don't have: {name} ({', '.join(missing)})")
            v, d, b, o = separate._demucs(source, None)
            got = {"vocals": v, "drums": d, "bass": b, "other": o}
            for s in missing:
                if s == "vocals" and "inst" in tot and "mix" not in tot:
                    continue                           # an instrumental has no vocals to give
                stems[s] = got[s]
    for s, path in out.items():
        y = stems[s] if stems[s] is not None else np.zeros_like(mix)
        sf.write(path, np.clip(pad(y) if y.shape[1] < n else y[:, :n], -1, 1).T, audio.SR, subtype="PCM_24")
    return mix_path


def prepare(archive, work_dir, cache_dir, name=None, progress=None):
    """Unpack a stem pack and get every song in it ready: [(song name, mix path)]."""
    say = progress or (lambda m: None)
    name = name or os.path.splitext(os.path.basename(archive))[0]
    dest = os.path.join(work_dir, "unpacked")
    shutil.rmtree(dest, ignore_errors=True)
    say(f"Unpacking stems: {name}")
    unpack(archive, dest)
    songs = songs_in(dest, name)
    if not songs:
        raise RuntimeError(f"No audio files found in {name}.")
    out = [(song, build(song, files, cache_dir, say)) for song, files in songs]
    shutil.rmtree(dest, ignore_errors=True)
    return out
