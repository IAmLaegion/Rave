"""Separation through a StemDeck server (RAVE_STEMDECK_URL), which runs Demucs htdemucs_6s on a GPU:
seconds per song instead of minutes on a CPU.

The song goes up as FLAC (StemDeck takes mp3/wav/flac/m4a/ogg/opus, not webm), StemDeck splits it into
vocals, drums, bass, guitar, piano and other, and the stems come back as WAV. Guitar and piano are
folded into "other" for Rave's four lanes. The StemDeck job is deleted afterwards (Rave keeps its own
cache), so nothing piles up in StemDeck's library.
"""
import io
import json
import os
import tempfile
import time
import urllib.request
import uuid

import numpy as np
import soundfile as sf

from . import audio

URL = os.environ.get("RAVE_STEMDECK_URL", "").rstrip("/")
STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")


def available():
    if not URL:
        return False
    try:
        with urllib.request.urlopen(f"{URL}/api/health", timeout=3) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:
        return False


def _call(method, path, data=None, headers=None, timeout=60):
    req = urllib.request.Request(URL + path, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return body


def separate(y, progress=None, timeout=1800):
    """(vocals, drums, bass, other), each (2, n), for stereo audio `y`."""
    say = progress or (lambda m: None)
    buf = io.BytesIO()
    sf.write(buf, y.T, audio.SR, format="FLAC", subtype="PCM_24")
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"stems\"\r\n\r\n{json.dumps(list(STEMS))}\r\n"
            f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"rave-song.flac\"\r\n"
            f"Content-Type: audio/flac\r\n\r\n").encode() + buf.getvalue() + f"\r\n--{b}--\r\n".encode()
    say("Separating on the GPU (StemDeck)...")
    job = json.loads(_call("POST", "/api/jobs", body,
                           {"Content-Type": f"multipart/form-data; boundary={b}"}, timeout=300))
    jid = job.get("job_id") or job.get("id")
    t0 = time.time()
    try:
        while True:
            j = json.loads(_call("GET", f"/api/jobs/{jid}"))
            status = j.get("status")
            # The stems are final once StemDeck's post-processing has finished; what it does after that
            # (beat grid, and above all lyric transcription, minutes per song) is for its own player.
            if status == "done":
                break
            if "post" in (j.get("stage_timings") or {}):
                try:                                   # recorded a moment before the files are served
                    _call("HEAD", f"/api/jobs/{jid}/stems/other.wav")
                    break
                except Exception:
                    pass
            if status in ("failed", "error", "cancelled", "canceled", "stopped"):
                raise RuntimeError(f"StemDeck couldn't separate the song ({j.get('stage') or status}).")
            if time.time() - t0 > timeout:
                raise RuntimeError("StemDeck took too long to separate the song.")
            say(f"Separating on the GPU (StemDeck): {j.get('stage') or status} "
                f"{int(100 * float(j.get('progress') or 0))}%")
            time.sleep(1.5)
        stems = {}
        for s in STEMS:
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                f.write(_call("GET", f"/api/jobs/{jid}/stems/{s}.wav", timeout=300))
                f.flush()
                stems[s] = audio.load(f.name)
    finally:
        try:
            _call("POST", f"/api/jobs/{jid}/cancel", b"")
            for _ in range(40):                        # a cancelled job can only be deleted once it stops
                if json.loads(_call("GET", f"/api/jobs/{jid}")).get("status") in ("done", "error", "cancelled", "stopped"):
                    break
                time.sleep(0.5)
            _call("DELETE", f"/api/jobs/{jid}")
        except Exception:
            pass
    n = y.shape[1]
    fit = lambda s: s[:, :n] if s.shape[1] >= n else np.pad(s, ((0, 0), (0, n - s.shape[1])))
    other = fit(stems["other"]) + fit(stems["guitar"]) + fit(stems["piano"])
    return fit(stems["vocals"]), fit(stems["drums"]), fit(stems["bass"]), other.astype(np.float32)
