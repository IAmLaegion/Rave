"""Lyrics: what's sung, when, and which line from one song carries on from a line of another.

Transcription runs on the separated vocal stem (much cleaner than the full mix), in two passes:

  * LocalAI's whisper-large (RAVE_LOCALAI_URL), when it's reachable, for the words themselves and
    where the sentences break. It's far more accurate on sung vocals, but only times whole segments.
  * The small faster-whisper bundled in the image, for a timestamp on every word.

The two are lined up word by word, so the accurate words get precise times. Without LocalAI, the
small model's words are used as they are. Transcripts are cached next to the stems.

The words are then cut into lines (at pauses and sentence starts), and the lyric planner hands over
between songs on lines that read on from each other: whatever chat model LocalAI has loaded picks the
continuation that makes the most sense, from a shortlist the rules below put together. If no model is
loaded (Rave never makes LocalAI load one: that takes minutes) or it's slow, the shortlist's best
pick is used.
"""
import difflib
import importlib.util
import json
import os
import re
import time
import urllib.request
import uuid

import numpy as np

from . import audio

# Words too common to make a handover mean anything.
STOP = set("""a an the and or but so i i'm im you you're he she it it's we they me my your our to of
in on at for with is am are was be been do don't just like that this oh ooh ohh yeah yeh ah ahh na
nah la hey uh huh mm whoa woah baby""".split())

# A line that ends on one of these is unfinished: the next line has to carry the sentence on.
CARRY = set("""and or but so to the a an my your our their his her of in on at for with from into
like than that if when cause 'cause because where while i you we they is are was be just""".split())

LOCALAI = os.environ.get("RAVE_LOCALAI_URL", "").rstrip("/")
GAP = 0.7           # seconds of silence between words that ends a line (held notes leave shorter gaps)

_model = None


def available():
    return importlib.util.find_spec("faster_whisper") is not None


def clean(word):
    return re.sub(r"[^a-z']", "", word.lower()).strip("'")


# --- Transcription -------------------------------------------------------------------------------

def transcribe(vocals, cache_dir=None, key=None, sr=audio.SR):
    """[(start_s, end_s, word, starts_line)] for the sung words in `vocals` (2, n), in order; [] when
    no Whisper is available."""
    path = os.path.join(cache_dir, f"{key}.words6.json") if cache_dir and key else None
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return [tuple(w) for w in json.load(f)]
    import librosa
    m = librosa.resample(audio.mono(vocals), orig_sr=sr, target_sr=16000).astype(np.float32)
    peak = float(np.abs(m).max())
    if peak < 1e-4:
        return []                                      # no vocals (an instrumental-only stem pack)
    m = m / peak * 0.9
    timed = _local_words(m)
    big = _localai_segments(m)
    words = _align(big, timed) if big else _mark_lines(timed)
    if path and (words or timed is not None):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(words, f)
    return words


def _local_words(m):
    """[(start, end, word)] from the bundled faster-whisper, with a timestamp per word; None if it
    isn't installed."""
    if not available():
        return None
    global _model
    from faster_whisper import WhisperModel
    if _model is None:
        _model = WhisperModel(os.environ.get("RAVE_WHISPER_MODEL", "small.en"), device="cpu",
                              compute_type="int8")
    # No voice-activity filter: it's tuned for speech and throws away processed, chopped-up pop vocals.
    # Instead, words "heard" where the vocal stem is near silent are dropped.
    segments, _ = _model.transcribe(m, word_timestamps=True, vad_filter=False, beam_size=5,
                                    condition_on_previous_text=False)
    env = _env(m)
    floor = 0.06 * float(np.percentile(env, 95))
    out = []
    for seg in segments:
        for w in seg.words or []:
            c = clean(w.word)
            if not c or w.end <= w.start or (w.probability is not None and w.probability < 0.2):
                continue
            a, b = int(w.start * 100), max(int(w.start * 100) + 1, int(w.end * 100))
            if env[a:b].size and env[a:b].mean() < floor:
                continue
            out.append((round(float(w.start), 3), round(float(w.end), 3), c))
    return out


def _env(m):
    hop = 160                                           # 10 ms at 16 kHz
    return np.sqrt(np.mean(np.square(m[:len(m) // hop * hop].reshape(-1, hop)), axis=1))


def _localai_segments(m):
    """[(start, end, text)] from LocalAI's whisper-large, or None if it isn't configured/reachable."""
    if not LOCALAI:
        return None
    import io
    import soundfile as sf
    fields = {"model": os.environ.get("RAVE_LOCALAI_WHISPER", "whisper-large"),
              "response_format": "verbose_json"}
    # Whisper falls into repetition loops on a whole song (one line "heard" over and over, a second
    # each, for minutes). Sending ~30 s chunks, cut where the singer is quietest, stops a loop from
    # snowballing; the odd repeat that still slips through is dropped below.
    sr, env = 16000, _env(m)
    cuts, at = [0], 30.0
    while at < len(m) / sr - 10:
        lo, hi = int((at - 4) * 100), int((at + 4) * 100)
        cuts.append((lo + int(np.argmin(env[lo:hi]))) / 100)
        at = cuts[-1] + 30.0
    cuts.append(len(m) / sr)
    segs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        buf = io.BytesIO()
        sf.write(buf, m[int(a * sr):int(b * sr)], sr, format="WAV", subtype="PCM_16")
        try:
            d = _post_multipart(f"{LOCALAI}/v1/audio/transcriptions", fields,
                                ("file", "vocals.wav", buf.getvalue()), timeout=300)
        except Exception:
            return None
        for s in d.get("segments") or []:
            s0, s1, text = a + float(s.get("start", 0)), a + float(s.get("end", 0)), s.get("text", "")
            if not text.strip():
                continue
            if any(text.strip() == x[2].strip() for x in segs[-2:]) and s1 - s0 < 1.5:
                continue                               # a repetition loop
            segs.append((min(s0, b), min(s1, b), text))
    return segs or None


def _post_multipart(url, fields, file, timeout):
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    name, fname, data = file
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{fname}\"\r\n"
             f"Content-Type: audio/wav\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _align(segments, timed):
    """Combine the two transcripts. The small model's words keep their timing exactly as heard (its
    timestamps are the reliable part); whisper-large marks where its sentences start, and fixes the
    spelling of words where the two agree on how many words were sung. Words only whisper-large heard
    are left out: guessing their times put words in the wrong place (repeated choruses especially)."""
    big = []                                            # (word, starts a sentence)
    for s, e, text in segments:
        toks = text.split()
        for k, tok in enumerate(toks):
            c = clean(tok)
            if c:
                starts = k == 0 or (tok[:1].isupper() and c != "i" and not c.startswith("i'")) or                     toks[k - 1].endswith((".", "!", "?", ","))
                big.append((c, starts))
    if not timed:
        return []
    words = [list(w) + [False] for w in timed]
    sm = difflib.SequenceMatcher(None, [w for _, _, w in timed], [w for w, _ in big], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                words[i1 + k][2] = big[j1 + k][0]
                words[i1 + k][3] = big[j1 + k][1]
        elif op in ("replace", "insert") and i1 < len(words):
            if any(st for _, st in big[j1:j2]):
                words[i1][3] = True
    out = []
    for k, (a, b, w, starts) in enumerate(words):
        if k and a - words[k - 1][1] > GAP:
            starts = True
        out.append((a, b, w, bool(starts or k == 0)))
    return out


def _mark_lines(timed):
    """Line starts for words without sentence information: after a pause."""
    if not timed:
        return []
    out = []
    for k, (a, b, w) in enumerate(timed):
        out.append((a, b, w, k == 0 or a - timed[k - 1][1] > GAP))
    return out


# --- Matching ------------------------------------------------------------------------------------

def _rhyme_key(word):
    try:
        import pronouncing
        phones = pronouncing.phones_for_word(word)
        if phones:
            return pronouncing.rhyming_part(phones[0])
    except ImportError:
        pass
    return word[-3:] if len(word) >= 4 else None


def link(a, b):
    """How well word b answers word a: 1 for the same word, 0.6 for a rhyme, else 0. Filler words
    ("the", "oh", "baby") don't count."""
    if not a or not b or a in STOP or b in STOP:
        return 0.0
    if a == b:
        return 1.0
    ka, kb = _rhyme_key(a), _rhyme_key(b)
    return 0.6 if ka and ka == kb else 0.0


def carries_on(prev, nxt):
    """Rule-based score for how well line `nxt` (a list of words) carries on from line `prev`: an
    unfinished line wants a continuation; the same word or a rhyme across the join; a shared topic."""
    if not prev or not nxt:
        return 0.0
    score = 0.0
    if prev[-1] in CARRY:
        score += 0.6
    score += max(link(prev[-1], w) for w in nxt[:2])
    shared = {w for w in prev if w not in STOP} & {w for w in nxt if w not in STOP}
    score += 0.15 * min(3, len(shared))
    return score


class Judge:
    """Asks LocalAI's already-loaded chat model which line continues best. Never makes LocalAI load a
    model; gives up for the rest of the mashup after `budget` seconds or the first failure."""

    def __init__(self, budget=30.0, per_call=8.0):
        self.model = _loaded_chat_model()
        self.budget, self.per_call = budget, per_call
        self.spent = 0.0
        self.calls = 0

    @property
    def active(self):
        return bool(self.model) and self.spent < self.budget

    def pick(self, prev_text, options):
        """Index into `options` (texts) of the line that best carries on `prev_text`, or None."""
        if not self.active or len(options) < 2:
            return None
        listing = "\n".join(f"{k + 1}. \"{o}\"" for k, o in enumerate(options))
        prompt = (f"Mashup lyric handoff. The last line sung was:\n\"{prev_text}\"\n\n"
                  f"Which of these lines, sung next by a different singer, continues it most "
                  f"naturally, so the two read as one sentence or a sensible reply?\n{listing}\n\n"
                  f"Answer with the number only.")
        t0 = time.time()
        try:
            req = urllib.request.Request(
                f"{LOCALAI}/v1/chat/completions", method="POST",
                headers={"Content-Type": "application/json"},
                data=json.dumps({"model": self.model, "temperature": 0, "max_tokens": 6,
                                 "reasoning_effort": "none",
                                 "messages": [{"role": "system", "content":
                                               "You arrange lyrics for song mashups. Reply with a "
                                               "single number."},
                                              {"role": "user", "content": prompt}]}).encode())
            with urllib.request.urlopen(req, timeout=self.per_call) as r:
                text = json.loads(r.read().decode())["choices"][0]["message"]["content"] or ""
        except Exception:
            self.model = None                           # don't keep waiting on a failing server
            return None
        finally:
            self.spent += time.time() - t0
            self.calls += 1
        m = re.search(r"\d+", re.sub(r"<think>.*?</think>", "", text, flags=re.S))
        k = int(m.group()) - 1 if m else -1
        return k if 0 <= k < len(options) else None


def _loaded_chat_model():
    """The chat model LocalAI currently has in memory (llama.cpp backend), or None."""
    if not LOCALAI:
        return None
    try:
        with urllib.request.urlopen(f"{LOCALAI}/system", timeout=3) as r:
            loaded = json.loads(r.read().decode()).get("loaded_models") or []
    except Exception:
        return None
    for m in loaded:
        if "llama" in str(m.get("backend", "")):
            return m.get("id")
    return None
