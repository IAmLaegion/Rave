"""The Rave engine and its styles. One engine (versatile) arranges every mashup; a style is a preset
of how it works, learned from the reference mashups:

  earworm        DJ Earworm: one song's instrumental all the way through, a collage of lyric lines from
                 all the songs on top, a refrain that keeps coming back, and a final chorus that builds
                 up voice by voice, in the key that does the least damage to everyone.
  riffoff        Pitch Perfect riff-off: lines handed from song to song, then quick cut-ins over just the
                 beat, an a cappella hit, and every chorus at once.
  megamix        Pop Danthology / minlee: one beat, a one- or two-bar hook from every song in turn, the
                 best hooks coming back again and again.

and three timeline styles, where the base song plays through as itself:

  swap           Vocal swap (Spotify's mashup patent): its vocals are kept, swapped for another song's,
                 or added where it has none, with low-pass sweeps on the changes.
  automashupper  AutoMashUpper (Davies et al. 2014): each phrase gets the most "mashable" phrase of
                 another song layered on, as an instrumental layer under the singing or as vocals in the
                 gaps.
  ravedj         rave.dj: the first song is the timeline, and a search decides in every 4 bars whether its
                 own vocal, a guest vocal or nobody sings. Each run uses a new random seed.

  classic        The earlier DJ Earworm arrangement (4-bar lines traded over per-section harmony), kept for
                 comparison.
"""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from . import audio, lyrics
from .engine import Part, fit


@dataclass
class Style:
    key: str
    name: str
    blurb: str
    arrange: callable
    key_mode: str = "base"          # "base" or "least_damage" (see engine._plan_keys)
    base_first: bool = False        # the first song in the lineup is the base unless one is starred


class Ctx:
    """What an arranger works with, plus the bookkeeping they share."""

    def __init__(self, tracks, base, opts, bar_len):
        self.tracks, self.base, self.opts, self.bar_len = tracks, base, opts, bar_len
        self.B = tracks[base]
        self.n = len(tracks)
        self.order = [base] + [i for i in range(self.n) if i != base]
        length = opts.length if opts.length and opts.length > 0 else 150.0      # 0 = the style's own
        self.total = max(24, int(round(length * audio.SR / bar_len)))
        self.used = defaultdict(int)                   # (track, bar) -> times used
        self.seed = opts.seed if getattr(opts, "seed", None) is not None \
            else int(np.random.default_rng().integers(1 << 31))
        self.rng = np.random.default_rng(self.seed)
        self.lines = {i: Lines(t) for i, t in enumerate(tracks)}
        self.used_lines = {}

    def room(self, i, s, bars):
        return 0 <= s and s + bars <= self.tracks[i].nbars() - 1

    def reuse(self, i, s, bars):
        return sum(self.used[(i, s + k)] for k in range(bars)) / max(1, bars)

    def mark(self, i, s, bars):
        for k in range(bars):
            self.used[(i, s + k)] += 1

    def positions(self, i, bars, step=4):
        t = self.tracks[i]
        return sorted({g + o for g in t.grid for o in range(0, 8, step) if self.room(i, g + o, bars)}
                      or ({0} if self.room(i, 0, bars) else set()))

    def best_backing(self, vocals, bars, pool, prefer=None):
        """The backing passage (track, bar) whose chords best fit the vocal lines [(singer, bar, offset,
        length)] sung over it. A singer's own song only counts in sync with them. Passages already used
        cost a little (Spotify's horizontal score), as does leaving the backing that was playing."""
        best = None
        for i in pool:
            t = self.tracks[i]
            own = [v for v in vocals if v[0] == i]
            cands = [own[0][1] - own[0][2]] if own else self.positions(i, bars)
            for c in cands:
                if not self.room(i, c, bars):
                    continue
                score = np.mean([fit(self.tracks[s], vp, t, c + off, ln) for s, vp, off, ln in vocals]) \
                    if vocals else 0.0
                score += (0.1 if own else 0.0) - 0.15 * self.reuse(i, c, bars) + \
                    (0.08 if prefer == (i, c) else 0.0)
                if best is None or score > best[0]:
                    best = (score, i, c)
        if best is None:
            return self.base, self.B.grid[0]
        return best[1], best[2]

    def intro(self):
        """The base song's bars leading into its first verse: (start bar, length)."""
        v0 = self.lines[self.base].verses[0]
        bars = int(min(8, max(4, v0 - v0 % 4)))
        return max(0, v0 - bars), bars

    # Lyrics: hand over on the word the last singer ended on, or a rhyme (riff-off rules).
    def last_word(self, i, end_bar):
        """The last meaningful word sung before bar `end_bar` of track i ends its turn."""
        t = self.tracks[i]
        if not t.words or not 0 <= end_bar < len(t.bars):
            return None
        e = t.bars[end_bar]
        # A line's last word often lands on the next downbeat ("...set this house on | fire").
        ws = [w for s, en, w in t.words if e - self.bar_len < s < e + self.bar_len / 4]
        return next((w for w in reversed(ws) if w not in lyrics.STOP), None)

    def first_words(self, i, start_bar, n=2):
        """The first meaningful words of a turn starting at bar `start_bar` (pickup included)."""
        t = self.tracks[i]
        if not t.words or not 0 <= start_bar < len(t.bars):
            return []
        a = t.bars[start_bar]
        ws = [w for s, en, w in t.words if a - self.bar_len / 4 <= s < a + self.bar_len / 2]
        return [w for w in ws if w not in lyrics.STOP][:n]

    def link(self, prev, i, start_bar):
        """(score, word): how well a turn of track i at `start_bar` answers the previous turn
        `prev` = (track, end bar). 1 = same word, 0.6 = rhyme."""
        if prev is None:
            return 0.0, None
        pw = self.last_word(*prev)
        best = (0.0, None)
        for w in self.first_words(i, start_bar):
            s = lyrics.link(pw, w)
            if s > best[0]:
                best = (s, w if s == 1.0 else f"{pw}/{w}")
        return best

    def openers(self, j):
        """Every meaningful word that can open a passage of track j (cached)."""
        if not hasattr(self, "_openers"):
            self._openers = {}
        if j not in self._openers:
            self._openers[j] = {w for s in range(self.tracks[j].nbars()) for w in self.first_words(j, s)}
        return self._openers[j]

    def answerable(self, i, end_bar, j):
        """Can track j answer the line track i ends at `end_bar` (same word 1, rhyme 0.6)?"""
        w = self.last_word(i, end_bar)
        return max((lyrics.link(w, o) for o in self.openers(j)), default=0.0) if w else 0.0

    def handoff(self, prev, i, bars, P=8, nxt=None):
        """Track i's next turn of `bars` bars, preferring one that answers the previous turn's last
        word, and (looking one turn ahead) one that ends on a word the next singer `nxt` can answer.
        Returns (bar, label suffix)."""
        def bonus(p):
            b = self.link(prev, i, p)[0] if prev else 0.0
            return b + (self.answerable(i, p + bars, nxt) if nxt is not None else 0.0)
        v = self.lines[i].take(bars, P, bonus=bonus if (prev or nxt is not None) else None)
        score, word = self.link(prev, i, v)
        return v, (f' (on "{word}")' if score >= 1.0 else f' (rhymes {word})' if score > 0 else "")

    @property
    def judge(self):
        if not hasattr(self, "_judge"):
            self._judge = lyrics.Judge()
            self.judged = 0
        return self._judge

    def line_candidates(self, i, max_bars):
        """Track i's lyric lines still unused that fit in `max_bars` bars, with clean edges."""
        t = self.tracks[i]
        used = self.used_lines.setdefault(i, set())
        lines = self.lines[i]
        out = []
        for L in t.lines or []:
            n = L["b1"] - L["b0"]
            if n > max_bars or L["b0"] in used or len(L["words"]) < 2 or L["text"] in lines.sung:
                continue
            if L.get("edge", 0.0) > 1.0:
                continue                               # cut would land in full voice either side
            out.append(L)
        return out

    def use_line(self, i, L):
        self.used_lines.setdefault(i, set()).update(range(L["b0"], L["b1"]))
        self.lines[i].sung.add(L["text"])

    def stack(self, offset=0, pans=None):
        """Every song's chorus at once, spread across the stereo field."""
        pans = np.linspace(-self.opts.stack_pan, self.opts.stack_pan, self.n) if pans is None else pans
        g = 1.25 / np.sqrt(self.n)
        return [(i, self.tracks[i].chorus + offset, g, float(pans[k])) for k, i in enumerate(self.order)]

    def chorus_again(self, i):
        """Where a second pass of the chorus picks up: the rest of it if it carries on, else the top."""
        t = self.tracks[i]
        nxt = t.chorus + 8
        return nxt if nxt + 8 <= len(t.active) and t.active[nxt:nxt + 8].mean() >= 0.5 else t.chorus

    def outro_bar(self, after):
        B = self.B
        return after if self.room(self.base, after, 8) else self.intro()[0]


class Lines:
    """A song's sung passages in order, handed out a few bars at a time. Verses come first; choruses
    are kept back for the finale. Each turn is nudged (up to a phrase ahead) to start and end on bar
    lines the singer isn't singing straight through, so turns are whole lines rather than chopped ones."""

    def __init__(self, t):
        self.t = t
        self.verses = [g for g in t.phrases if g not in t.choruses] or list(t.phrases)
        span = (t.grid[1] - t.grid[0]) if len(t.grid) > 1 else 8
        self.bars = [g + k for g in self.verses for k in range(span) if g + k < t.nbars()]
        self.cur = 0
        self.taken = set()
        self.sung = set()                                # lyrics already used, by their first words

    def _clean(self, b):
        a = self.t.across
        return float(a[b]) if 0 <= b < len(a) else 1.0

    def take(self, bars, P=8, bonus=None):
        """The next `bars` bars to sing. With `bonus` (a score for a start bar, e.g. how well its first
        word answers the last singer) the whole song is searched, nearer and unused passages first."""
        n = len(self.bars)
        best = None
        for step in range(n if bonus else min(P, n)):
            i = (self.cur + step) % n
            if i + bars > n or self.bars[i + bars - 1] != self.bars[i] + bars - 1:
                continue
            p = self.bars[i]
            clean = self._clean(p) + self._clean(p + bars)
            cost = clean + (0.004 if bonus else 0.04) * step
            if bonus:
                cost += 0.5 * any(b in self.taken for b in range(p, p + bars))
                if clean < 0.8:                          # a lyric link never buys a messy cut
                    cost -= 1.5 * bonus(p)
            if self._text(p, bars) in self.sung:
                cost += 0.6                              # the same words again, elsewhere in the song
            if best is None or cost < best[0]:
                best = (cost, i)
        i = best[1] if best else self.cur % max(1, n)
        self.cur = (i + bars) % max(1, n)
        if n:
            self.taken.update(range(self.bars[i], self.bars[i] + bars))
            text = self._text(self.bars[i], bars)
            if text:
                self.sung.add(text)
        return self.bars[i] if n else self.t.phrases[0]

    def _text(self, p, bars):
        """The first few words of the passage at bar p, to spot a lyric that repeats."""
        t = self.t
        if not t.words or p + bars >= len(t.bars):
            return ""
        a, b = t.bars[p], t.bars[p + bars]
        return " ".join([w for s, e, w in t.words if a <= s < b][:5])

    def peek(self):
        return self.bars[self.cur % len(self.bars)] if self.bars else self.t.phrases[0]


# --- DJ Earworm --------------------------------------------------------------------------------------

def earworm(c):
    """A new pop song: intro, verse, chorus, verse, chorus, bridge, final chorus, outro. The base's drums
    drive the whole thing; each 4 bars the harmony comes from whichever song's chords fit the voice
    best, and another song's instrumental hook sits on top. Verses trade 4-bar lines between songs; each
    chorus is a different song's; the bridge drops the drums; the final chorus brings the choruses in
    one at a time and ends with all of them."""
    long = c.total >= 72
    plan = [("Verse 1", 16), ("Chorus 1", 8)] + \
        ([("Verse 2", 16 if c.total >= 96 else 8), ("Chorus 2", 8)] if long else []) + [("Bridge", 8)]
    if c.total < 64:
        plan = [("Verse 1", 8), ("Chorus 1", 8), ("Bridge", 8)]
    start, intro = c.intro()
    st = {"drum": start, "hook": 0, "harm": None, "singer": 0}

    def drums(bars):
        pos = st["drum"] if c.room(c.base, st["drum"], bars) else c.B.grid[0]
        st["drum"] = pos + bars
        return pos

    def hooks_over(harm_i, harm_pos, skip, bars, gain=0.45):
        """The next song's hook (not the harmony's or the singer's), if its notes sit on the harmony."""
        for _ in range(c.n):
            j = c.order[st["hook"] % c.n]
            st["hook"] += 1
            if j in skip or j == harm_i:
                continue
            t, h = c.tracks[j], c.tracks[harm_i]
            a = t.ichroma[t.hook:t.hook + 4]
            b = h.ichroma[harm_pos:harm_pos + 4]
            if len(a) == 4 and len(b) == 4:
                cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
                if cos.mean() >= 0.65:
                    pan = 0.35 if st["hook"] % 2 else -0.35
                    return [(j, t.hook + k * 4, gain, pan) for k in range(bars // 4)]
        return []

    def block(label, bars, vocals, drop_drums=False, sweep=False, riser=False):
        """`bars` (a multiple of 4) of music under `vocals` [(singer, bar, gain, pan)], 4 bars at a time."""
        out = []
        for k in range(0, bars, 4):
            vs = [(s, v + k, g, p) for s, v, g, p in vocals]
            lead = [(s, v, 0, 4) for s, v, g, p in vs if g >= 0.99] or [(s, v, 0, 4) for s, v, g, p in vs]
            singers = {s for s, *_ in vs}
            pool = [i for i in range(c.n) if i not in singers or i == c.base] or list(range(c.n))
            i, pos = c.best_backing(lead, 4, pool, prefer=st["harm"])
            c.mark(i, pos, 4)
            st["harm"] = (i, pos + 4)
            p = Part(f"{label}", 4, vocals=vs, sweep=sweep and k == 0, riser=riser and k == 0)
            if drop_drums:
                p.harm = [(i, pos, 1.0)]
            elif i == c.base:
                p.bed = [(i, pos)]
                st["drum"] = pos + 4
            else:
                p.harm = [(i, pos, 1.0)]
                p.drums = [(c.base, drums(4))]
            p.hooks = hooks_over(i, pos, singers, 4)
            out.append(p)
        return out

    parts = [Part("Intro", intro, bed=[(c.base, start)])]
    st["drum"] = start + intro
    if intro >= 8:
        parts[0] = Part("Intro", 4, bed=[(c.base, start)])
        parts.append(Part("Intro + hook", intro - 4, bed=[(c.base, start + 4)],
                          hooks=hooks_over(c.base, start + 4, {c.base}, intro - 4)))
    chorus_k = 1                                         # the base saves its chorus for the end
    for label, bars in plan:
        if label.startswith("Verse"):
            for k in range(0, bars, 4):                  # 4-bar lines, a different voice each time,
                s = c.order[st["singer"] % c.n]          # each answering the last line's final word
                st["singer"] += 1
                v, on = c.handoff(st.get("prev"), s, 4, nxt=c.order[st["singer"] % c.n])
                st["prev"] = (s, v + 4)
                parts += block(f"{label}: {c.tracks[s].name}{on}", 4, [(s, v, 1.0, 0.0)])
        elif label.startswith("Chorus"):
            s = c.order[chorus_k % c.n]
            chorus_k += 1
            parts += block(f"{label}: {c.tracks[s].name}", 8, [(s, c.tracks[s].chorus, 1.0, 0.0)])
        else:                                            # bridge: no drums, a fresh voice
            s = c.order[st["singer"] % c.n]
            st["singer"] += 1
            parts += block(f"{label}: {c.tracks[s].name}", 8, [(s, c.lines[s].take(8), 1.0, 0.0)],
                           drop_drums=True)

    # Final chorus: the base's chorus leads, the others come in one every 2 bars, then all of them.
    lead = c.base
    g = 1.25 / np.sqrt(c.n)
    pans = np.linspace(-c.opts.stack_pan, c.opts.stack_pan, c.n)
    others = [i for i in c.order if i != lead]
    for k in range(0, 8, 2):
        vs = [(lead, c.B.chorus + k, 1.0, 0.0)]
        vs += [(i, c.tracks[i].chorus + k, g, float(pans[m + 1])) for m, i in enumerate(others)
               if m < k // 2 + 1]
        parts.append(Part(f"Final chorus: {len(vs)} voices", 2, bed=[(c.base, c.B.chorus + k)],
                          vocals=vs, sweep=k == 0, riser=k == 0))
    again = c.chorus_again(c.base)
    vs = [(lead, again, 1.0, 0.0)] + [(i, c.chorus_again(i), g, float(pans[m + 1]))
                                       for m, i in enumerate(others)]
    p = Part("Final chorus: everyone", 8, bed=[(c.base, again)], vocals=vs)
    p.hooks = hooks_over(c.base, again, set(), 8, gain=0.4)
    parts.append(p)
    out_at = c.outro_bar(again + 8)
    parts.append(Part("Outro", 8, bed=[(c.base, out_at)], hooks=hooks_over(c.base, out_at, set(), 8),
                      fade=True))
    return parts


# --- AutoMashUpper -----------------------------------------------------------------------------------

def mashability(c, s, j, h, bars=4):
    """Davies et al.'s mashability of song j's bars h.. against the base's bars s..: harmonic match
    weighs most, then rhythmic similarity, then spectral balance (better when the two fill different
    bands), less a penalty for reusing a passage."""
    B, t = c.B, c.tracks[j]
    a = B.ichroma[s:s + bars] + B.vchroma[s:s + bars]
    b = t.ichroma[h:h + bars]
    harm = ((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)).mean()
    rhythm = (B.rhythm[s:s + bars] * t.rhythm[h:h + bars]).sum(1).mean()
    overlap = np.minimum(B.bands[s:s + bars], t.bands[h:h + bars]).sum(1).mean()
    return 0.6 * harm + 0.25 * rhythm + 0.15 * (1.0 - overlap) - 0.2 * c.reuse(j, h, bars)


def automashupper(c):
    """The base song plays from its intro, as itself. Every 4 bars, the most mashable 4 bars of another
    song go on top: its instrumental (no bass, so there's one bass line) while the base is singing, or
    its vocals where the base isn't, so two voices never sing at once."""
    start, _ = c.intro()
    length = min(c.total, c.B.nbars() - 1 - start)
    length -= length % 4
    parts = []
    others = [i for i in c.order if i != c.base]
    for s in range(start, start + length - 4, 4):
        sings = c.B.active[s:s + 4].mean() >= 0.5
        best = None
        for j in others:
            t = c.tracks[j]
            for h in c.positions(j, 4):
                if not sings and t.active[h:h + 4].mean() < 0.5:
                    continue
                m = mashability(c, s, j, h)
                if not sings:
                    m = 0.5 * m + 0.5 * fit(t, h, c.B, s, 4)
                if best is None or m > best[0]:
                    best = (m, j, h)
        p = Part("Base", 4, bed=[(c.base, s)])
        if sings:
            p.vocals = [(c.base, s, 1.0, 0.0)]
        if best:
            _, j, h = best
            c.mark(j, h, 4)
            if sings:
                p.hooks = [(j, h, 0.55, 0.0)]
                p.label = f"{c.B.name} + {c.tracks[j].name} (instrumental)"
            else:
                p.vocals = [(j, h, 1.0, 0.0)]
                p.label = f"{c.B.name} backing + {c.tracks[j].name} vocals"
        parts.append(p)
    end = start + len(parts) * 4
    parts.append(Part("Outro", 4, bed=[(c.base, end)], fade=True))
    return _merge(parts)


# --- Spotify vocal swap ------------------------------------------------------------------------------

def swap(c):
    """The base's backing plays through as itself. Phrase by phrase: where the base has no vocals, add
    the best-fitting vocals from another song; where it does, alternate between keeping them and
    swapping in another song's (a chorus over the base's chorus where there's one). Each change of
    who's singing opens with a low-pass sweep."""
    start, intro = c.intro()
    length = min(c.total, c.B.nbars() - 1 - start)
    others = [i for i in c.order if i != c.base]
    parts = [Part("Intro", intro, bed=[(c.base, start)])]
    s, last, kept = start + intro, None, 0
    while s + 8 <= start + length - 4:
        frac = c.B.active[s:s + 8].mean()
        in_chorus = any(abs(s - g) < 4 for g in c.B.choruses)
        best = None
        for j in others:
            t = c.tracks[j]
            cands = sorted(t.choruses) if in_chorus and t.choruses else \
                [g for g in t.phrases if c.room(j, g, 8)]
            for h in cands:
                if not c.room(j, h, 8):
                    continue
                f = fit(t, h, c.B, s, 8) - 0.2 * c.reuse(j, h, 8) + (0.1 if in_chorus else 0.0)
                if best is None or f > best[0]:
                    best = (f, j, h)
        own = fit(c.B, s, c.B, s, 8)
        if frac < 0.1:
            label = "add"
        elif best and last != "swap" and best[0] >= own - 0.15:
            label = "swap"
        else:
            label = "keep"
        p = Part("", 8, bed=[(c.base, s)], sweep=last is not None and label != last)
        if label == "keep" or not best:
            if frac >= 0.1:
                p.vocals = [(c.base, s, 1.0, 0.0)]
            p.label = f"Keep {c.B.name} vocals" if frac >= 0.1 else "Instrumental"
            kept += 1
        else:
            _, j, h = best
            c.mark(j, h, 8)
            p.vocals = [(j, h, 1.0, 0.0)]
            p.label = f"{'Add' if label == 'add' else 'Swap in'} {c.tracks[j].name} vocals"
            kept = 0
        parts.append(p)
        last, s = label, s + 8
    parts.append(Part("Outro", 4, bed=[(c.base, s)], fade=True))
    return parts


# --- rave.dj -----------------------------------------------------------------------------------------

def ravedj(c):
    """The first song is the timeline and plays from its intro. Every 4 bars: its own vocal, a guest
    vocal, or nobody. A Viterbi search picks the sequence: hearing the base and a well-fitting guest
    both score, a guest chorus over the base's chorus scores extra, and switching costs (more off the
    8-bar grid). A random seed nudges the scores, so every run is a different take."""
    start, intro = c.intro()
    length = min(c.total, c.B.nbars() - 1 - start, int(270 * audio.SR / c.bar_len))
    slots = list(range(start + intro, start + length - 4, 4))
    if not slots:
        return versatile(c, PRESETS["earworm"])
    others = [i for i in c.order if i != c.base]
    noise = lambda: float(c.rng.normal(0, 0.12))

    guests, fits = [], []
    for s in slots:
        frac = c.B.active[s:s + 4].mean()
        in_chorus = any(0 <= s - g < 8 for g in c.B.choruses)
        best = None
        for j in others:
            t = c.tracks[j]
            for h in c.positions(j, 4):
                if t.active[h:h + 4].mean() < 0.5:
                    continue
                f = fit(t, h, c.B, s, 4) + (0.3 if in_chorus and any(0 <= h - g < 8 for g in t.choruses)
                                            else 0.0)
                if best is None or f > best[0]:
                    best = (f, j, h)
        guests.append(best)
        fits.append(best[0] if best else np.nan)
    # Guest fits are judged against each other, so a guest gets the slots where it fits best.
    f = np.array(fits)
    zf = (f - np.nanmean(f)) / (np.nanstd(f) + 1e-9) if np.isfinite(f).any() else f
    scores = []
    for k, s in enumerate(slots):
        frac = c.B.active[s:s + 4].mean()
        scores.append([0.6 + 0.4 * frac + noise() if frac > 0.2 else -1.0,           # base sings
                       0.8 + 0.25 * zf[k] + noise() if np.isfinite(zf[k]) else -1.0,  # guest sings
                       0.2 + noise()])                                                  # instrumental

    # Viterbi over (base, guest, none) with a switching cost.
    S = np.array(scores)
    cost = np.full((len(slots), 3), -np.inf)
    back = np.zeros((len(slots), 3), dtype=int)
    cost[0] = S[0]
    for k in range(1, len(slots)):
        sw = 0.25 if (slots[k] - start) % 8 == 0 else 0.45
        for a in range(3):
            prev = cost[k - 1] - sw * (np.arange(3) != a)
            back[k, a] = int(np.argmax(prev))
            cost[k, a] = prev[back[k, a]] + S[k, a]
    path = [int(np.argmax(cost[-1]))]
    for k in range(len(slots) - 1, 0, -1):
        path.append(back[k, path[-1]])
    path.reverse()

    parts = [Part(f"Intro (seed {c.seed})", intro, bed=[(c.base, start)])]
    prev = None
    for k, (s, state) in enumerate(zip(slots, path)):
        p = Part("", 4, bed=[(c.base, s)])
        if state == 0:
            p.vocals, p.label = [(c.base, s, 1.0, 0.0)], f"{c.B.name}"
        elif state == 1:
            _, j, h = guests[k]
            if prev and prev[0] == j and c.room(j, prev[1] + 4, 4) and \
                    c.tracks[j].active[prev[1] + 4:prev[1] + 8].mean() >= 0.5:
                h = prev[1] + 4                        # keep a guest's line going rather than jump
            c.mark(j, h, 4)
            prev = (j, h)
            p.vocals, p.label = [(j, h, 1.0, 0.0)], f"{c.tracks[j].name} over {c.B.name}"
        else:
            p.label = "Instrumental"
        if state != 1:
            prev = None
        parts.append(p)
    parts.append(Part("Outro", 4, bed=[(c.base, slots[-1] + 4)], fade=True))
    return _merge(parts)


def _merge(parts):
    """Join consecutive parts with the same label and continuous material, for a readable cue sheet."""
    out = []
    for p in parts:
        q = out[-1] if out else None
        if q and q.label == p.label and not (p.sweep or p.riser or p.fade or q.fade) and \
                _follows(q, p):
            q.bars += p.bars
        else:
            out.append(p)
    return out


def _follows(q, p):
    def cont(a, b, w):
        return len(a) == len(b) and all(x[0] == y[0] and x[1] + w == y[1] and x[2:] == y[2:]
                                        for x, y in zip(a, b))
    return all(cont(getattr(q, f), getattr(p, f), q.bars)
               for f in ("bed", "vocals", "drums", "harm", "hooks"))


# --- Rave: the best of all of them -------------------------------------------------------------------

def layer_fit(c, a, ap, j, h, bars=4):
    """How well song j's bars h.. sit on top of song a's bars ap..: AutoMashUpper's mashability
    (chords most, then rhythm, then the two filling different frequency bands), or None if their chords
    clash outright."""
    ta, tj = c.tracks[a], c.tracks[j]
    x, y = ta.ichroma[ap:ap + bars], tj.ichroma[h:h + bars]
    if len(x) < bars or len(y) < bars:
        return None
    cos = ((x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-9)).mean()
    if cos < 0.65:
        return None
    rhythm = (ta.rhythm[ap:ap + bars] * tj.rhythm[h:h + bars]).sum(1).mean()
    overlap = np.minimum(ta.bands[ap:ap + bars], tj.bands[h:h + bars]).sum(1).mean()
    return 0.6 * cos + 0.25 * rhythm + 0.15 * (1.0 - overlap) - 0.2 * c.reuse(j, h, bars)


def plan_lines(c, bars, singers, max_len, prev=None, forward=None):
    """Fill `bars` bars with whole lyric lines, alternating `singers`, each picked to carry on from
    the line before it. Returns ([(singer, line, offset)], last (singer, line)).

    For each handoff the next singer's unused lines that fit (at most `max_len` bars, the bars left,
    and clean edges: nobody singing across the bar line where the turn starts or ends) are scored by
    the rules in lyrics.carries_on, plus a little for staying in the song's own order; the
    best few go to the chat model LocalAI has loaded, which picks the one that reads on best."""
    seq, off, k = [], 0, 0
    forward = forward if forward is not None else {}
    while off < bars:
        placed = False
        for attempt in range(len(singers)):
            sg = singers[(k + attempt) % len(singers)]
            cands = c.line_candidates(sg, bars - off)
            if not cands:
                continue
            pos = forward.get(sg, -1)

            def rule(L):
                score = lyrics.carries_on(prev[1]["words"], L["words"]) if prev else 0.0
                score -= 0.5 * max(0, L["b1"] - L["b0"] - max_len)     # prefer the section's line length
                # Dense lines: the references sing through ~84% of the track, so a line that leaves most
                # of its bars empty is a poor pick.
                span = (c.tracks[sg].bars[L["b1"]] - c.tracks[sg].bars[L["b0"]]) if L["b1"] < len(c.tracks[sg].bars) else 1
                score += 0.8 * min(1.0, (L["end"] - L["start"]) / max(1, span))
                score -= 0.8 * L.get("edge", 0.0)                       # and lines that lift out cleanly
                return score + (0.25 if L["b0"] > pos else 0.0) - 0.002 * abs(L["b0"] - pos)
            ranked = sorted(cands, key=rule, reverse=True)[:6]
            pick = None
            if prev is not None:
                pick = c.judge.pick(prev[1]["text"], [L["text"] for L in ranked])
                if pick is not None and pick != 0:
                    c.judged += 1
            L = ranked[pick if pick is not None else 0]
            c.use_line(sg, L)
            forward[sg] = L["b0"]
            seq.append((sg, L, off))
            off += L["b1"] - L["b0"]
            prev = (sg, L)
            k += attempt + 1
            placed = True
            break
        if not placed:
            break                                      # nothing fits the bars left: let it breathe
    return seq, prev


# --- The Rave engine: one engine, many styles --------------------------------------------------------
#
# What the references do (decoded: DJ Earworm's Summermash '16, minlee's Shape of You megamix; surveyed:
# 15 Earworm/Pop Danthology mashups): ONE song's instrumental carries the whole track, re-sequenced in
# long sections (Can't Stop The Feeling; Shape of You), while the vocals are a collage of short lines
# from every song, back to back (singing through 76-88% of the track), with a refrain that keeps coming
# back ("welcome to my house"), steady energy to the end, a fairly narrow stereo image, and no long
# fade. The styles differ in how long the lines are, whether there's a refrain, how the finale goes.

PRESETS = {
    "earworm": dict(length=240, block=8, max_len=4, refrain_every=16, hooks=True, beat_only_tail=0,
                    acappella=False, final="build", final_bars=16, outro=8, pan=0.15, line_pan=0.12),
    # Riff-offs are a cappella: the references stay low (voices over just the beat) and build to a
    # peak ~80% through, so the first 60% runs over the drums only, then the band comes in.
    "riffoff": dict(length=270, block=8, max_len=4, refrain_every=0, hooks=False, beat_only_tail=0,
                    breakdowns=[(0.0, 0.55, "beat")], acappella=True, final="stack", final_bars=16, outro=8,
                    pan=0.2, intro=0, line_pan=0.15, bgv=0.2),
    # Megamixes dip once, ~55-70% through (a breakdown without drums), before the last run of hooks.
    "megamix": dict(length=330, block=8, max_len=2, refrain_every=8, hooks=False, beat_only_tail=0,
                    breakdowns=[(0.0, 0.07, "drop"), (0.55, 0.68, "drop")], acappella=False, final="anchors",
                    final_bars=8, outro=8, pan=0.1, intro=0, line_pan=0.2),
}


def versatile(c, p):
    """Arrange with preset `p` (see PRESETS):

    intro      the backbone's own bars before its first verse;
    body       8-bar blocks: whole lyric lines from all the songs in turn, each chosen to carry on
               from the one before (lyrics.carries_on, plus LocalAI's loaded chat model as judge), with
               a refrain line at the top of every `refrain_every` bars; any bars no line fits are filled
               with the next song's verse; under each block, the backbone's 8 bars whose chords fit those
               lines best (preferring its fuller sections, never two backbones at once). The riff-off
               runs its last `beat_only_tail` bars over just the backbone's drums.
    finale     "build": the backbone's chorus, the others joining one at a time, then all of them;
               "stack": an a cappella hit and every chorus at once; "anchors": the refrain hooks in
               quick rotation over the backbone's chorus;
    outro      a few backbone bars, fading only over the last two.
    """
    bar_sec = c.bar_len / audio.SR
    c.opts.bgv = p.get("bgv", c.opts.bgv)              # riff-offs leave more room between the groups
    length = c.opts.length if c.opts.length and c.opts.length > 0 else p["length"]
    total = max(40, int(round(length / bar_sec)))
    B, base = c.B, c.base
    pans = np.linspace(-p["pan"], p["pan"], c.n)
    start, intro = c.intro()
    if p.get("intro") is not None:                     # megamixes start singing at once
        start, intro = start + intro - p["intro"], p["intro"]
    st = {"bed": None, "drum": start + intro, "singer": 0, "prev": None, "forward": {}}
    energy = _z_levels(B)

    def backbone(lead, bars):
        """The base's `bars` bars (start bar) under the lines `lead` [(singer, bar, offset, length)]."""
        best = None
        for pos in c.positions(base, bars):
            f = np.mean([fit(c.tracks[s], v, B, pos + o, n) for s, v, o, n in lead]) if lead else 0.0
            sc = f + 0.08 * (st["bed"] == pos) - 0.15 * c.reuse(base, pos, bars) + \
                0.1 * float(np.mean(energy[pos:pos + bars]))
            if best is None or sc > best[0]:
                best = (sc, pos)
        pos = best[1] if best else B.grid[0]
        c.mark(base, pos, bars)
        st["bed"] = pos + bars
        return pos

    def hook_over(pos, skip):
        if not p["hooks"]:
            return None
        best = None
        for j in c.order:
            if j == base or j in skip:
                continue
            m = layer_fit(c, base, pos, j, c.tracks[j].hook)
            if m is not None and (best is None or m > best[0]):
                best = (m, j)
        return (best[1], c.tracks[best[1]].hook, float(pans[c.order.index(best[1])])) if best else None

    def block(label, bars, vocals, beat_only=False, sweep=False, drop=False):
        """Music under `vocals` [(singer, bar, gain, pan, offset, length, name, span)], split into parts
        wherever a line starts or ends."""
        lead = [(x[0], x[1], x[4], x[5]) for x in vocals if x[2] >= 0.99]
        if beat_only:
            pos = st["drum"] if c.room(base, st["drum"], bars) else B.grid[0]
            st["drum"] = pos + bars
            hook = None
        else:
            pos = backbone(lead, bars)
            st["drum"] = pos + bars
            hook = hook_over(pos, {x[0] for x in vocals})
        cuts = sorted({0, bars} | set(range(4, bars, 4)) | {x[4] for x in vocals if 0 < x[4] < bars} |
                      {x[4] + x[5] for x in vocals if 0 < x[4] + x[5] < bars})
        out = []
        for a0, b0 in zip(cuts[:-1], cuts[1:]):
            here = [x for x in vocals if x[4] <= a0 and b0 <= x[4] + x[5]]
            name = " / ".join(dict.fromkeys(x[6] for x in here)) or "instrumental"
            q = Part(f"{label}: {name}", b0 - a0, sweep=sweep and a0 == 0)
            if beat_only:
                q.drums = [(base, pos + a0)]
            elif drop:
                q.harm = [(base, pos + a0, 1.0)]              # a breakdown: the backing without drums
            else:
                q.bed = [(base, pos + a0)]
                if hook:
                    q.hooks = [(hook[0], hook[1] + a0 % 4, 0.35, hook[2])]
            q.vocals = [(x[0], x[1] + (a0 - x[4]), x[2], x[3]) for x in here]
            q.spans = [(x[0], *x[7]) for x in here if x[7] and x[4] == a0]
            out.append(q)
        return out

    refrain = _refrain_line(c) if p["refrain_every"] else None

    def fill(bars, max_len):
        """Vocals for `bars` bars: the refrain if it's due, lyric lines handed on from one song to the
        next, and the next song's verse bars in any gap."""
        vocals, off = [], 0
        if refrain and st.get("blocks", 0) % max(1, p["refrain_every"] // p["block"]) == 1:
            sg, L = refrain
            n = min(L["b1"] - L["b0"], bars)
            vocals.append((sg, L["b0"], 1.0, 0.0, 0, n, c.tracks[sg].name, (L["start"], L["end"])))
            off = n
            st["prev"] = (sg, L)
        st["blocks"] = st.get("blocks", 0) + 1
        singers = [c.order[(st["singer"] + m) % c.n] for m in range(c.n)]
        seq, st["prev"] = plan_lines(c, bars - off, singers, max_len, st["prev"], st["forward"])
        st["singer"] += 1
        for sg, L, o in seq:
            st["side"] = -st.get("side", 1.0)
            vocals.append((sg, L["b0"], 1.0, p.get("line_pan", 0.0) * st["side"], off + o, L["b1"] - L["b0"],
                           c.tracks[sg].name, (L["start"], L["end"])))
        used = off + sum(L["b1"] - L["b0"] for _, L, _ in seq)
        k = 0
        while used < bars:                             # no line fits: the next song's verse
            sg = c.order[(st["singer"] + k) % c.n]
            n = min(max_len, bars - used)
            v = c.lines[sg].take(n)
            vocals.append((sg, v, 1.0, 0.0, used, n, c.tracks[sg].name, None))
            used += n
            k += 1
        return vocals

    parts = [Part("Intro", intro, bed=[(base, start)])] if intro else []
    finale = p["final_bars"] + (1 if p["acappella"] else 0)
    body = max(p["block"], total - intro - finale - p["outro"])
    body -= body % 4
    k = 0
    while k < body:
        n = min(p["block"], body - k)
        frac = (intro + k) / total
        bd = next((b for b in p.get("breakdowns", []) if b[0] <= frac < b[1]), None)
        beat_only = (p["beat_only_tail"] and k >= body - p["beat_only_tail"]) or (bd is not None and bd[2] == "beat")
        drop = bd is not None and bd[2] == "drop"
        label = "Riff-off" if beat_only else "Breakdown" if drop else ("Verse" if k < body / 2 else "Build")
        was = parts[-1].label.split(":")[0] if parts else ""
        parts += block(label, n, fill(n, p["max_len"] if not beat_only else 2), beat_only=beat_only,
                       sweep=label != was and label in ("Build", "Verse") and was in ("Riff-off", "Breakdown"),
                       drop=drop)
        k += n

    g = 1.25 / np.sqrt(c.n)
    others = [i for i in c.order if i != base]
    stack_pans = {i: float(pans[k]) for k, i in enumerate(c.order)}
    if p["acappella"]:
        parts.append(Part("A cappella hit", 1, vocals=[(i, c.tracks[i].chorus, g, stack_pans[i]) for i in c.order],
                          riser=True))
    if p["final"] == "build":
        steps = [(0, 2), (2, 2), (4, 4)]
        for m, (o, n) in enumerate(steps):
            vs = [(base, B.chorus + o, 1.0, 0.0)] + [(i, c.tracks[i].chorus + o, g, stack_pans[i])
                                                     for x, i in enumerate(others) if x <= m]
            parts.append(Part(f"Final chorus: {len(vs)} voices", n, bed=[(base, B.chorus + o)], vocals=vs,
                              riser=m == 0))
        again = c.chorus_again(base)
        vs = [(base, again, 1.0, 0.0)] + [(i, c.chorus_again(i), g, stack_pans[i]) for i in others]
        parts.append(Part("Final chorus: everyone", p["final_bars"] - 8, bed=[(base, again)], vocals=vs))
        after = again + p["final_bars"] - 8
    elif p["final"] == "stack":
        vs = [(i, c.tracks[i].chorus + (1 if p["acappella"] else 0), g, stack_pans[i]) for i in c.order]
        parts.append(Part("Finale: every chorus", p["final_bars"] // 2, bed=[(base, B.chorus + 1)], vocals=vs))
        again = c.chorus_again(base)
        vs = [(i, c.chorus_again(i), g, stack_pans[i]) for i in c.order]
        parts.append(Part("Finale: every chorus (again)", p["final_bars"] // 2, bed=[(base, again)], vocals=vs))
        after = again + p["final_bars"] // 2
    else:                                              # anchors: the best hooks, one bar each, in turn
        anchors = [(i, L) for i, L in ((i, _refrain_line(c, i)) for i in c.order) if L]
        for m in range(p["final_bars"]):
            i, L = anchors[m % len(anchors)] if anchors else (base, None)
            vs = [(i, L["b0"], 1.0, 0.0)] if L else []
            q = Part(f"Finale: {c.tracks[i].name}", 1, bed=[(base, B.chorus + m)], vocals=vs, riser=m == 0)
            if L:
                q.spans = [(i, L["start"], L["end"])]
            parts.append(q)
        after = B.chorus + p["final_bars"]
    out_at = c.outro_bar(after)
    o = p["outro"]
    if o > 4:
        parts.append(Part("Outro", o - 4, harm=[(base, out_at, 1.0)]))
        parts.append(Part("Outro", 2, bed=[(base, out_at + o - 4)]))
    elif o > 2:
        parts.append(Part("Outro", o - 2, bed=[(base, out_at)]))
    parts.append(Part("Outro", 2, bed=[(base, out_at + max(0, o - 2))], fade=True))
    return parts


def _z_levels(t):
    lv = np.log(t.bed_level + 1e-6)
    return (lv - lv.mean()) / (lv.std() + 1e-9)


def _refrain_line(c, prefer=None):
    """A catchy line to keep coming back to: the first clean, short line of a chorus, from `prefer`, or
    from the first song after the base (the backbone's own chorus is saved for the finale)."""
    order = [prefer] if prefer is not None else [i for i in c.order if i != c.base] + [c.base]
    for i in order:
        t = c.tracks[i]
        for L in t.lines or []:
            if t.chorus <= L["b0"] < t.chorus + 8 and L["b1"] - L["b0"] <= 2 and len(L["words"]) >= 2 \
                    and L.get("edge", 0) < 1.0:
                return (i, L) if prefer is None else L
    return None if prefer is None else None


def _preset(key):
    return lambda c: versatile(c, PRESETS[key])


STYLES = {s.key: s for s in [
    Style("earworm", "DJ Earworm",
          "One song's instrumental all the way through, and a collage of lines from all of them on top, "
          "with a refrain that keeps coming back and a final chorus that builds up voice by voice.",
          _preset("earworm"), key_mode="least_damage"),
    Style("riffoff", "Pitch Perfect riff-off",
          "Lines handed from song to song, then quick cut-ins over just the beat, an a cappella hit, and "
          "every chorus at once.", _preset("riffoff"), key_mode="least_damage"),
    Style("megamix", "Megamix",
          "Pop Danthology / minlee: one beat, a one- or two-bar hook from every song in turn, the best "
          "hooks coming back again and again.", _preset("megamix"), key_mode="least_damage"),
    Style("classic", "DJ Earworm (classic)",
          "The earlier arrangement: 4-bar lines traded between songs over harmony picked per section, "
          "and a final chorus that builds voice by voice.", earworm, key_mode="least_damage"),
    Style("swap", "Vocal swap",
          "The base song plays as itself; its vocals are kept, swapped for another song's, or added where "
          "it has none.", swap),
    Style("automashupper", "AutoMashUpper",
          "The base song plays as itself, with the most mashable bits of the others layered on, phrase "
          "by phrase.", automashupper),
    Style("ravedj", "rave.dj",
          "The first song is the timeline; a search decides who sings every 4 bars. Each run is a "
          "different take.", ravedj, base_first=True),
]}
STYLES["danthology"] = STYLES["megamix"]          # old name


def get(key):
    return STYLES.get(key or "earworm", STYLES["earworm"])
