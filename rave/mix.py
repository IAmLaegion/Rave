"""Mixing helpers: placement with crossfades, EQ, panning, ducking, riser, limiter."""
import numpy as np
from scipy import ndimage, signal

from . import audio


def ms(t):
    return int(round(t * audio.SR / 1000.0))


def place(dst, src, s, d, length, pre=0, post=0, fade_in=0, fade_out=0, gain=1.0, pan=0.0):
    """Mix src[:, s-pre : s+length+post] into dst at d-pre, with linear fades over the first `fade_in`
    and last `fade_out` samples. Linear fades of the same signal sum to unity at bar joints."""
    a, b = s - pre, s + length + post
    da = d - pre
    # Clip to what exists in both buffers.
    lo = max(0, -a, -da)
    hi = min(b, src.shape[1]) - a
    hi = min(hi, dst.shape[1] - da)
    if hi <= lo:
        return
    chunk = src[:, a + lo:a + hi].astype(np.float32, copy=True)
    n = b - a
    fade_in = min(fade_in, n)
    fade_out = min(fade_out, n - fade_in)
    env = np.ones(n, dtype=np.float32)
    if fade_in > 0:
        env[:fade_in] = np.linspace(0.0, 1.0, fade_in, endpoint=False, dtype=np.float32)
    if fade_out > 0:
        env[n - fade_out:] *= np.linspace(1.0, 0.0, fade_out, endpoint=False, dtype=np.float32)
    chunk *= env[lo:hi] * gain
    if pan:
        chunk = panned(chunk, pan)
    dst[:, da + lo:da + hi] += chunk


def panned(y, pan):
    """Constant-power pan of a stereo signal, pan in -1 (left) .. 1 (right); 0 is unchanged."""
    theta = (pan + 1.0) * np.pi / 4.0
    gl, gr = np.cos(theta) * np.sqrt(2.0), np.sin(theta) * np.sqrt(2.0)
    return np.stack([y[0] * gl, y[1] * gr])


def highpass(y, hz, order=2):
    sos = signal.butter(order, hz, btype="highpass", fs=audio.SR, output="sos")
    return signal.sosfilt(sos, y, axis=-1).astype(np.float32)


def envelope(y, smooth_ms=80):
    """Smoothed amplitude envelope of a (2, n) signal."""
    e = np.abs(y).max(axis=0)
    return ndimage.uniform_filter1d(ndimage.maximum_filter1d(e, ms(10)), ms(smooth_ms))


def duck(bed, vocals, depth_db=3.0):
    """Sidechain-style: pull the bed down by up to `depth_db` while vocals are singing."""
    env = envelope(vocals, 150)
    ref = np.percentile(env[env > 1e-4], 90) if np.any(env > 1e-4) else 1.0
    amount = np.clip(env / (ref + 1e-9), 0.0, 1.0)
    gain = audio.undb(-depth_db * amount).astype(np.float32)
    return bed * gain


def riser(n, peak=0.18, seed=7):
    """A white-noise sweep that builds over `n` samples into the finale."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((2, n)).astype(np.float32)
    out = np.zeros_like(noise)
    block = 1024
    cutoffs = np.geomspace(400.0, 9000.0, max(1, n // block + 1))
    zi = None
    for i, c in enumerate(cutoffs):
        sl = slice(i * block, min(n, (i + 1) * block))
        if sl.start >= n:
            break
        sos = signal.butter(2, [c * 0.5, min(c * 1.6, audio.SR / 2 - 100)], btype="bandpass",
                            fs=audio.SR, output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2, 2))
        out[:, sl], zi = signal.sosfilt(sos, noise[:, sl], axis=-1, zi=zi)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32) ** 2.5
    return out / (np.abs(out).max() + 1e-9) * ramp * peak


def normalize_rms(y, target_db, mask=None):
    """Scale so the RMS (over `mask`ed samples, if given) hits target_db."""
    sel = y[:, mask] if mask is not None and mask.any() else y
    current = audio.rms(sel)
    if current < 1e-6:
        return y
    return y * audio.undb(target_db - audio.db(current))


def limit(y, ceiling_db=-1.0, window_ms=30.0):
    """Look-ahead brickwall limiter. The gain is the windowed minimum of the gain each sample needs,
    averaged over the same centred window: every window in the average contains the sample, so the
    smoothed gain never exceeds what that sample needs, and gain changes are ramps, not clicks."""
    thr = audio.undb(ceiling_db)
    peak = np.abs(y).max(axis=0)
    need = np.minimum(1.0, thr / np.maximum(peak, 1e-9))
    w = max(1, ms(window_ms))
    g = ndimage.uniform_filter1d(ndimage.minimum_filter1d(need, w), w)
    return np.clip(y * g.astype(np.float32), -thr, thr)


def fade_out(y, n):
    n = min(n, y.shape[1])
    if n > 0:
        y[:, -n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return y


def lowpass_sweep(y, start, n, f0=2000.0, f1=18000.0, block=512):
    """Open a low-pass filter on y[:, start:start+n] from f0 up to f1 on a log scale, in place (the
    transition sweep from Spotify's mashup patent)."""
    end = min(y.shape[1], start + n)
    if end - start < block:
        return y
    cut = np.geomspace(f0, min(f1, audio.SR / 2 - 500), (end - start) // block + 1)
    zi = None
    for k, fc in enumerate(cut):
        sl = slice(start + k * block, min(end, start + (k + 1) * block))
        if sl.start >= end:
            break
        sos = signal.butter(2, fc, btype="lowpass", fs=audio.SR, output="sos")
        if zi is None:
            zi = signal.sosfilt_zi(sos)[:, None, :] * y[:, sl.start][None, :, None]
        y[:, sl], zi = signal.sosfilt(sos, y[:, sl], axis=-1, zi=zi)
    return y


def nightcore(y, speed):
    """Nightcore: play the whole mix faster by resampling, so tempo and pitch rise together (1.25x is
    about +3.9 semitones), the way nightcore channels do it, not a pitch-preserving stretch."""
    from fractions import Fraction
    f = Fraction(speed).limit_denominator(200)
    y = signal.resample_poly(y, f.denominator, f.numerator, axis=-1)
    # Speeding up thins the kick and bass; nightcore makers put back ~+3 dB below ~120 Hz.
    low = signal.sosfilt(signal.butter(2, 120, btype="lowpass", fs=audio.SR, output="sos"), y, axis=-1)
    return (y + (audio.undb(3.0) - 1.0) * low).astype(np.float32)


# --- 8D / binaural ---------------------------------------------------------------------------------

HEAD_R, SOUND_C = 0.0875, 343.0          # head radius (m), speed of sound (m/s): max ITD ~0.66 ms


def binaural(y, azimuth, block=512):
    """Place a sound at a moving azimuth around the listener's head (radians, 0 = front, +pi/2 = right),
    given per sample. A spherical-head model: the far ear hears it later (Woodworth ITD), quieter and
    duller (head shadow), and from behind both ears lose a little treble (the pinna's front/back cue).
    For headphones."""
    x = audio.mono(y).astype(np.float64)
    n = len(x)
    az = np.asarray(azimuth, dtype=np.float64)
    lat = np.arcsin(np.clip(np.sin(az), -1, 1))           # how far to the side, -pi/2..pi/2
    itd = HEAD_R / SOUND_C * (np.abs(lat) + np.sin(np.abs(lat))) * audio.SR   # samples
    t = np.arange(n, dtype=np.float64)
    right_far = lat < 0                                  # source on the left: right ear is far
    dl = np.where(right_far, 0.0, itd)                   # delay of each ear
    dr = np.where(right_far, itd, 0.0)
    # Front/back: the ear canal's ~3.5 kHz resonance stands out from the front and dips from behind,
    # and the pinna notches ~9 kHz for sounds behind. Built from two fixed bands added per ear.
    presence = signal.sosfilt(signal.butter(2, [2800, 4500], btype="bandpass", fs=audio.SR, output="sos"), x)
    notch = signal.sosfilt(signal.butter(2, [7500, 10500], btype="bandpass", fs=audio.SR, output="sos"), x)
    front = np.cos(az)
    x = x + presence * np.where(front > 0, 0.41 * front, 0.29 * front) - notch * 0.5 * np.clip(-front, 0, 1)
    left = np.interp(t - dl, t, x, left=0.0)
    right = np.interp(t - dr, t, x, left=0.0)
    side = np.abs(np.sin(lat))
    back = np.clip(-np.cos(az), 0, 1)
    out = np.zeros((2, n))
    zi = [None, None]
    for k in range(0, n, block):
        sl = slice(k, min(n, k + block))
        s, b = float(side[k]), float(back[k])
        for ch, (sig, far) in enumerate(((left, not right_far[k]), (right, bool(right_far[k])))):
            # The far ear: up to 6 dB quieter and low-passed down to ~1.8 kHz at full side.
            far_amt = s if far else 0.0
            near_amt = s if not far else 0.0
            cut = 20000.0 * (1 - far_amt) + 1800.0 * far_amt
            cut = min(cut, 18000.0 - 9000.0 * b)                         # behind: a little duller
            sos = signal.butter(1, min(cut, audio.SR / 2 - 100), btype="lowpass", fs=audio.SR, output="sos")
            if zi[ch] is None:
                zi[ch] = signal.sosfilt_zi(sos) * sig[sl.start]
            seg, zi[ch] = signal.sosfilt(sos, sig[sl], zi=zi[ch])
            gain = audio.undb(-6.0 * far_amt + 1.5 * near_amt - 1.5 * b)
            out[ch, sl] = seg * gain
    return out.astype(np.float32)


def orbit_angle(n, period, start=0.0, direction=1.0, sway=None):
    """Per-sample azimuth for a full orbit every `period` seconds, or (with `sway`, radians) a slow
    swing between -sway and +sway across the front."""
    t = np.arange(n) / audio.SR
    if sway is not None:
        return sway * np.sin(2 * np.pi * t / period + start)
    return start + direction * 2 * np.pi * t / period


def room(y, seconds=1.1, wet_db=-15.0, seed=11):
    """A small, soft room (decaying, decorrelated noise impulse responses), so headphone sound seems
    to come from outside the head. Returns only the reverb."""
    rng = np.random.default_rng(seed)
    n = int(seconds * audio.SR)
    env = np.exp(-np.arange(n) / (seconds * audio.SR / 6.9))   # -60 dB at `seconds`
    ir = rng.standard_normal((2, n)) * env
    ir = np.stack([signal.sosfilt(signal.butter(2, 6000, fs=audio.SR, output="sos"), ch) for ch in ir])
    ir[:, :int(0.012 * audio.SR)] = 0                          # pre-delay
    ir /= np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True)) + 1e-12
    m = audio.mono(y)
    wet = np.stack([signal.fftconvolve(m, ir[c])[:len(m)] for c in range(2)])
    return (wet * audio.undb(wet_db)).astype(np.float32)


def bandpass(y, lo, hi, order=2):
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=audio.SR, output="sos")
    return signal.sosfilt(sos, y, axis=-1).astype(np.float32)


def gap_fill(fill, lead, open_db=-30.0, look_ms=120, hold_ms=200):
    """`fill` only where `lead` is silent: a gate that closes ~120 ms before the lead starts singing
    (looking ahead) and opens again ~200 ms after it stops, with smooth ramps."""
    env = envelope(lead, 30)
    thr = audio.undb(open_db) * (np.percentile(env[env > 1e-5], 95) if np.any(env > 1e-5) else 1.0)
    busy = (env > thr).astype(np.float32)
    # Look ahead and hold: busy if the lead sings within the next look_ms or the last hold_ms.
    busy = ndimage.maximum_filter1d(busy, ms(look_ms + hold_ms), origin=-(ms(look_ms) - ms(hold_ms)) // 2)
    gate = ndimage.uniform_filter1d(1.0 - busy, ms(60))
    return (fill * gate).astype(np.float32)
