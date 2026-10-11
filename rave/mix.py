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
