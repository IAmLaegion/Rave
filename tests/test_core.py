import numpy as np

from rave import analysis as an
from rave import mix


def test_camelot():
    assert an.camelot(0, False) == "8B"    # C major
    assert an.camelot(9, True) == "8A"     # A minor
    assert an.camelot(7, False) == "9B"    # G major
    assert an.camelot(4, True) == "9A"     # E minor


def test_key_shift_matches_relative_keys():
    assert an.key_shift(0, False, 2, False) == 2      # C -> D
    assert an.key_shift(9, True, 0, False) == 0       # Am already fits C
    assert an.key_shift(4, True, 2, False) == -5      # Em (G) -> D
    assert an.key_shift(0, False, 6, False) == -6     # tritone wraps to the down shift
    for s in range(12):
        for d in range(12):
            assert -6 <= an.key_shift(s, False, d, True) <= 5


def test_tempo_fold():
    assert an.tempo_fold(60, 120) == 2.0
    assert an.tempo_fold(170, 90) == 0.5
    assert an.tempo_fold(118, 124) == 1.0


def test_place_linear_fades_sum_to_unity():
    src = np.ones((2, 1000), dtype=np.float32)
    dst = np.zeros((2, 1000), dtype=np.float32)
    mix.place(dst, src, 100, 100, 400, pre=20, post=20, fade_in=40, fade_out=40)
    mix.place(dst, src, 500, 500, 400, pre=20, post=20, fade_in=40, fade_out=40)
    assert np.allclose(dst[:, 130:870], 1.0, atol=0.03)


def test_limiter_ceiling():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((2, 44100)).astype(np.float32)
    out = mix.limit(y, -1.0)
    assert np.abs(out).max() <= 10 ** (-1 / 20) + 1e-6


def test_link_parsing():
    from rave import sources as s
    assert s.youtube_id("https://www.youtube.com/watch?v=jNQXAC9IVRw&t=3") == "jNQXAC9IVRw"
    assert s.youtube_id("https://youtu.be/jNQXAC9IVRw?si=x") == "jNQXAC9IVRw"
    assert s.youtube_id("https://music.youtube.com/watch?v=jNQXAC9IVRw") == "jNQXAC9IVRw"
    assert s.youtube_id("https://www.youtube.com/shorts/jNQXAC9IVRw") == "jNQXAC9IVRw"
    assert s.youtube_id("https://evil.example/watch?v=jNQXAC9IVRw") is None
    assert s.youtube_id("https://www.youtube.com/watch?v=../../etc") is None
    assert s.spotify_id("https://open.spotify.com/track/39LLxExYz6ewLAcYrzQQyP?si=abc") == "39LLxExYz6ewLAcYrzQQyP"
    assert s.spotify_id("https://open.spotify.com/intl-de/track/39LLxExYz6ewLAcYrzQQyP") == "39LLxExYz6ewLAcYrzQQyP"
    assert s.spotify_id("https://open.spotify.com/album/39LLxExYz6ewLAcYrzQQyP") is None
