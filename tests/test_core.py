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


def test_clean_grid_straightens_jitter_and_fills_gaps():
    rng = np.random.default_rng(1)
    true = 0.31 + 0.5 * np.arange(200)
    tracked = np.delete(true + rng.normal(0, 0.01, len(true)), [50, 51, 120])   # jitter + dropped beats
    beats, steady = an.clean_grid(tracked, 100.5)
    assert steady
    assert np.allclose(np.diff(beats), 0.5, atol=1e-3)
    assert np.abs(beats[:200] - true[:len(beats[:200])]).max() < 0.01


def test_clean_grid_keeps_drifting_tempo():
    beats = np.cumsum(np.linspace(0.45, 0.55, 300))     # slows down steadily
    out, steady = an.clean_grid(beats, beats[-1] + 1)
    assert not steady
    assert np.allclose(out, beats)


def test_downbeat_phase_from_votes():
    beats = 0.5 * np.arange(64)
    assert an.downbeat_phase_from(beats, beats[2::4] + 0.01) == 2


def test_fold_keeps_downbeats():
    beats = np.arange(16) * 1.0
    nb, ph = an.fold(beats, 1, 2.0)
    assert nb[ph] in beats[1::4] or (nb[ph] - 0.5) in beats
    hb, ph = an.fold(beats, 1, 0.5)
    assert hb[ph] in beats[1::4]


def test_stem_roles_from_file_names():
    from rave.stempack import role_of
    cases = {
        "Song - Acapella.wav": "vocals", "Lead Vox.wav": "vocals", "BGV 2.wav": "vocals",
        "Ad Libs.wav": "vocals", "Instrumental.mp3": "inst", "Song (No Vocals).flac": "inst",
        "Kick.wav": "drums", "Snare_01.wav": "drums", "HiHats.wav": "drums", "Perc Loop.wav": "drums",
        "808.wav": "bass", "Sub Bass.wav": "bass", "Synth Pad.wav": "other", "Piano.wav": "other",
        "Guitar L.wav": "other", "Master.wav": "mix", "Full Mix.wav": "mix",
    }
    for name, role in cases.items():
        assert role_of(name) == role, (name, role_of(name))


def test_binaural_places_sound_to_the_side():
    from rave import audio
    n = audio.SR
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, n)).astype(np.float32) * 0.1
    right = mix.binaural(x, np.full(n, np.pi / 2))            # hard right
    front = mix.binaural(x, np.zeros(n))
    l, r = audio.rms(right[0]), audio.rms(right[1])
    assert r > 1.8 * l                                        # louder in the right ear
    lag = np.argmax(np.correlate(right[0, 2000:6000], right[1, 2000:6000], "full")) - 3999
    assert 20 <= lag <= 35                                    # left ear ~0.66 ms later (~29 samples)
    assert abs(audio.rms(front[0]) - audio.rms(front[1])) < 0.02 * audio.rms(front[0])


def test_gap_fill_only_between_lead_lines():
    from rave import audio
    n = 3 * audio.SR
    lead = np.zeros((2, n), dtype=np.float32)
    lead[:, audio.SR:2 * audio.SR] = 0.3                     # the lead sings from 1 s to 2 s
    fill = np.full((2, n), 0.1, dtype=np.float32)
    out = mix.gap_fill(fill, lead)
    sr = audio.SR
    assert np.abs(out[0, int(0.3 * sr):int(0.6 * sr)]).mean() > 0.08   # open well before the lead
    assert np.abs(out[0, int(0.95 * sr):int(1.9 * sr)]).max() < 0.02   # shut just before and during
    assert np.abs(out[0, int(2.5 * sr):int(2.9 * sr)]).mean() > 0.08   # open again after
