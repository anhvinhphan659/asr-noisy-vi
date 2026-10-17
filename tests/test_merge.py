import numpy as np
import pytest

from core import MAX_PEAK, SAMPLE_RATE, make_babble, measure_snr, mix


def power(x):
    return np.mean(np.asarray(x, dtype=np.float64) ** 2)


@pytest.mark.parametrize("snr_db", [20, 10, 5, 0, -5])
def test_mix_hits_target_snr(speech, noise, snr_db):
    out = mix(speech, noise, snr_db)
    assert out.dtype == np.float32
    assert abs(measure_snr(speech, out) - snr_db) < 0.5


def test_mix_output_length_equals_speech(speech, noise):
    assert len(mix(speech, noise, 10)) == len(speech)


def test_mix_scales_only_noise(speech, noise):
    snr_db = 10
    out = mix(speech, noise, snr_db)
    segment = noise[: len(speech)].astype(np.float64)
    gain = np.sqrt(power(speech) / (power(segment) * 10 ** (snr_db / 10)))
    np.testing.assert_allclose(out, speech + gain * segment, atol=1e-6)


def test_mix_reads_noise_from_offset(speech, noise):
    offset_s = 0.5
    start = int(offset_s * SAMPLE_RATE)
    expected = mix(speech, noise[start:], 10)
    np.testing.assert_array_equal(mix(speech, noise, 10, offset_s=offset_s), expected)


def test_mix_ignores_noise_tail(speech, noise):
    np.testing.assert_array_equal(mix(speech, noise, 10), mix(speech, noise[: len(speech)], 10))


def test_mix_prevents_clipping_and_keeps_snr(speech, noise):
    loud = speech * 4
    out = mix(loud, noise, 0)
    assert np.max(np.abs(out)) <= MAX_PEAK + 1e-6
    assert abs(measure_snr(loud, out) - 0) < 0.5


def test_mix_does_not_modify_inputs(speech, noise):
    s0, n0 = speech.copy(), noise.copy()
    mix(speech * 4, noise, 0)
    mix(speech, noise, 10, offset_s=0.5)
    np.testing.assert_array_equal(speech, s0)
    np.testing.assert_array_equal(noise, n0)


def test_mix_accepts_lists():
    out = mix([0.1, -0.1, 0.2, -0.2], [0.5, 0.5, -0.5, -0.5], 0)
    assert out.dtype == np.float32 and len(out) == 4


def test_mix_rejects_short_noise(speech, noise):
    with pytest.raises(ValueError, match="loop the noise"):
        mix(speech, noise[: len(speech) - 1], 10)
    with pytest.raises(ValueError, match="loop the noise"):
        mix(speech, noise, 10, offset_s=1.5)  # 1.5 s left, speech is 2 s


def test_mix_rejects_negative_offset(speech, noise):
    with pytest.raises(ValueError, match="offset_s"):
        mix(speech, noise, 10, offset_s=-0.1)


def test_mix_rejects_silence(speech, noise):
    with pytest.raises(ValueError, match="speech is silent"):
        mix(np.zeros_like(speech), noise, 10)
    with pytest.raises(ValueError, match="noise segment is silent"):
        mix(speech, np.zeros_like(noise), 10)


def test_make_babble_length_is_shortest_voice(rng):
    voices = [rng.standard_normal(n).astype(np.float32) for n in (8000, 5000, 12000)]
    out = make_babble(voices)
    assert out.shape == (5000,)
    assert out.dtype == np.float32


def test_make_babble_balances_voice_levels(rng):
    a, b, c = (rng.standard_normal(4000).astype(np.float32) for _ in range(3))
    np.testing.assert_allclose(make_babble([a, b, c]), make_babble([0.01 * a, 5 * b, c]), atol=1e-5)


def test_make_babble_matches_formula(rng):
    a = rng.standard_normal(1000)
    b = 0.3 * rng.standard_normal(1200)
    expected = (a / np.sqrt(power(a)) + b[:1000] / np.sqrt(power(b[:1000]))) / 2
    np.testing.assert_allclose(make_babble([a, b]), expected, atol=1e-6)


def test_make_babble_rejects_bad_input(rng):
    with pytest.raises(ValueError, match="empty"):
        make_babble([])
    with pytest.raises(ValueError, match=r"voices\[1\] is silent"):
        make_babble([rng.standard_normal(100), np.zeros(100)])
