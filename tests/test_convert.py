import numpy as np
import pytest
import soundfile as sf

from core import SAMPLE_RATE, convert, load_audio


def test_convert_mono_same_rate_returns_float32_copy(speech):
    out = convert(speech, SAMPLE_RATE)
    assert out.dtype == np.float32
    assert out is not speech
    np.testing.assert_array_equal(out, speech)


def test_convert_averages_channels():
    stereo = np.array([[1.0, 0.0], [0.5, 0.5], [-1.0, 1.0]])
    np.testing.assert_allclose(convert(stereo, SAMPLE_RATE), [0.5, 0.5, 0.0])


def test_convert_resamples():
    x = np.zeros(44_100, dtype=np.float32)
    assert abs(len(convert(x, 44_100)) - SAMPLE_RATE) <= 1


def test_convert_accepts_list():
    out = convert([0.1, -0.2, 0.3], SAMPLE_RATE)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.1, -0.2, 0.3], rtol=1e-6)


def test_convert_does_not_modify_input():
    stereo = np.ones((100, 2), dtype=np.float64)
    before = stereo.copy()
    convert(stereo, 8000)
    np.testing.assert_array_equal(stereo, before)


@pytest.mark.parametrize(
    "audio, error",
    [
        (np.zeros((2, 2, 2)), ValueError),
        (np.zeros(10, dtype=np.int16), TypeError),
        (np.float32(0.5), ValueError),
    ],
)
def test_convert_rejects_bad_input(audio, error):
    with pytest.raises(error):
        convert(audio, SAMPLE_RATE)


def test_convert_rejects_bad_sample_rate(speech):
    with pytest.raises(ValueError):
        convert(speech, 0)


def test_load_audio_roundtrip(tmp_path, speech):
    path = tmp_path / "speech.wav"
    sf.write(path, speech, SAMPLE_RATE)
    loaded = load_audio(path)
    assert loaded.dtype == np.float32
    assert loaded.shape == speech.shape
    assert np.max(np.abs(loaded - speech)) < 1e-4  # PCM_16 rounding


def test_load_audio_downmixes_and_resamples(tmp_path):
    t = np.arange(44_100) / 44_100
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.stack([tone, tone], axis=1), 44_100)
    x = load_audio(path)
    assert x.ndim == 1
    assert abs(len(x) - SAMPLE_RATE) <= 1


def test_load_audio_missing_file(tmp_path):
    with pytest.raises(sf.LibsndfileError):
        load_audio(tmp_path / "missing.wav")
