from dataclasses import asdict

import numpy as np
import pytest
import soundfile as sf

from core import SAMPLE_RATE, extract_info, measure_snr, mix


def write(path, x, sr=SAMPLE_RATE):
    sf.write(path, x, sr)
    return path


def test_extract_info_fields(tmp_path):
    t = np.arange(44_100) / 44_100
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    path = write(tmp_path / "stereo.wav", np.stack([tone, tone], axis=1), 44_100)
    info = extract_info(path)
    assert info.path == str(path)
    assert info.orig_sr == 44_100
    assert info.orig_channels == 2
    assert info.duration_s == pytest.approx(1.0, abs=1e-3)
    assert info.peak == pytest.approx(0.5, abs=0.01)
    assert not info.is_empty
    assert not info.has_clipping
    assert info.snr_db is None
    assert set(asdict(info)) == {
        "path", "orig_sr", "orig_channels", "duration_s",
        "peak", "is_empty", "has_clipping", "snr_db",
    }


def test_extract_info_empty(tmp_path):
    assert extract_info(write(tmp_path / "silent.wav", np.zeros(1600))).is_empty
    assert extract_info(write(tmp_path / "zero_len.wav", np.zeros(0))).is_empty


def test_extract_info_detects_clipping_saved_as_pcm16(tmp_path, speech):
    path = write(tmp_path / "clipped.wav", np.clip(speech * 10, -1, 1))
    assert extract_info(path).has_clipping


def test_extract_info_snr_with_clean_path(tmp_path, speech, noise):
    clean = write(tmp_path / "clean.wav", speech)
    noisy = write(tmp_path / "noisy.wav", mix(speech, noise, 5))
    assert abs(extract_info(noisy, clean_path=clean).snr_db - 5) < 0.5


def test_extract_info_snr_length_mismatch(tmp_path, speech, noise):
    clean = write(tmp_path / "clean.wav", speech[:-100])
    noisy = write(tmp_path / "noisy.wav", mix(speech, noise, 5))
    with pytest.raises(ValueError, match="length mismatch"):
        extract_info(noisy, clean_path=clean)


def test_measure_snr_ignores_global_gain(speech, noise):
    out = mix(speech, noise, 10)
    assert measure_snr(speech, out) == pytest.approx(10, abs=0.2)
    assert measure_snr(speech, 0.3 * out) == pytest.approx(10, abs=0.2)


def test_measure_snr_clean_signal_is_infinite(speech):
    assert measure_snr(speech, speech) == float("inf")


def test_measure_snr_rejects_bad_input(speech):
    with pytest.raises(ValueError, match="length mismatch"):
        measure_snr(speech, speech[:-1])
    with pytest.raises(ValueError, match="clean is silent"):
        measure_snr(np.zeros_like(speech), speech)
