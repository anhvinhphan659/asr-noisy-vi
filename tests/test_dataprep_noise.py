from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from core import SAMPLE_RATE, load_audio
from dataprep import (
    band_limit,
    random_offset_s,
    read_segment,
    select_fsd50k,
    select_musan_speech,
    silent_fraction,
    take_segment,
)


def _write_wav(path, seconds, sr=SAMPLE_RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, 0.1 * np.ones(int(seconds * sr)), sr)


def test_take_segment_reads_from_offset():
    noise = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(take_segment(noise, 2 / SAMPLE_RATE, 3), [2, 3, 4])


def test_take_segment_loops_short_noise():
    noise = np.arange(4, dtype=np.float32)
    np.testing.assert_array_equal(take_segment(noise, 3 / SAMPLE_RATE, 6), [3, 0, 1, 2, 3, 0])


@pytest.mark.parametrize("offset_s, seconds", [
    (0.0, 1.0),    # start of the file
    (1.25, 0.5),   # middle of the file
    (2.5, 1.2),    # runs past the end: loops back to the start
    (0.5, 7.0),    # longer than the file: loops several times
])
def test_read_segment_matches_full_load(tmp_path, rng, offset_s, seconds):
    path = tmp_path / "voice.wav"
    sf.write(path, 0.1 * rng.standard_normal(3 * SAMPLE_RATE), SAMPLE_RATE, subtype="PCM_16")
    n = int(seconds * SAMPLE_RATE)
    expected = take_segment(load_audio(path), offset_s, n)
    np.testing.assert_array_equal(read_segment(path, offset_s, n), expected)


def test_read_segment_resamples_other_rates(tmp_path):
    path = tmp_path / "voice44k.wav"
    t = np.arange(3 * 44100) / 44100
    sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), 44100)
    out = read_segment(path, 0.5, SAMPLE_RATE)
    assert out.dtype == np.float32 and len(out) == SAMPLE_RATE
    expected = take_segment(load_audio(path), 0.5, SAMPLE_RATE)
    np.testing.assert_allclose(out[100:-100], expected[100:-100], atol=1e-2)


def test_random_offset_fits_speech_when_noise_is_long(rng):
    offsets = [random_offset_s(rng, 10.0, 4.0) for _ in range(200)]
    assert 0 <= min(offsets) and max(offsets) <= 6.0


def test_band_limit_removes_high_frequencies():
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    low = np.sin(2 * np.pi * 1000 * t)
    high = np.sin(2 * np.pi * 6000 * t)
    out = band_limit((low + high).astype(np.float32), 8000)
    assert len(out) == SAMPLE_RATE
    spectrum = np.abs(np.fft.rfft(out[1000:-1000]))
    freqs = np.fft.rfftfreq(len(out) - 2000, 1 / SAMPLE_RATE)
    assert spectrum[freqs > 4500].max() < 0.01 * spectrum[np.abs(freqs - 1000) < 5].max()


def test_band_limit_is_identity_at_full_band(noise):
    np.testing.assert_array_equal(band_limit(noise, SAMPLE_RATE), noise)


def test_select_fsd50k_filters_labels_and_splits(tmp_path):
    gt = tmp_path / "FSD50K.ground_truth"
    gt.mkdir()
    pd.DataFrame({"fname": [1, 2, 3, 4], "split": "train", "mids": "",
                  "labels": ["Car,Vehicle", "Car,Speech", "Music", "Motorcycle,Vehicle"]}
                 ).to_csv(gt / "dev.csv", index=False)
    pd.DataFrame({"fname": [5], "mids": "", "labels": ["Traffic_noise_and_roadway_noise"]}
                 ).to_csv(gt / "eval.csv", index=False)
    for name, seconds in [("1", 5), ("2", 5), ("3", 5), ("4", 1)]:
        _write_wav(tmp_path / "FSD50K.dev_audio" / f"{name}.wav", seconds, 44100)
    _write_wav(tmp_path / "FSD50K.eval_audio" / "5.wav", 4, 44100)

    pool = select_fsd50k(tmp_path)
    # 2 has speech, 3 is music, 4 is shorter than 3 s.
    assert pool.set_index("noise_id")["split"].to_dict() == {"fsd50k_1": "train",
                                                             "fsd50k_5": "test"}


def test_silent_fraction_counts_digital_silence(tmp_path):
    sr = 44100
    audio = np.concatenate([0.1 * np.ones(3 * sr), np.zeros(sr)])
    sf.write(tmp_path / "a.wav", audio, sr)
    assert silent_fraction(tmp_path / "a.wav") == pytest.approx(0.25, abs=0.01)


def test_select_fsd50k_drops_clips_padded_with_silence(tmp_path):
    gt = tmp_path / "FSD50K.ground_truth"
    gt.mkdir()
    pd.DataFrame({"fname": [1, 2], "split": "train", "mids": "", "labels": "Car"}
                 ).to_csv(gt / "dev.csv", index=False)
    pd.DataFrame({"fname": [], "mids": [], "labels": []}).to_csv(gt / "eval.csv", index=False)
    sr = 44100
    _write_wav(tmp_path / "FSD50K.dev_audio" / "1.wav", 5, sr)
    # 2: traffic for 2 s, then 8 s of zeros, like fsd50k_210254.
    sf.write(tmp_path / "FSD50K.dev_audio" / "2.wav",
             np.concatenate([0.1 * np.ones(2 * sr), np.zeros(8 * sr)]), sr)

    pool = select_fsd50k(tmp_path)
    assert pool["noise_id"].tolist() == ["fsd50k_1"]
    assert len(select_fsd50k(tmp_path, max_silent_fraction=0.9)) == 2


def test_select_musan_speech_splits_by_file(tmp_path):
    folder = tmp_path / "speech" / "librivox"
    lines = []
    for i in range(10):
        _write_wav(folder / f"speech-librivox-{i:04d}.wav", 4)
        lines.append(f"speech-librivox-{i:04d} {'m' if i % 2 else 'f'} english")
    (folder / "ANNOTATIONS").write_text("\n".join(lines), encoding="utf-8")

    pool = select_musan_speech(tmp_path, test_fraction=0.4, seed=0)
    assert len(pool) == 10
    assert (pool["split"] == "test").sum() == 4
    assert pool.groupby("gender")["split"].apply(lambda s: (s == "test").sum()).tolist() == [2, 2]
    pd.testing.assert_frame_equal(pool, select_musan_speech(tmp_path, test_fraction=0.4, seed=0))


def test_take_segment_rejects_empty_noise():
    with pytest.raises(ValueError):
        take_segment(np.array([], dtype=np.float32), 0.0, 10)


# --- Downloads and materialization -------------------------------------------------------

def _fake_hub(files):
    """A stand-in for huggingface_hub.hf_hub_download serving ``{filename: bytes}``."""
    def download(repo_id, filename, repo_type=None, local_dir=None, **kwargs):
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(files[filename])
        return str(out)
    return download


def _wav_bytes(tmp_path, seconds, sr):
    path = tmp_path / "tmp.wav"
    sf.write(path, 0.1 * np.ones(int(seconds * sr)), sr)
    return path.read_bytes()


def test_download_fsd50k_traffic_fetches_only_selected_clips(tmp_path, monkeypatch):
    import huggingface_hub

    dev = "fname,labels,mids,split\n1,\"Car,Vehicle\",,train\n2,\"Car,Speech\",,train\n3,Music,,train\n"
    ev = "fname,labels,mids\n5,Motorcycle,\n"
    files = {"labels/dev.csv": dev.encode(), "labels/eval.csv": ev.encode(),
             "clips/dev/1.wav": _wav_bytes(tmp_path, 4, 44100),
             "clips/eval/5.wav": _wav_bytes(tmp_path, 4, 44100)}
    requested = []
    fake = _fake_hub(files)

    def spy(repo_id, filename, **kwargs):
        requested.append(filename)
        return fake(repo_id, filename, **kwargs)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", spy)
    from dataprep import download_fsd50k_traffic

    pool = download_fsd50k_traffic(tmp_path / "raw")
    assert pool.set_index("noise_id")["split"].to_dict() == {"fsd50k_1": "train",
                                                             "fsd50k_5": "test"}
    assert "clips/dev/2.wav" not in requested and "clips/dev/3.wav" not in requested


def test_download_musan_speech_extracts_only_wanted_files(tmp_path, monkeypatch, capsys):
    import tarfile

    import huggingface_hub

    src = tmp_path / "src" / "musan"
    for rel in ["music/fma/m.wav", "speech/librivox/speech-librivox-0000.wav",
                "speech/librivox/speech-librivox-0001.wav", "speech/us-gov/g.wav"]:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        sf.write(src / rel, 0.1 * np.ones(4 * SAMPLE_RATE), SAMPLE_RATE)
    archive = tmp_path / "musan.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(src, arcname="musan")
    annotations = b"speech-librivox-0000 m english\nspeech-librivox-0001 f english\n"
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        _fake_hub({"speech/librivox/ANNOTATIONS": annotations}))
    from dataprep import download_musan_speech

    from dataprep import set_progress_interval

    set_progress_interval(0)
    try:
        root = download_musan_speech(tmp_path / "raw", url=archive.as_uri())
    finally:
        set_progress_interval(60)
    printed = capsys.readouterr().out
    assert "MUSAN: streaming" in printed and "voices 2/2" in printed
    assert sorted(p.name for p in (root / "speech" / "librivox").glob("*.wav")) == [
        "speech-librivox-0000.wav", "speech-librivox-0001.wav"]
    assert not (root / "music").exists() and not (root / "speech" / "us-gov").exists()
    pool = select_musan_speech(root)
    assert set(pool["gender"]) == {"male", "female"}
    # Nothing left to fetch: the archive is not opened again.
    assert download_musan_speech(tmp_path / "raw", url="file:///does/not/exist") == root


def test_materialize_pool_writes_16k_files_with_relative_paths(tmp_path):
    from dataprep import materialize_pool

    src = tmp_path / "src.wav"
    sf.write(src, 0.1 * np.ones(4 * 44100), 44100)
    pool = pd.DataFrame([{"noise_id": "fsd50k_1", "source": "fsd50k", "split": "train",
                          "path": str(src), "duration_s": 4.0, "gender": None, "labels": "Car"}])
    out = materialize_pool(pool, tmp_path / "data", "traffic")
    assert out.loc[0, "path"] == "noise/traffic/fsd50k_1.wav"
    info = sf.info(tmp_path / "data" / out.loc[0, "path"])
    assert info.samplerate == SAMPLE_RATE and info.channels == 1
    assert out.loc[0, "duration_s"] == pytest.approx(4.0, abs=1e-3)
    assert pool.loc[0, "path"] == str(src)  # input unchanged
