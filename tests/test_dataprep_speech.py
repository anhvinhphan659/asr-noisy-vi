import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from core import SAMPLE_RATE
from dataprep import (
    choose_dev_speakers,
    process_utterance,
    read_vivos_split,
    save_manifests,
    speaker_overlap,
    subsample,
)


def test_process_utterance_writes_kept_audio(tmp_path, speech):
    row = process_utterance(speech, 16000, tmp_path, "vivos", "train", "u1", "XIN CHÀO!",
                            speaker="s1")
    assert row["drop_reason"] is None
    assert row["text"] == "xin chào"
    assert row["path"] == "clean/vivos/train/u1.wav"
    audio, sr = sf.read(tmp_path / row["path"])
    assert sr == SAMPLE_RATE and len(audio) == len(speech)
    assert row["duration_s"] == pytest.approx(len(speech) / SAMPLE_RATE)


@pytest.mark.parametrize("audio, text, reason", [
    (np.zeros(SAMPLE_RATE, dtype=np.float32), "a", "empty_audio"),
    (np.full(SAMPLE_RATE, 0.1, dtype=np.float32), " ?! ", "empty_text"),
    (np.full(SAMPLE_RATE // 4, 0.1, dtype=np.float32), "a", "too_short"),
    (np.full(31 * SAMPLE_RATE, 0.1, dtype=np.float32), "a", "too_long"),
])
def test_process_utterance_filters(tmp_path, audio, text, reason):
    row = process_utterance(audio, 16000, tmp_path, "d", "test", "u", text)
    assert row["drop_reason"] == reason
    assert not (tmp_path / row["path"]).exists()


def test_save_manifests_splits_kept_and_dropped(tmp_path):
    rows = [{"utt_id": "a", "drop_reason": None}, {"utt_id": "b", "drop_reason": "too_long"}]
    kept = save_manifests(rows, tmp_path, "x")
    assert list(kept["utt_id"]) == ["a"] and "drop_reason" not in kept
    assert list(pd.read_csv(tmp_path / "dropped_x.csv")["utt_id"]) == ["b"]


def test_read_vivos_split(tmp_path):
    (tmp_path / "prompts.txt").write_text("SPK01_R001 XIN CHÀO\nSPK02_R001 TẠM BIỆT\n",
                                          encoding="utf-8")
    (tmp_path / "genders.txt").write_text("SPK01 m\nSPK02 f\n", encoding="utf-8")
    items, genders = read_vivos_split(tmp_path)
    assert items == [("SPK01_R001", "SPK01", "XIN CHÀO"), ("SPK02_R001", "SPK02", "TẠM BIỆT")]
    assert genders == {"SPK01": "male", "SPK02": "female"}


def test_choose_dev_speakers_is_balanced_and_reproducible():
    speakers = [f"s{i}" for i in range(10)]
    genders = {s: "male" if i % 2 else "female" for i, s in enumerate(speakers)}
    chosen = choose_dev_speakers(speakers, genders, per_gender=2, seed=1)
    assert len(chosen) == 4
    assert sorted(genders[s] for s in chosen) == ["female", "female", "male", "male"]
    assert chosen == choose_dev_speakers(speakers, genders, per_gender=2, seed=1)


def test_subsample_keeps_proportions():
    manifest = pd.DataFrame({"utt_id": range(100), "cond": ["a"] * 80 + ["b"] * 20})
    out = subsample(manifest, 10, by="cond", seed=0)
    assert len(out) == 10
    assert out["cond"].value_counts().to_dict() == {"a": 8, "b": 2}
    pd.testing.assert_frame_equal(out, subsample(manifest, 10, by="cond", seed=0))


def test_speaker_overlap_counts_shared_speakers():
    manifest = pd.DataFrame({"dataset": "d", "split": ["train", "train", "test"],
                             "speaker": ["s1", "s2", "s2"]})
    out = speaker_overlap(manifest).set_index("splits")["shared"]
    assert out["train/test"] == 1
